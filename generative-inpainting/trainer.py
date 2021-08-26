import os
import torch
import torch.nn as nn
from torch import autograd
from networks import Generator, LocalDis, GlobalDis


from utils.tools import get_model_list, local_patch2, spatial_discounting_mask
from utils.logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config, mode):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        if self.use_cuda:
            self.netG.to(self.device_ids[0])

        if mode=='train':
            self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
            self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

            self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                                betas=(self.config['beta1'], self.config['beta2']))
            d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
            self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                                betas=(self.config['beta1'], self.config['beta2']))
            if self.use_cuda:
                self.localD.to(self.device_ids[0])
                self.globalD.to(self.device_ids[0])


    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x1, x2, offset_flow = self.netG(x, masks)
        # bboxes_ = bboxes.reshape(-1, 2, 4)[:, 0, :]

        local_patch_gt = local_patch2(ground_truth, bboxes, self.config['image_size'])
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)
        local_patch_x1_inpaint = local_patch2(x1_inpaint, bboxes, self.config['image_size'])
        local_patch_x2_inpaint = local_patch2(x2_inpaint, bboxes, self.config['image_size'])

        # D part
        # wgan d loss
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        global_real_pred, global_fake_pred = self.dis_forward(
            self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
            torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        local_penalty = self.calc_gradient_penalty(
            self.localD, local_patch_gt, local_patch_x2_inpaint.detach())
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth, x2_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty

        # G part
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config)
            losses['l1'] = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * \
                self.config['coarse_l1_alpha'] + \
                l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt * sd_mask)
            losses['ae'] = l1_loss(x1 * (1. - masks), ground_truth * (1. - masks)) * \
                self.config['coarse_l1_alpha'] + \
                l1_loss(x2 * (1. - masks), ground_truth * (1. - masks))

            # wgan g loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt, local_patch_x2_inpaint)
            global_real_pred, global_fake_pred = self.dis_forward(
                self.globalD, ground_truth, x2_inpaint)
            losses['wgan_g'] = - torch.mean(local_patch_fake_pred) - \
                torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        return losses, x2_inpaint, offset_flow

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    def dis_forward_glob(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output, clss_score = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)
        clss_score_real, clss_score_fake = torch.split(clss_score, batch_size, dim=0)

        return real_pred, fake_pred, clss_score_fake

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        print('Saving weights to', checkpoint_dir, '...')
        torch.save({
            'G': self.netG.state_dict(),
            'localD': self.localD.state_dict(),
            'globalD': self.globalD.state_dict(),
            'optimG': self.optimizer_g.state_dict(),
            'optimD': self.optimizer_d.state_dict(),
            'iteration': iteration
        }, os.path.join(checkpoint_dir, 'model/weights_{}.pth'.format(iteration)))

    # def save_model(self, checkpoint_dir, iteration):
    #     # Save generators, discriminators, and optimizers
    #     gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
    #     dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
    #     opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
    #     torch.save({'G':self.netG.state_dict(),
    #                 'iteration': iteration}, gen_name)
    #     torch.save({'localD': self.localD.state_dict(),
    #                 'globalD': self.globalD.state_dict()}, dis_name)
    #     torch.save({'gen': self.optimizer_g.state_dict(),
    #                 'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, test=False):
        print('Loading saved weights from', checkpoint_dir, '...')
        states = torch.load(checkpoint_dir, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.netG.load_state_dict(states['G'])
        if 'iteration' in states:
            iteration = states['iteration']

        if not test:
            if 'localD' in states:
                self.localD.load_state_dict(states['D'])
            if 'globalD' in states:
                self.globalD.load_state_dict()
            if 'optimG' in states:
                self.optimizer_g.load_state_dict(states['optimG'])
            if 'optimD' in states:
                self.optimizer_d.load_state_dict(states['optimD'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration