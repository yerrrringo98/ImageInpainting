import os
import torch
import torch.nn as nn
from torch import autograd
from model.networks import Generator, LocalDis, GlobalDis


from utils.tools import get_model_list, local_patch, spatial_discounting_mask
from utils.logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']

        self.netG = Generator(self.config['netG'], self.use_cuda, self.device_ids)
        self.localD = LocalDis(self.config['netD'], self.use_cuda, self.device_ids)
        self.globalD = GlobalDis(self.config['netD'], self.use_cuda, self.device_ids)

        self.optimizer_g = torch.optim.Adam(self.netG.parameters(), lr=self.config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        d_params = list(self.localD.parameters()) + list(self.globalD.parameters())
        self.optimizer_d = torch.optim.Adam(d_params, lr=config['lr'],
                                            betas=(self.config['beta1'], self.config['beta2']))
        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.localD.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

    def forward(self, x, bboxes, masks, ground_truth, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        ground_truth_2 = ground_truth.repeat(8, 1, 1, 1, 1).view(-1, 3, ground_truth.size()[2], ground_truth.size()[3])
        masks_2 = masks.repeat(8,1,1,1,1).view(-1,1, masks.size()[2], masks.size()[3])
        bboxes_2 =[]
        for bbox in bboxes:
            bboxes_2 += [bbox]*8

        x1, x2, offset_flow = self.netG(x, masks)
        local_patch_gt = local_patch(ground_truth, bboxes)
        x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks_2 + x.repeat(8,1,1,1,1).view(-1,3, x.size()[2], x.size()[3]) * (1. - masks_2)
        local_patch_x1_inpaint = local_patch(x1_inpaint, bboxes)
        local_patch_x2_inpaint = local_patch(x2_inpaint, bboxes_2)

        # D part
        # wgan d loss
        local_patch_gt_2 = local_patch_gt.repeat(8,1,1,1,1).view(-1,3, local_patch_gt.size()[2], local_patch_gt.size()[3])
        local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
            self.localD, local_patch_gt_2, local_patch_x2_inpaint.detach())
        global_real_pred, global_fake_pred = self.dis_forward(
            self.globalD, ground_truth_2, x2_inpaint.detach())

        d_local_loss = local_patch_fake_pred - local_patch_real_pred
        d_local_loss = d_local_loss.view(-1, 8).mean(1)
        d_global_loss = global_fake_pred - global_real_pred
        d_global_loss = d_global_loss.view(-1, 8).mean(1)
        losses['wgan_d'] = torch.mean(d_local_loss) + torch.mean(d_global_loss) * self.config['global_wgan_loss_alpha']
        # losses['wgan_d'] = torch.mean(local_patch_fake_pred - local_patch_real_pred) + \
        #     torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        local_penalty = self.calc_gradient_penalty(
            self.localD, local_patch_gt_2, local_patch_x2_inpaint.detach())
        global_penalty = self.calc_gradient_penalty(self.globalD, ground_truth_2, x2_inpaint.detach())
        losses['wgan_gp'] = local_penalty + global_penalty

        # G part
        if compute_loss_g:
            sd_mask = spatial_discounting_mask(self.config)
            sd_mask_2 = sd_mask.repeat(8,1,1,1,1).view(-1,1, sd_mask.size()[2], sd_mask.size()[3])

            l1_loss_1 = l1_loss(local_patch_x1_inpaint * sd_mask, local_patch_gt * sd_mask) * \
                self.config['coarse_l1_alpha']
            l1_loss_2 = l1_loss(local_patch_x2_inpaint * sd_mask, local_patch_gt_2 * sd_mask).mean()
            losses['l1'] = l1_loss_1 + l1_loss_2


            losses['ae'] = l1_loss(x1 * (1. - masks), ground_truth * (1. - masks)) * \
                self.config['coarse_l1_alpha'] + \
                l1_loss(x2 * (1. - masks_2), ground_truth_2 * (1. - masks_2))

            # wgan g loss
            local_patch_real_pred, local_patch_fake_pred = self.dis_forward(
                self.localD, local_patch_gt_2, local_patch_x2_inpaint)
            global_real_pred, global_fake_pred = self.dis_forward(
                self.globalD, ground_truth_2, x2_inpaint)
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
        gradients = gradients.view(12, 8, -1).mean(1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, masks):
        self.eval()
        x1, x2, offset_flow = self.netG(x, masks)
        # x1_inpaint = x1 * masks + x * (1. - masks)
        x2_inpaint = x2 * masks + x * (1. - masks)

        return x2_inpaint, offset_flow

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
        dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
        opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.netG.state_dict(), gen_name)
        torch.save({'localD': self.localD.state_dict(),
                    'globalD': self.globalD.state_dict()}, dis_name)
        torch.save({'gen': self.optimizer_g.state_dict(),
                    'dis': self.optimizer_d.state_dict()}, opt_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
        self.netG.load_state_dict(torch.load(last_model_name))
        iteration = int(last_model_name[-11:-3])

        if not test:
            # Load discriminators
            last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
            state_dict = torch.load(last_model_name)
            self.localD.load_state_dict(state_dict['localD'])
            self.globalD.load_state_dict(state_dict['globalD'])
            # Load optimizers
            state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
            self.optimizer_d.load_state_dict(state_dict['dis'])
            self.optimizer_g.load_state_dict(state_dict['gen'])

        print("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration
