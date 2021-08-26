import os,json,sys
from os import path
import random
from argparse import ArgumentParser

import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from utils.psnr import *

from utils.tools import get_config
from trainer import Trainer
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
from data.dataset import VGdataset, vg_collate_fn, test_mask

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/generative-config.yaml',help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    print("Arguments: {}".format(args))

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print("Configuration: {}".format(config))

    if not os.path.exists(config['test_result']):
        os.makedirs(config['test_result'])
        # os.makedirs(os.path.join(config['test_result'], 'image'))

    try:  # for unexpected error logging
        with open(config['vocab_path'], 'r') as f:
            vocab = json.load(f)
        test_dataset = VGdataset(vocab=vocab, h5_path=config['h5_path'],
                                  image_dir=config['image_dir'], image_size=config['image_size'], include_relationships=False)

        loader_kwargs = {
            'batch_size': config['batch_size'],
            'num_workers': config['num_workers'],
            'shuffle': False,
            'collate_fn': vg_collate_fn,
        }
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, **loader_kwargs)

        trainer = Trainer(config, mode='test')
        #Resume weight
        iteration = trainer.resume(config['test_load'], test=True)
        print("Load model from iteration: {}".format(iteration))

        for index, batch in enumerate(test_loader):
            batch = [tensor.cuda() for tensor in batch]
            ground_truth, objs, bboxes, triples, masks, obj_to_img, triple_to_img = batch
            x = ground_truth * (1. - masks)

            with torch.no_grad():
                # Inference
                trainer.netG.eval()
                x1, x2, offset_flow = trainer.netG(x, masks)

                inpainted_result = x2 * masks + x

                #print psnr
                psnr_tensor, l2_tensor = psnr(ground_truth, inpainted_result)
                print('psnr: {:.3f}, l2: {:.3f}'.format(psnr_tensor, l2_tensor))

                viz_max_out = config['viz_max_out']
                if x.size(0) > viz_max_out:
                    viz_images = torch.stack([ground_truth[:viz_max_out], inpainted_result[:viz_max_out],
                                              offset_flow[:viz_max_out]], dim=1)
                else:
                    viz_images = torch.stack([x, inpainted_result, offset_flow], dim=1)

                viz_images = viz_images.view(-1, *list(x.size())[1:])
                # writer.add_image('valid_img', make_grid(viz_images, nrow=3*4, normalize=True))
                save_image(viz_images,
                           '%s/niter_%03d.png' % (config['test_result'], iteration),
                           nrow=3 * 4,
                           normalize=True)

        # exit no grad context
    except Exception as e:  # for unexpected error logging
        print("Error: {}".format(e))
        raise e


if __name__ == '__main__':
    main()
