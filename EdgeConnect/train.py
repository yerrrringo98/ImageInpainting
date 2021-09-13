import os, random
import torch.backends.cudnn as cudnn
from src.utils import get_config
from argparse import ArgumentParser
import cv2
import torch
import numpy as np
from src.edge_connect import EdgeConnect


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/edge-config.yaml',help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--psnr', type=bool)

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    # CUDA configuration
    cuda = config['CUDA']
    device_ids = config['GPU_IDS']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        config['DEVICE'] = torch.device('cuda')
        cudnn.benchmark = True

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config['SEED'])
    torch.cuda.manual_seed_all(config['SEED'])
    np.random.seed(config['SEED'])
    random.seed(config['SEED'])

    # build the model and initialize
    model = EdgeConnect(config)
    model.load()

    #model training
    print('Start training..')
    model.train()

if __name__ == '__main__':
    main()