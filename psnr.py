import torch
import piq
import tqdm
import time
import argparse
import torch.nn as nn

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@torch.no_grad()
def psnr(gt_batch, img_batch):
    psnr_tensor = 0
    ssim_tensor = 0
    l1_tensor = 0
    l2_tensor = 0
    # lpips_tensor = 0
    count = 0

    # gt_batch -= gt_batch.min(1, keepdim=True)[0]
    # gt_batch /= gt_batch.max(1, keepdim=True)[0]
    # img_batch -= img_batch.min(1, keepdim=True)[0]
    # img_batch /= img_batch.max(1, keepdim=True)[0]

    print("Calculating image quality metrics ...")
    if torch.cuda.is_available():
        # Move to GPU to make computaions faster
        img_batch = img_batch.cuda()
        gt_batch = gt_batch.cuda()

    for i in range(gt_batch.shape[0]):
        gt, img = gt_batch[i], img_batch[i]
        gt = torch.clamp(gt, min=0, max=1)
        img = torch.clamp(img, min=0, max=1)
        # MS-SIM
        # ms_ssim_index: torch.Tensor = piq.multi_scale_ssim(gt, img, data_range=1.)
        # PSNR
        psnr_index: torch.Tensor = piq.psnr(gt, img, data_range=1., reduction='mean')
        # L1 Error
        # l1_index = nn.L1Loss(reduction='mean')(gt, img)
        # L1 Error
        l2_index = nn.MSELoss(reduction='mean')(gt, img)
        # LPIPS
        # lpips_loss: torch.Tensor = piq.LPIPS(reduction='mean')(gt, img)

        # Adding for computing average value
        # ssim_tensor += ms_ssim_index
        psnr_tensor += psnr_index
        # l1_tensor += l1_index
        l2_tensor += l2_index
        # lpips_tensor += lpips_loss.item()

        count += 1

    return (psnr_tensor/count, l2_tensor/count)
    # print(
    #     "Avg. SSIM: {} \nAvg. PSNR: {} \nAvg. L1: {} \nAvg. L2: {} \n".format(
    #         ssim_tensor / count,
    #         psnr_tensor / count,
    #         l1_tensor / count,
    #         l2_tensor / count))
