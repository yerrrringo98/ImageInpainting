from skimage.feature import canny
from skimage.color import rgb2gray
from joblib import  Parallel, delayed
import numpy as np
import random
import torch
from PIL import Image
import torchvision.transforms.functional as F

def load_data(img, mask, sigma, mode, augment=False):
    img_gray = rgb2gray(img.permute(0,2,3,1))
    edge = load_edge(img_gray, mask, sigma, mode)

    # augment data
    if augment and np.random.binomial(1, 0.5) > 0:
        img = img[:, ::-1, ...]
        img_gray = img_gray[:, ::-1, ...]
        edge = edge[:, ::-1, ...]
        mask = mask[:, ::-1, ...]

    return img.float(), torch.tensor(img_gray).unsqueeze(1).float(), edge.unsqueeze(1).float(), mask.float()

def load_edge(imgs, mask, sigma, mode):
    mask = None if mode=='train' else (1 - mask / 255).astype(np.bool)

    # canny
    if sigma == -1:
        return np.zeros(imgs.shape).astype(np.float)

    # random sigma
    if sigma == 0:
        sigma = random.randint(1, 4)
        imgs = Parallel(n_jobs=4)(delayed(joblib_loop)(img, sigma, mask) for img in imgs)
        return torch.cat(imgs)
        # return canny(img, sigma=sigma, mask=mask).astype(np.float)


def joblib_loop(img, sigma, mask):
    img = img.squeeze()
    img = canny(img, sigma=sigma, mask=mask).astype(np.float)
    img = torch.from_numpy(img).unsqueeze(0)
    return img

# def to_tensor(img):
#     img = Image.fromarray(img)
#     img_t = F.to_tensor(img).float()
#     return img_t