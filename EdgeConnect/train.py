from utils.tools import *
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--config', type=str, default='../configs/generative-config.yaml',help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--psnr', type=bool)

def main():
    args = parser.parse_args()
    config = get_config()