import os
import argparse
import time

import numpy as np
import torch
from PIL import Image
from lib import ReCoNetModel
from ffmpeg_tools import VideoReader, VideoWriter
from model import ReCoNet
from train import stylize_image, stylize_batch
import torchvision.transforms as transforms

from utils import preprocess_for_reconet

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Path to model file")
    parser.add_argument("--input_image", help="Input images")
    parser.add_argument("--use-cpu", action='store_true', help="Use CPU instead of GPU")
    parser.add_argument("--gpu-device", type=int, default=None, help="GPU device index")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--fps", type=int, default=None, help="FPS of output video")
    parser.add_argument("--frn", action='store_true', help="Use Filter Response Normalization and TLU ")

    args = parser.parse_args()

    batch_size = args.batch_size

    model = ReCoNet().cuda()
    model.load_state_dict(torch.load(args.model))



    input = Image.open(args.input_image)
    image = transforms.ToTensor()(input)
    image = image.cuda().unsqueeze_(0)
    image = preprocess_for_reconet(image)

    data = [image] * 100

    start = time.time()
    for image in data:
        model(image)
    end = time.time()
    print('infer done time cost is {}'.format(end - start))
    #styled_test_image = stylize_image(Image.open("test_image.jpeg"), model)


