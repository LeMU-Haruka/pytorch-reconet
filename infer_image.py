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

    start = time.time()
    for i in range(0, 100):
        input = Image.open(args.input_image)

        output = stylize_image(input, model)

        if isinstance(output, torch.autograd.Variable):
            x = output.data
        output = x.cpu().numpy()

        input_format = 'CHW'
        index = [input_format.find(c) for c in 'HWC']
        tensor = output.transpose(index)

        scale_factor = 255
        tensor = tensor.astype(np.float32)
        tensor = (tensor * scale_factor).astype(np.uint8)

        image = Image.fromarray(tensor)
        image.save('test.png')
    end = time.time()
    print('done')
    print('infer done time cost is {}'.format(end - start))
    #styled_test_image = stylize_image(Image.open("test_image.jpeg"), model)


