import os, sys

from PIL import Image

current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)

import numpy as np
import torch

from model import ReCoNet
from utils import preprocess_for_reconet, postprocess_reconet, Dummy, nhwc_to_nchw, nchw_to_nhwc

sys.path.remove(current_dir)


class ReCoNetModel:

    def __init__(self, state_dict_path, use_gpu=True, gpu_device=None, frn=False):
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device

        with self.device():
            self.model = ReCoNet(frn=frn)
            self.model.load_state_dict(torch.load(state_dict_path))
            self.model = self.to_device(self.model)

    def run(self, images):
        images = torch.from_numpy(images).cuda()

        with self.device():
            with torch.no_grad():
                images = self.to_device(images)
                images = preprocess_for_reconet(images)
                styled_images = self.model(images)
                styled_images = postprocess_reconet(styled_images)
                styled_images = styled_images.data.cpu().numpy()
                result = []
                for output in styled_images:
                    input_format = 'CHW'
                    index = [input_format.find(c) for c in 'HWC']
                    tensor = output.transpose(index)

                    scale_factor = 255
                    tensor = tensor.astype(np.float32)
                    tensor = (tensor * scale_factor).astype(np.uint8)
                    result.append(tensor)
                    image = Image.fromarray(tensor)
                    image.save('test.png')
                return result

    def to_device(self, x):
        if self.use_gpu:
            with self.device():
                return x.cuda()
        else:
            return x

    def device(self):
        if self.use_gpu and self.gpu_device is not None:
            return torch.cuda.device(self.gpu_device)
        else:
            return Dummy()
