import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.nn.functional import interpolate

def PIL2Tensor(img, upscale=1, min_size=1024, fix_resize=None):
    '''
    PIL.Image -> Tensor[C, H, W], RGB, [0, 1]
    '''
    # size
    w, h = img.size
    w *= upscale
    h *= upscale
    w0, h0 = round(w), round(h)
    if min(w, h) < min_size:
        _upscale = min_size / min(w, h)
        w *= _upscale
        h *= _upscale
    if fix_resize is not None:
        _upscale = fix_resize / min(w, h)
        w *= _upscale
        h *= _upscale
        w0, h0 = round(w), round(h)
    w = int(np.round(w / 16.0)) * 16
    h = int(np.round(h / 16.0)) * 16
    x = img.resize((w, h), Image.BICUBIC)
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    x = x / 255
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    return x, h0, w0


def Tensor2PIL(x, h0, w0):
    '''
    Tensor[C, H, W], RGB, [0, 1] -> PIL.Image
    '''
    x = x.unsqueeze(0)
    x = interpolate(x, size=(h0, w0), mode='bicubic')
    x = (x.squeeze(0).permute(1, 2, 0) * 255.0).cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)