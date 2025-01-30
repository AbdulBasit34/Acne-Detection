"""
Transforms to distort local or global information of an image
"""


import torch as th
import numpy as np
import random

from torchvision import transforms

class Scramble(object):
    """
    Create blocks of an image and scramble them
    """
    def __init__(self, blocksize):
        self.blocksize = blocksize

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            # Ensure input is in (C, H, W) format
            if _input.dim() == 2:
                _input = _input.unsqueeze(0)
            elif _input.dim() == 3 and _input.shape[0] not in [1, 3]:
                _input = _input.permute(2, 0, 1)

            size = _input.size()
            img_height, img_width = size[1], size[2]

            x_blocks = img_height // self.blocksize  # number of x blocks
            y_blocks = img_width // self.blocksize

            ind = th.randperm(x_blocks * y_blocks)

            new = th.zeros_like(_input)
            count = 0
            for i in range(x_blocks):
                for j in range(y_blocks):
                    row, column = divmod(ind[count].item(), x_blocks)  # Fix applied here
                    new[:, i * self.blocksize:(i + 1) * self.blocksize,
                           j * self.blocksize:(j + 1) * self.blocksize] = \
                        _input[:, row * self.blocksize:(row + 1) * self.blocksize,
                                     column * self.blocksize:(column + 1) * self.blocksize]
                    count += 1
            outputs.append(new)
        return outputs if idx > 1 else outputs[0]

 

class RandomChoiceScramble(object):

    def __init__(self, blocksizes):
        self.blocksizes = blocksizes

    def __call__(self, *inputs):
        blocksize = random.choice(self.blocksizes)
        outputs = Scramble(blocksize=blocksize)(*inputs)
        return outputs


def _blur_image(image, H):
    if image.ndim == 3 and image.shape[0] != 3:
        image = np.transpose(image, (2, 0, 1))  # Ensure (C, H, W)

    size = image.shape
    imr, img, imb = image[0, :, :], image[1, :, :], image[2, :, :]

    # Fourier Transform
    Fim1r, Fim1g, Fim1b = np.fft.fftshift(np.fft.fft2(imr)), \
                           np.fft.fftshift(np.fft.fft2(img)), \
                           np.fft.fftshift(np.fft.fft2(imb))

    # Apply filter
    filtered_imager, filtered_imageg, filtered_imageb = H * Fim1r, H * Fim1g, H * Fim1b

    newim = np.zeros(size)
    newim[0, :, :] = np.abs(np.fft.ifft2(filtered_imager))
    newim[1, :, :] = np.abs(np.fft.ifft2(filtered_imageg))
    newim[2, :, :] = np.abs(np.fft.ifft2(filtered_imageb))

    return newim.astype('uint8')


def _butterworth_filter(rows, cols, thresh, order):
    # X and Y matrices with ranges normalised to +/- 0.5
    array1 = np.ones(rows)
    array2 = np.ones(cols)
    array3 = np.arange(1,rows+1)
    array4 = np.arange(1,cols+1)

    x = np.outer(array1, array4)
    y = np.outer(array3, array2)

    x = x - float(cols/2) - 1
    y = y - float(rows/2) - 1

    x = x / cols
    y = y / rows

    radius = np.sqrt(np.square(x) + np.square(y))

    matrix1 = radius/thresh
    matrix2 = np.power(matrix1, 2*order)
    f = np.reciprocal(1 + matrix2)

    return f


class Blur(object):
    """
    Blur an image with a Butterworth filter with a frequency
    cutoff matching local block size
    """
    def __init__(self, threshold, order=5):
        """
        scramble blocksize of 128 => filter threshold of 64
        scramble blocksize of 64 => filter threshold of 32
        scramble blocksize of 32 => filter threshold of 16
        scramble blocksize of 16 => filter threshold of 8
        scramble blocksize of 8 => filter threshold of 4
        """
        self.threshold = threshold
        self.order = order

    def __call__(self, *inputs):
        """
        inputs should have values between 0 and 255
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            rows = _input.size(1)
            cols = _input.size(2)
            fc = self.threshold # threshold
            fs = 128.0 # max frequency
            n  = self.order # filter order
            fc_rad = (fc/fs)*0.5
            H = _butterworth_filter(rows, cols, fc_rad, n)
            _input_blurred = _blur_image(_input.numpy().astype('uint8'), H)
            _input_blurred = th.from_numpy(_input_blurred).float()
            outputs.append(_input_blurred)

        return outputs if idx > 1 else outputs[0]


class RandomChoiceBlur(object):

    def __init__(self, thresholds, order=5):
        """
        thresholds = [64.0, 32.0, 16.0, 8.0, 4.0]
        """
        self.thresholds = thresholds
        self.order = order

    def __call__(self, *inputs):
        threshold = random.choice(self.thresholds)
        outputs = Blur(threshold=threshold, order=self.order)(*inputs)
        return outputs






