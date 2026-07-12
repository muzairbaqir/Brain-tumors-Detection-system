import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

class MedicalCorruptions:
    @staticmethod
    def gaussian_noise(x, severity):
        c = [0.04, 0.06, 0.08, 0.12, 0.16][severity-1]
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1).astype(np.float32)

    @staticmethod
    def motion_blur(x, severity):
        c = [3, 5, 7, 9, 12][severity-1]
        kernel = np.zeros((c, c))
        kernel[int((c - 1) / 2), :] = np.ones(c)
        kernel = kernel / c
        return cv2.filter2D(x, -1, kernel)

    @staticmethod
    def elastic_deform(x, severity):
        c = [(1, 0.1), (2, 0.2), (3, 0.3), (4, 0.4), (5, 0.5)][severity-1]
        shape = x.shape
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), c[0], mode="constant", cval=0) * c[1] * 100
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), c[0], mode="constant", cval=0) * c[1] * 100
        x_grid, y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y_grid+dy, (-1, 1)), np.reshape(x_grid+dx, (-1, 1))
        return map_coordinates(x, indices, order=1).reshape(shape)

    @staticmethod
    def intensity_shift(x, severity):
        c = [0.1, 0.2, 0.3, 0.4, 0.5][severity-1]
        return np.clip(x * (1.0 + (np.random.uniform(-c, c))), 0, 1)

    @staticmethod
    def spike_noise(x, severity):
        c = [0.01, 0.02, 0.05, 0.1, 0.15][severity-1]
        mask = np.random.choice([0, 1], size=x.shape, p=[1-c, c])
        return np.clip(x + mask * np.random.uniform(0, 1, size=x.shape), 0, 1)
