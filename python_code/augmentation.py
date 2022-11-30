import copy

import numpy as np
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset


class AugmentedData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        sample = {'IEGM_seg': self.x[idx], 'label': self.y[idx]}
        return sample

    def __len__(self):
        return len(self.y)


class Jittering:
    def __init__(self, percentage, sigma):
        self.percentage = percentage
        self.sigma = sigma

    def run(self, x, y):
        x_aug = np.full_like(x, 0).astype('float32')
        for i in range(x.shape[0]):
            for k in range(x.shape[1]):
                noise = np.random.normal(
                    loc=0,
                    scale=self.sigma * max(abs(x[i][k])),
                    size=(x.shape[2])
                    )
                x_aug[i][k] = x[i][k] + noise
        return x_aug, y


class MagWarp:
    def __init__(self, percentage, sigma, knot):
        self.percentage = percentage
        self.sigma = sigma
        self.knot = knot

    def run(self, x, y):
        x1 = np.full_like(x, 0).astype('float32')
        xx = (np.ones((x.shape[1], 1)) *
              (np.arange(0, x.shape[2], (x.shape[2]-1) / (self.knot+1)))
              ).transpose()
        yy = np.random.normal(
            loc=1.0,
            scale=self.sigma,
            size=(self.knot+2, x.shape[1])
            )
        x_range = np.arange(x.shape[2])
        cs = CubicSpline(xx[:, 0], yy[:, 0])
        curve = np.array([cs(x_range)])
        for i in range(x.shape[0]):
            x1[i] = x[i] * curve
        return x1, y


class Scaling:
    def __init__(self, percentage, sigma):
        self.percentage = percentage
        self.sigma = sigma

    def run(self, x, y):
        scale_factor = np.random.normal(
            loc=1.0,
            scale=self.sigma,
            size=(x.shape[2])
        )
        noise = (np.ones((x.shape[2])) * scale_factor).astype('float32')
        return x * noise, y


class TimeWarp:
    def __init__(self, percentage, sigma, knot):
        self.percentage = percentage
        self.sigma = sigma
        self.knot = knot

    def run(self, x, y):
        xx = (np.ones((x.shape[1], 1)) *
              (np.arange(0, x.shape[2], (x.shape[2]-1) / (self.knot+1)))
              ).transpose()
        yy = np.random.normal(
            loc=1.0,
            scale=self.sigma,
            size=(self.knot+2, x.shape[1])
            )
        x_range = np.arange(x.shape[2])
        cs = CubicSpline(xx[:, 0], yy[:, 0])
        tt = np.array([cs(x_range)]).transpose()
        tt_cum = np.cumsum(tt, axis=0)
        t_scale = [
            (x.shape[2]-1) / tt_cum[-1, 0],
            ]
        tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
        x_new = np.zeros(x.shape).astype('float32')
        x_range = np.arange(x.shape[2])
        for i in range(x.shape[0]):
            x_new[i][0] = np.interp(x_range, tt_cum[:, 0], x[i][0])
        return x_new, y


class DataAugment:
    def __init__(self, x, y, augments=[], seed=42):
        self.x = x
        self.y = y
        self.augments = augments
        self.seed = seed

    def run(self):
        np.random.seed(self.seed)
        x_orig = copy.deepcopy(self.x)
        y_orig = copy.deepcopy(self.y)
        x_out = copy.deepcopy(self.x)
        y_out = copy.deepcopy(self.y)

        for augment in self.augments:
            shuffler = np.random.permutation(len(x_orig))
            x_aug = x_orig[shuffler]
            y_aug = y_orig[shuffler]
            x_aug = x_aug[
                :int(augment.percentage * len(x_aug))
                ]
            y_aug = y_aug[
                :int(augment.percentage * len(y_aug))
                ]
            x_aug1, y_aug1 = augment.run(x_aug, y_aug)
            if x_aug1.shape[0] > x_aug.shape[0]:
                shuffler = np.random.permutation(len(x_aug1))
                x_aug1 = x_aug1[shuffler]
                y_aug1 = y_aug1[shuffler]
                x_aug1 = x_aug1[:int(augment.percentage * len(x_aug1))]
                y_aug1 = y_aug1[:int(augment.percentage * len(y_aug1))]
            x_out = np.concatenate((x_out, x_aug1))
            y_out = np.concatenate((y_out, y_aug1))
        return x_out, y_out[:, 0]
