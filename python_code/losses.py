import torch
import torch.nn.functional as F

# Ops to Latency linear-fit conversion constant
M = 2.86e-4


def combined_reg_loss(sz, op,
                      min_sz=5, max_sz=256, code_sz=17,
                      min_lt=1, max_lt=200,
                      strength=0.2, mode='max'):
    # import pdb; pdb.set_trace()
    sz_k = sz / 1000  # size in [kB]
    lt_m = op * M  # latency in [ms] obtained from linear fit lat vs ops
    if mode == 'max':
        sz_loss = strength * \
            max((sz_k + code_sz - min_sz) / (max_sz - min_sz),
                torch.tensor(0.))
        lt_loss = strength * \
            max((lt_m - min_lt) / (max_lt - min_lt),
                torch.tensor(0.))
    elif mode == 'abs':
        sz_loss = strength * \
            torch.abs((sz_k + code_sz - min_sz) / (max_sz - min_sz))
        lt_loss = strength * \
            torch.abs((lt_m - min_lt) / (max_lt - min_lt))
    else:
        raise ValueError('kwargs "mode" possible values are "max" or "abs"')

    return sz_loss + lt_loss


# https://tinyurl.com/soft-fb-loss
def soft_fb_loss(beta=1, eps=1e-16):
    def loss(pred, target):
        bs = target.shape[0]
        norm_pred = F.softmax(pred, dim=-1)
        # tp = norm_pred[range(bs), target].sum()
        tp = (norm_pred[range(bs), target] * target).sum()
        # fp = norm_pred[range(bs), 1 - target].sum()
        fp = (norm_pred[range(bs), 1 - target] * (1 - target)).sum()
        # fn = (1 - norm_pred)[range(bs), target].sum()
        fn = ((1 - norm_pred)[range(bs), target] * target).sum()
        num = (1 + (beta ** 2)) * tp
        den = (1 + (beta ** 2)) * tp + (beta ** 2) * fn + fp + eps
        soft_fb = num / den

        return 1 - soft_fb
    return loss


# https://tinyurl.com/soft-fb-loss-springer
def soft_fb_loss_v1(beta=1, eps=1e-16):
    def loss(pred, target):
        bs = target.shape[0]
        norm_pred = F.softmax(pred, dim=-1)
        num = (1 + (beta ** 2)) * (norm_pred[range(bs), target] * target).sum()
        den = (norm_pred[range(bs), target] + (beta ** 2) * target).sum() + eps
        soft_fb = num / den

        return 1 - soft_fb
    return loss
