import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor


# def mixup_data(x, y, alpha=1.0, use_cuda=True):
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#  #       index = torch.randperm(batch_size)
#     else:
#         index = torch.randperm(batch_size)

#     mixed_x = lam * x + (1 - lam) * x[index, :]
#     y_a, y_b = y, y[index]

#     return mixed_x, y_a, y_b, lam


# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# cut out

# def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
#     mask_size_half = mask_size // 2
#     offset = 1 if mask_size % 2 == 0 else 0

#     def _cutout(image):
#         image = np.asarray(image).copy()

#         if np.random.random() > p:
#             return image

#         h, w = image.shape[:2]

#         if cutout_inside:
#             cxmin = max(mask_size_half, 0)
#             cxmax = min(w + offset - mask_size_half, w)
#             cymin = max(mask_size_half, 0)
#             cymax = min(h + offset - mask_size_half, h)
#         else:
#             cxmin = 0
#             cxmax = w + offset
#             cymin = 0
#             cymax = h + offset

#         if cxmin >= cxmax or cymin >= cymax:
#             return image  # Return original image if the cutout region is invalid

#         cx = np.random.randint(cxmin, cxmax)
#         cy = np.random.randint(cymin, cymax)
#         xmin = cx - mask_size_half
#         ymin = cy - mask_size_half
#         xmax = xmin + mask_size
#         ymax = ymin + mask_size
#         xmin = max(0, xmin)
#         ymin = max(0, ymin)
#         xmax = min(w, xmax)
#         ymax = min(h, ymax)
#         image[ymin:ymax, xmin:xmax] = mask_color
#         return image

#     return _cutout


# def cutout_data(x, y, cutout_prob, cutout_inside, mask_size, mask_color=(0, 0, 0), use_cuda=True):
#     '''Returns cutout-applied inputs, pairs of targets, and lambda'''
#     if cutout_prob > 0:
#         cutout_fn = cutout(mask_size, cutout_prob, cutout_inside, mask_color)
#         x_cutout = cutout_fn(x.cpu().numpy()) if use_cuda else cutout_fn(x.numpy())

#         # If using CUDA, convert back to CUDA tensor
#         if use_cuda:
#             x_cutout = torch.from_numpy(x_cutout).to(x.device)

#         # Return cutout-applied inputs and original targets
#         return x_cutout, y

#     # If cutout probability is 0, return original inputs and targets
#     return x, y


# def mixup_data(x, y, alpha=1.0, cutout_prob=0.5, cutout_inside=True, mask_size=16, mask_color=(0, 0, 0), use_cuda=True):
#     '''Returns cutout-mixup-applied inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)

#     # Apply cutout to the original input
#     x_cutout, y_a = cutout_data(x, y, cutout_prob=lam, cutout_inside=cutout_inside, mask_size=mask_size, mask_color=mask_color, use_cuda=use_cuda)

#     # Apply cutout to the randomly selected input
#     x_cutout_rand, y_b = cutout_data(x[index, :], y[index], cutout_prob=lam, cutout_inside=cutout_inside, mask_size=mask_size, mask_color=mask_color, use_cuda=use_cuda)

#     # Combine the cutout-applied inputs with the lambda value
#     mixed_x = lam * x_cutout + (1 - lam) * x_cutout_rand

#     return mixed_x, y_a, y_b, lam

# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

#   Cut_Mix

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns cut-mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

   
class LabelAwareSmoothing(nn.Module):
    def __init__(self, cls_num_list, smooth_head, smooth_tail, shape='concave', power=None):
        super(LabelAwareSmoothing, self).__init__()

        n_1 = max(cls_num_list)
        n_K = min(cls_num_list)

        if shape == 'concave':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.sin((np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'linear':
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * (np.array(cls_num_list) - n_K) / (n_1 - n_K)

        elif shape == 'convex':
            self.smooth = smooth_head + (smooth_head - smooth_tail) * np.sin(1.5 * np.pi + (np.array(cls_num_list) - n_K) * np.pi / (2 * (n_1 - n_K)))

        elif shape == 'exp' and power is not None:
            self.smooth = smooth_tail + (smooth_head - smooth_tail) * np.power((np.array(cls_num_list) - n_K) / (n_1 - n_K), power)

        self.smooth = torch.from_numpy(self.smooth)
        self.smooth = self.smooth.float()
        if torch.cuda.is_available():
            self.smooth = self.smooth.cuda()

    def forward(self, x, target):
        smoothing = self.smooth[target]
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss

        return loss.mean()


class LearnableWeightScaling(nn.Module):
    def __init__(self, num_classes):
        super(LearnableWeightScaling, self).__init__()
        self.learned_norm = nn.Parameter(torch.ones(1, num_classes))

    def forward(self, x):
        return self.learned_norm * x
