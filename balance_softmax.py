import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, cls_num_list):
        super(BalancedSoftmaxLoss, self).__init__()
        cls_num_list = np.array(cls_num_list, dtype=np.float32)
        cls_prior = cls_num_list / sum(cls_num_list)
        self.log_prior = torch.log(torch.tensor(cls_prior, dtype=torch.float32).cuda()).unsqueeze(0)

    def forward(self, logits, labels):
        adjusted_logits = logits + self.log_prior
        label_loss = F.cross_entropy(adjusted_logits, labels)

        return label_loss
