import torch
import torch.nn as nn
from torch.autograd import Function

def pairwise_distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)


class TripletLoss(Function):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
            positive_dist = pairwise_distance(anchor, positive)
            negative_dist = pairwise_distance(anchor, negative)
            loss = positive_dist - negative_dist + self.margin
            relu = nn.ReLU()
            loss_final = relu(loss).mean()
            return loss_final


def triplet_loss(anchor, positive, negative, margin=0.2):
    t_loss = TripletLoss(margin)
    loss = t_loss(anchor, positive, negative)
    return loss