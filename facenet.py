import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
from torchvision.models import resnet18
import utils

class FaceNet(nn.Module):
    def __init__(self, embedding_dimensions=64):
        super(FaceNet, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.model = resnet18(pretrained=True)
        resnet_fc_in = self.model.fc.in_features
        self.model.fc = nn.Linear(resnet_fc_in, embedding_dimensions)

        # Initialize weights
        self.model.fc.weight.data.normal_(0.0, 0.02)
        self.model.fc.bias.data.fill_(0)

        self.minibatch_size = 10
        self.lr = 1e-4
        self.loss_fn = utils.triplet_loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, input_images):
        output = self.model(input_images)
        return output

    def train(self):
        total_epochs = 50000
        last_saved_epoch = 0
        for epoch in range(last_saved_epoch, total_epochs):
            self.model.train()
            pass
