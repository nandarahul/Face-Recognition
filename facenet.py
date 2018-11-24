import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import utils

use_cuda = False

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

        self.batch_size = 10
        self.num_triplets_train = self.batch_size
        self.num_triplets_test = self.num_triplets_train // 10
        self.lr = 1e-4
        self.loss_fn = utils.triplet_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward(self, input_images):
        output = self.model(input_images)
        return output

    def train(self):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        train_data_loader = utils.TripletDataset("./dataset/lfw/train_dataset", self.num_triplets_train, transform)
        train_data = np.array(train_data_loader.imgs)[:, 0]
        train_images = torch.stack([transform(Image.open(image_path)) for image_path in train_data])
        train_labels = np.array(train_data_loader.imgs)[:, 1]
        test_data_loader = utils.TripletDataset("./dataset/lfw/test_dataset", self.num_triplets_test, transform)
        test_data = np.array(test_data_loader.imgs)[:, 0]
        test_images = torch.stack([transform(Image.open(image_path)) for image_path in test_data])
        test_labels = np.array(test_data_loader.imgs)[:, 1]

        break_batches = False
        total_epochs = 100
        last_saved_epoch = 0

        for epoch in range(last_saved_epoch, total_epochs):
            self.model.train()
            train_loss = 0
            print("Epoc: ", epoch)
            for idx, (anchor_img, pos_img, neg_img, anchor_class, neg_class) in enumerate(train_data_loader):
                anchor_img, pos_img, neg_img = anchor_img.unsqueeze(0), pos_img.unsqueeze(0), neg_img.unsqueeze(0)
                if torch.cuda.is_available() and use_cuda:
                    anchor_img, pos_img, neg_img = anchor_img.cuda(), pos_img.cuda(), neg_img.cuda()
                anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
                anchor_emb, pos_emb, neg_emb = self.forward(anchor_img), self.forward(pos_img), self.forward(neg_img)
                loss = utils.triplet_loss(anchor_emb, pos_emb, neg_emb)
                train_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(loss)
                if break_batches:
                    break

            print("Train %.2f" % (train_loss))

            if epoch % 10 == 0:
                test_loss = 0
                self.model.eval()
                correct_ct, total_ct = 0, 0
                with torch.no_grad():
                    # Get updated embeddings
                    train_embeddings = self.forward(train_images)
                    test_embeddings = self.forward(test_images)
                    for test_embedding, test_truth in zip(test_embeddings, test_labels):
                        dist = torch.pow(train_embeddings - test_embedding, 2).sum(1)
                        train_index = torch.argmin(dist).tolist()
                        pred_label = train_labels[train_index]
                        if (pred_label == test_truth):
                            correct_ct += 1
                    total_ct = len(test_embeddings)
                    accuracy = correct_ct / total_ct

                    for idx, (anchor_img, pos_img, neg_img, anchor_class, neg_class) in enumerate(test_data_loader):
                        anchor_img, pos_img, neg_img = anchor_img.unsqueeze(0), pos_img.unsqueeze(0), neg_img.unsqueeze(
                            0)
                        if torch.cuda.is_available() and use_cuda:
                            anchor_img, pos_img, neg_img = anchor_img.cuda(), pos_img.cuda(), neg_img.cuda()
                        anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
                        anchor_emb, pos_emb, neg_emb = self.forward(anchor_img), self.forward(pos_img), self.forward(
                            neg_img)
                        loss = utils.triplet_loss(anchor_emb, pos_emb, neg_emb)
                        test_loss += loss
                        if break_batches:
                            break

                    print("Test Loss %.2f, Accuracy : %.2f" % (test_loss, accuracy))
                    # print("Test Loss %.2f" % (test_loss))

fn = FaceNet()
fn.train()
