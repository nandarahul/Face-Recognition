from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import utils
import os
use_cuda = True

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

        if torch.cuda.is_available() and use_cuda:
            self.model.cuda()

        self.batch_size = 10
        self.num_triplets_train = self.batch_size ** 3 #(all possible triplets N^3)
        self.num_triplets_test = self.num_triplets_train // 10
        self.lr = 1e-5
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

        train_data, train_labels = [], []
        for class_label, class_images in train_data_loader.class_map.items():
            train_data.append(class_images[0])
            train_labels.append(class_label)
        train_images = [transform(Image.open(image_path)) for image_path in train_data]
        if torch.cuda.is_available() and use_cuda:
            train_images = [image.cuda() for image in train_images]
        train_images = torch.stack(train_images)

        test_data_loader = utils.TripletDataset("./dataset/lfw/test_dataset", self.num_triplets_test, transform)
        test_data = np.array(test_data_loader.imgs)[:, 0]
        # test_images = [transform(Image.open(image_path)) for image_path in test_data]
        # if torch.cuda.is_available() and use_cuda:
        #     test_images = [image.cuda() for image in test_images]
        # test_images = torch.stack(test_images)
        test_labels = np.array(test_data_loader.imgs)[:, 1]

        break_batches = False
        total_epochs = 1000
        last_saved_epoch = -1
        accuracy = 0
        training_state_file = "./state.pkl"
        if os.path.isfile(training_state_file):
            training_state = torch.load(training_state_file)
            last_saved_epoch = training_state["last_saved_epoch"]
            model_file = training_state["model_file"]
            saved_model = torch.load(model_file)
            accuracy = saved_model["accuracy"]
            self.model.load_state_dict(saved_model['state_dict'])
            self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
            print("Loaded state!")
            print(training_state)

        for epoch in range(last_saved_epoch + 1, total_epochs):
            self.model.train()
            train_loss = 0
            print("Epoc: ", epoch)
            total_train_data = len(train_data_loader.triplets)
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
                if idx % 100 == 0:
                    print(idx, "/", total_train_data, loss.item())
                if break_batches:
                    break

            print("Train %.2f" % (train_loss))

            if epoch % 5 == 0:
                self.model.eval()
                correct_ct, total_ct = 0, 0
                with torch.no_grad():
                    # Get updated embeddings
                    train_embeddings = self.forward(train_images)
                    test_embeddings = []
                    for image_path in test_data:
                        test_image = transform(Image.open(image_path))
                        if torch.cuda.is_available() and use_cuda:
                            test_image = test_image.cuda()
                        test_embeddings.append(self.forward(test_image.unsqueeze(0)))

                    test_embeddings = torch.stack(test_embeddings)
                    for test_embedding, test_truth in zip(test_embeddings, test_labels):
                        dist = torch.pow(train_embeddings - test_embedding, 2).sum(1)
                        train_index = torch.argmin(dist).tolist()
                        pred_label = train_labels[train_index]
                        if (pred_label == test_truth):
                            correct_ct += 1
                    total_ct = len(test_embeddings)
                    accuracy = correct_ct / total_ct
                    print("Test {}/{} = Accuracy {}".format(correct_ct, total_ct, accuracy))
            model_file = './saved_model/resnet_{}.pkl'.format(epoch)
            model_state = {
                'state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'accuracy': accuracy
            }
            torch.save(model_state, model_file)
            training_state["last_saved_epoch"] = epoch
            training_state["model_file"] = model_file
            torch.save(training_state, training_state_file)

fn = FaceNet()
fn.train()







