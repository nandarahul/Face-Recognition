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
from shutil import copyfile
use_cuda = torch.cuda.is_available()
from .utils import triplet_loss, TripletDataset

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

        self.batch_size = 60
        # self.num_triplets_train = self.batch_size ** 3 #(all possible triplets N^3)
        self.num_triplets_train = 1000
        self.num_triplets_test = self.num_triplets_train // 10
        self.lr = 1e-7
        self.loss_fn = triplet_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def load_saved_model(self, model_file=None):
        if model_file is None:
            model_file = os.path.join(os.path.dirname(__file__), "saved_model/resnet_best.pkl")
        if not use_cuda:
            saved_model = torch.load(model_file, map_location='cpu')
        else:
            saved_model = torch.load(model_file)
        self.model.load_state_dict(saved_model['state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        return saved_model

    def forward(self, input_images):
        output = self.model(input_images)
        return output

    def train(self):
        transform = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
        test_data_loader = TripletDataset("./dataset/lfw/test_dataset", self.num_triplets_test, transform)
        test_data = np.array(test_data_loader.imgs)[:, 0]
        # test_images = [transform(Image.open(image_path)) for image_path in test_data]
        # if torch.cuda.is_available() and use_cuda:
        #     test_images = [image.cuda() for image in test_images]
        # test_images = torch.stack(test_images)
        test_labels = np.array(test_data_loader.imgs)[:, 1].astype("int")

        break_batches = False
        save_models = True
        total_epochs = 1000
        last_saved_epoch = -1
        accuracy = 0
        best_accuracy = 0
        training_state_file = "./state.pkl"
        training_state = {}
        if os.path.isfile(training_state_file):
            if not use_cuda:
                training_state = torch.load(training_state_file, map_location='cpu')
            else:
                training_state = torch.load(training_state_file)
            print("Loading state..")
            print(training_state)

            last_saved_epoch = training_state["last_saved_epoch"]
            best_accuracy = training_state["best_accuracy"]
            model_file = training_state["model_file"]

            print("Loading: ", model_file)
            saved_model = self.load_saved_model(model_file)
            accuracy = saved_model["accuracy"]
            print("Loaded state!")

        for epoch in range(last_saved_epoch + 1, total_epochs):
            print("*" * 20)
            print("Epoc: ", epoch)
            print("Starting Train")
            train_data_set = TripletDataset("./dataset/lfw/train_dataset", self.num_triplets_train, transform)
            kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}
            train_data_loader = torch.utils.data.DataLoader(train_data_set, batch_size=self.batch_size, shuffle=False,
                                                            **kwargs)
            print("Trainloader done")
            train_data, train_labels = [], []
            for class_label, class_images in train_data_set.class_map.items():
                train_data.append(class_images[0])
                train_labels.append(class_label)
            train_images = [transform(Image.open(image_path)) for image_path in train_data]
            if torch.cuda.is_available() and use_cuda:
                train_images = [image.cuda() for image in train_images]
            train_images = torch.stack(train_images)
            print("Train images done")
            total_train_data = len(train_data_set.triplets)

            for batch_idx, (anchor_img, pos_img, neg_img, anchor_class, neg_class) in enumerate(train_data_loader):
                self.model.train()
                train_loss = 0
                print("Batch: ", batch_idx, end=' ')
                # print(anchor_img[:5][0][0][:5])
                # anchor_img, pos_img, neg_img = anchor_img.unsqueeze(0), pos_img.unsqueeze(0), neg_img.unsqueeze(0)
                if torch.cuda.is_available() and use_cuda:
                    anchor_img, pos_img, neg_img = anchor_img.cuda(), pos_img.cuda(), neg_img.cuda()
                anchor_img, pos_img, neg_img = Variable(anchor_img), Variable(pos_img), Variable(neg_img)
                anchor_emb, pos_emb, neg_emb = self.forward(anchor_img), self.forward(pos_img), self.forward(neg_img)
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb)
                train_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch_idx % 100 == 0:
                    print(batch_idx, "/", total_train_data, loss.item())
                if break_batches:
                    break

                print("loss %.2f" % (train_loss))

                if batch_idx % 10 == 0:
                    correct_ct, total_ct, accuracy = self.test(train_images, train_labels, test_data, test_labels, transform)
                    print("Test {}/{} = Accuracy {}".format(correct_ct, total_ct, accuracy))

                model_state = {
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'accuracy': accuracy
                }
                model_file = './saved_model/resnet_last.pkl'
                if save_models:
                    if os.path.isfile(model_file):
                        #Copy last model for backup, incase cutting python corrupts the file write
                        os.rename(model_file, './saved_model/resnet_last_prev.pkl')
                    torch.save(model_state, model_file)
                training_state["last_saved_epoch"] = epoch
                training_state["model_file"] = model_file
                training_state["loss"] = train_loss
                training_state["accuracy"] = accuracy
                training_state["best_accuracy"] = max(best_accuracy, accuracy)
                if save_models:
                    torch.save(training_state, training_state_file)

                if (epoch % 100 == 0) and save_models:
                    model_file = './saved_model/resnet_{}_{}.pkl'.format(epoch, batch_idx)
                    copyfile('./saved_model/resnet_last.pkl', model_file)
                if best_accuracy < accuracy and save_models:
                    #Save best model if it beats old best_accuracy
                    model_file = './saved_model/resnet_best.pkl'
                    copyfile('./saved_model/resnet_last.pkl', model_file)

    def test(self, train_images, train_labels, test_data, test_labels, transform):
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
            total_ct = len(test_embeddings)
            pred_labels = self.find_labels(train_embeddings, train_labels, test_embeddings)
            for pred_label, test_truth in zip(pred_labels, test_labels):
                if (pred_label == test_truth):
                    correct_ct += 1

            accuracy = (1.0 * correct_ct) / total_ct
        return correct_ct, total_ct, accuracy

    def find_labels(self, known_embeddings, train_labels, test_embeddings, threshold=0.02):
        with torch.no_grad():
            test_labels = []
            print(known_embeddings)
            print(train_labels)
            for test_embedding in test_embeddings:
                dist = torch.pow(known_embeddings - test_embedding, 2).sum(1)
                train_index = torch.argmin(dist).tolist()
                if dist[train_index] > threshold:
                    pred_label = None
                else:
                    pred_label = train_labels[train_index]
                test_labels.append(pred_label)
                print("*"*20)
                print(dist[train_index])
                print(train_index)
                print(test_embedding)
                print("*" * 20)
            return test_labels

if __name__ == "__main__":
    fn = FaceNet()
    fn.train()







