import torch
import torch.nn as nn
import random
from torchvision import datasets

def pairwise_distance(x1, x2):
    diff = torch.abs(x1 - x2)
    return torch.pow(diff, 2).sum(dim=1)


class TripletLoss(nn.Module):
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


def generate_random_triplets(class_map, num_triplets, num_classes):
    triplets = []

    for i in range(num_triplets):
        anchor_class = random.randint(0, num_classes - 1)
        while len(class_map[anchor_class]) < 2:
            # Need anchor with atleast two images
            anchor_class = random.randint(0, num_classes - 1)
        negative_class = random.randint(0, num_classes - 1)
        while negative_class == anchor_class:
            # Need anchor with atleast two images
            negative_class = random.randint(0, num_classes - 1)

        anchor, positive = random.sample(set(class_map[anchor_class]), 2)
        negative = random.sample(set(class_map[negative_class]), 1)[0]

        triplets.append([anchor, positive, negative, anchor_class, negative_class])

    return triplets


class TripletDataset(datasets.ImageFolder):
    def __init__(self, root, num_triplets, transform=None, *arg, **kw):
        super(TripletDataset, self).__init__(root, transform)

        # print(self.imgs)
        # print(self.classes)
        self.num_triplets = num_triplets

        class_map = {}
        # Map class_idx:[image_idx, image_idx2,...]
        for idx, (image_path, class_label) in enumerate(self.imgs):
            if class_label not in class_map:
                class_map[class_label] = []
            class_map[class_label].append(image_path)
        self.class_map = class_map
        #         print(class_map)

        self.triplets = generate_random_triplets(class_map, self.num_triplets, len(self.classes))

    #         print(self.triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        def get_image(image_path):
            image = self.loader(image_path)
            return self.transform(image)

        anchor, positive, negative, anchor_class, negative_class = self.triplets[item]
        anchor_image = get_image(anchor)
        positive_image = get_image(positive)
        negative_image = get_image(negative)

        return anchor_image, positive_image, negative_image, anchor_class, negative_class