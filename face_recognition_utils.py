import pickle
import cv2
import os
import glob
import utils
import dlib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
from . import facenet as FaceNet
from . import utils as FaceNetUtils

use_cuda = torch.cuda.is_available()

config = {
    "dataset": os.path.join(os.path.dirname(__file__), "people"),
    "embeddings": os.path.join(os.path.dirname(__file__), "embeddings.pickle"),
    "detection": "cnn",
    # "image": "test_images/flash2.jpg",
    "image": "test_images/Grant_Gustin_SDCC_2017.jpg",
    "video": "video/flash_trailer.mp4"
}



# cnn_face_detector = dlib.cnn_face_detection_model_v1("./mmod_human_face_detector.dat")
facenet = FaceNet.FaceNet()
facenet.load_saved_model()

def get_embedding(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
    image = Variable(transform(image))
    if use_cuda:
        image = image.cuda()
    embedding = facenet.forward(image.unsqueeze(0))
    return embedding

def save_sample_embeddings():
    image_extensions = ['jpg', 'jpeg', 'png']
    image_paths, image_labels  = [], []
    known_embeddings, known_labels = [], []
    def getPersonFromPath(path):
        return path.split(os.path.sep)[-2]
    for extension in image_extensions:
        image_paths.extend(glob.glob(config['dataset'] + '/*/*.' + extension))
    print(image_paths)

    for (i, image_path) in enumerate(image_paths):
        print("\rProcessing {} of {} images".format(i, len(image_paths)))
        label = getPersonFromPath(image_path)
        image_labels.append(label)
        image = cv2.imread(image_path)
        # print(type(image))
        image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_CUBIC)
        # print(type(image))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_image = FaceNetUtils.get_face_from_person(image)

        print("Image for : ", label)
        if face_image is None:
            print("No faces")
            continue
        else:
            print("face found")

        embedding = get_embedding(face_image)
        embedding = embedding[0]

        known_embeddings.append(embedding)
        known_labels.append(label)

    embeddings_data = {
        "embeddings": known_embeddings,
        "names": known_labels
    }
    f = open(config["embeddings"], "wb")
    f.write(pickle.dumps(embeddings_data))
    f.close()
    return embeddings_data


def get_embedding_data():
    if os.path.exists(config["embeddings"]):
        embeddings_data = pickle.loads(open(config["embeddings"], "rb").read())
    else:
        embeddings_data = save_sample_embeddings()
    return embeddings_data


def recognize_face_in_patch(image):
    face_image = FaceNetUtils.get_face_from_person(image)
    if face_image is None:
        return None
    embedding = get_embedding(face_image)
    embeddings_data = get_embedding_data()
    # print(embeddings_data["embeddings"])
    # print(embeddings_data["names"])
    trained_embeddings = torch.stack(embeddings_data["embeddings"])
    prediction = facenet.find_labels(trained_embeddings, embeddings_data["names"], embedding.unsqueeze(0))
    prediction = prediction[0]
    return prediction

def detect_image(image):
    name = recognize_face_in_patch(image)
    if name is None:
        name = "Unknown"
    cv2.putText(image, name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    # for ((top, right, bottom, left), name) in zip(boxes, names):
    #     # cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    #     y = top - 15 if top - 15 > 15 else top + 15
    #     cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return image

def single_image_detection(image_path):
    image = cv2.imread(image_path)
    detected_image = detect_image(image)
    cv2.imshow("Image", detected_image)
    cv2.waitKey(0)

def process_video():
    # single_image_detection(config["image"])
    frame_idx = 0
    videofile = config["video"]
    video_output = os.path.join(*videofile.split(os.path.sep)[:-1]) + "/output_" + videofile.split(os.path.sep)[-1]
    frame_width, frame_height  = 416, 416
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(video_output, fourcc, 30, (frame_width, frame_height))
    cap = cv2.VideoCapture(videofile)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert cap.isOpened(), 'Cannot capture source'
    while cap.isOpened():
        ret, image = cap.read()

        if ret:
            frame_idx += 1
            print(frame_idx, 'of', total_frames)
            image = cv2.resize(image, (frame_width, frame_height), interpolation=cv2.INTER_CUBIC)
            # print(image.shape)
            detected_image = detect_image(image, embeddings_data)
            # if frame_idx % 100 < 5:
            #     detected_image = detect_image(image)
            # else:
            #     detected_image = image
            out.write(detected_image)
            # cv2.imshow("frame", detected_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            break
            # pass
    cap.release()
    out.release()

if __name__ == "__main__":
    # process_video()

    image = cv2.imread(config["image"])
    output_image = detect_image(image)
    output_name = "output/det_" + config["image"].split("/")[-1]
    cv2.imwrite(output_name, output_image)
