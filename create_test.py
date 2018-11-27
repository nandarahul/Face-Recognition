import os, shutil

TRAINING_DIR_PATH, TEST_DIR_PATH= "./dataset/lfw/train_dataset", "./dataset/lfw/test_dataset"
LFW_DIR_PATH = "./dataset/lfw/lfw-deepfunneled"
os.mkdir(TRAINING_DIR_PATH)
os.mkdir(TEST_DIR_PATH)
train_image_count = 0
for label_directory_name in os.listdir(LFW_DIR_PATH):
        label_directory_path = os.path.join(LFW_DIR_PATH, label_directory_name)
        if not os.path.isdir(label_directory_path):
            continue
        image_count = len(os.listdir(label_directory_path))
        if image_count < 5:
            continue
        training_label_dir_path, test_label_dir_path = os.path.join(TRAINING_DIR_PATH, label_directory_name), \
                                                       os.path.join(TEST_DIR_PATH, label_directory_name)
        print("Creating directory: {}".format(test_label_dir_path))
        os.mkdir(training_label_dir_path)
        os.mkdir(test_label_dir_path)
        for i, image_name in enumerate(os.listdir(label_directory_path)):
            if i < image_count * 0.8:
                shutil.copy(os.path.join(label_directory_path, image_name), training_label_dir_path)
                train_image_count += 1
            else:
                shutil.copy(os.path.join(label_directory_path, image_name), test_label_dir_path)
        if train_image_count > 50:
            break
print("Training image count: {}".format(train_image_count))
