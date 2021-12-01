import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
import csv
import os



def loadCSV(file_name):
    f = open(file_name, 'r')
    csvreader = csv.reader(f)
    img_train = list(csvreader)

    # check and delete inexistent data
    for index, image in enumerate(img_train):
        if os.path.exists(image[0]) == False or os.path.exists(image[1]) == False:
            print("deleted " + str(img_train[index]))
            del img_train[index]

    # img_train = shuffle(img_train, random_state=0)

    print('Loaded ({0}).'.format(len(img_train)))

    return img_train

def transfer_16bit_to_8bit(image_path):
    image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit

def transfer_16bit_to_8bit_grey(image_path):
    image_16bit = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    # Not all the values are the same, but performance really similar from OpenHDR
    image_16bit = image_16bit[:, :, -1]
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit



class depthDataset(Dataset):
    def __init__(self, img_dataset, transform=None):
        self.img_dataset = img_dataset
        self.transform = transform

    def __getitem__(self, idx):

        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        sample = self.img_dataset[idx]

        # if cv2.imread(sample[0]) is None or cv2.imread(sample[1]) is None:
        #     print('None Image! ')
        #     pass

        # image = Image.open(sample[0])
        image = cv2.imread(sample[0], cv2.IMREAD_UNCHANGED)

        if image is not None:
            if self.transform is not None:
                image = self.transform(image)
            else:
                image = image.transpose(2, 0, 1)
        else:
            image = None

        # for exr file, 3 channels are the same
        if cv2.imread(sample[1], cv2.IMREAD_UNCHANGED) is not None:
            depth = cv2.imread(sample[1], cv2.IMREAD_UNCHANGED)[:, :, 0:1]
            # Resize depth map for matching the output shape of network
            depth = np.expand_dims(cv2.resize(depth, (256, 128)), 2)
        else:
            depth = None

        # transform to dict type
        sample = {'image': image, 'depth': depth}

        return sample

    def __len__(self):
        return len(self.img_dataset)

def my_collate_fn(batch):
    #  fileter NoneType data
    batch = list(filter(lambda x:x['depth'] is not None and x['image'] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)


def getTrainingTestingData(batch_size, csv_file, transform_input=None):

    file_name = csv_file

    print('Loading dataset: ' + file_name)
    img_train = loadCSV(file_name)
    print('Loaded CSV File')

    # transforms.ToTensor()
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ]
    )

    transformed_training = depthDataset(img_train, transform=transform1)

    return DataLoader(transformed_training, batch_size, shuffle=True, pin_memory=False, collate_fn=my_collate_fn)
