from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import time
import cv2
import torch
import csv
from sklearn.utils import shuffle
import os
import imageio
from load3D60 import getTrainingTestingData


# checking cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('The Device is: ', str(device))

def loadCSV(file_name):
    f = open(file_name, 'r')
    csvreader = csv.reader(f)
    data = list(csvreader)

    # check and delete inexistent data
    for index, image in enumerate(data):
        if os.path.exists(image[0]) == False or os.path.exists(image[1]) == False:
            del data[index]

    data = shuffle(data, random_state=0)

    print('Loaded ({0}).'.format(len(data)))

    return data


def transfer_16bit_to_8bit_grey(image_path):
    image_16bit = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    # Not all the values are the same, but performance really similar from OpenHDR
    # image_16bit = image_16bit[:, :, -1]
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_8bit = np.array(np.rint(255 * ((image_16bit - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit

def transfer_16bit_to_float(image_path):
    image_16bit = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Not all the values are the same, but performance really similar from OpenHDR
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    # image_8bit = np.array(np.rint((255.0 * (image_16bit - min_16bit)) / float(max_16bit - min_16bit)), dtype=np.uint8)
    image_float = (image_16bit - min_16bit) / float((max_16bit - min_16bit))
    return image_float


def DepthNorm(depth, minDepth=0, maxDepth=10.0):
    depth[depth < minDepth] = minDepth
    depth[depth > maxDepth] = maxDepth
    return maxDepth / depth


def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale
    if len(images.shape) < 3: images = np.stack((images, images, images), axis=2)
    if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth


def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open(file), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)




def compute_errors(gt, pred):

    # mask outliers
    mask_bottom = 0.01 > gt
    mask_top = gt > 10
    gt = np.ma.array(np.ma.array(gt, mask=mask_bottom), mask=mask_top)
    pred = np.ma.array(pred, mask=gt.mask)

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    # need to avoid 'nan' value (calculation between 'inf'œ)
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10

def save_output(batch_idx, img, depth, pre_depth, path='/home/yihongwu/Documents/Datasets/Small_3D60/Evaluation/evaluate_output'):
    for i in range(len(img)):
        cv2.imwrite((path + '/rgb_' + str(batch_idx) + '_' + str(i)) + '.png', img[i].transpose(1, 2, 0))
        imageio.imwrite(path + '/'+ 'depth_' + str(batch_idx) + '_' + str(i) + '.exr', depth[i].transpose(1, 2, 0))
        imageio.imwrite(path + '/pre_depth_' + str(batch_idx) + '_' + str(i) + '.exr', pre_depth[i].transpose(1, 2, 0))

def show_output(img, depth, pre_depth):
    for i in range(len(img)):
        plt.imshow(img[i].transpose(1, 2, 0))
        plt.show()
        plt.imshow(depth[i].transpose(1, 2, 0))
        plt.show()
        plt.imshow(pre_depth[i].transpose(1, 2, 0))
        plt.show()

def evaluate(model, test_data_csv, batch_size, minDepth, maxDepth, crop=None, verbose=False):
    # Load data
    test_loader = getTrainingTestingData(batch_size=batch_size, csv_file=test_data_csv)

    start = time.time()
    print('Testing...')

    # Start Testing...
    result_list = []

    # N = len(test_loader)

    # Switch to train mode
    model.eval()

    predictions = []
    testSetDepths = []

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        if str(device) == 'cpu':
            image = torch.autograd.Variable(
                sample_batched['image'].to(device))  # ndarray: torch.Size([4, 256, 512, 3])
            depth = torch.autograd.Variable(
                sample_batched['depth'].to(device))  # ndarray: torch.Size([4, 256, 512])
        else:
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

        if depth.shape[1] != 1:
            depth = depth.permute(0, 3, 1, 2)  # torch.Size([bs, 1, 256, 512])

        # Normalize depth
        depth_n = DepthNorm(depth, minDepth=minDepth, maxDepth=maxDepth)

        # Predict
        # output = model(image)
        # for ndarray type image
        with torch.no_grad():
            output, pre_domain = model(image.float())

        # detach output
        output = output.detach().cpu().numpy()
        true_y = depth_n.detach().cpu().numpy()

        # Crop based on Eigen et al. crop
        if crop is not None:
            true_y = true_y[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
            output = output[:, :, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append(output[j])
            testSetDepths.append(true_y[j])

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4],
                                                                                  e[5]))

    end = time.time()
    print('\nTest time', end - start, 's')

    return e



def img_normalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    img = img.astype("float") / 255.0
    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
    return img


