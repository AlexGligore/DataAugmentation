import numpy as np
import random
import scipy.ndimage
import cv2
import copy


# img grayscale
def img_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# img flip
def img_flip(img):

    flip = {
        0: np.flip,
        1: np.flipud,
        2: np.fliplr
    }

    operation = random.randint(0, 2)
    flipped_img = flip[operation](img)
    return flipped_img


# img rotation
def img_rotate(img):

    rotation = {
        0: cv2.ROTATE_90_CLOCKWISE,
        1: cv2.ROTATE_90_COUNTERCLOCKWISE,
        2: cv2.ROTATE_180
    }

    operation = random.randint(0, 2)
    rotated_img = cv2.rotate(img, rotation[operation])
    return rotated_img


# img shifting
def img_shift(img, min_shift=1, max_shift=25):

    # RGB img
    if len(img.shape) == 3:
        shift = [random.uniform(min_shift, max_shift), random.uniform(min_shift, max_shift), 0]
        shifted_img = scipy.ndimage.shift(img, shift)
    # Grayscale img
    else:
        shift = [random.uniform(min_shift, max_shift), random.uniform(min_shift, max_shift)]
        shifted_img = scipy.ndimage.shift(img, shift)

    return shifted_img


# adding noise
def img_noise(img, mean=0, var=0.1, sigma_ratio=0.8):
    # gaussian noise
    sigma = var**sigma_ratio

    if len(img.shape) == 3:
        row, col, channels = img.shape
        gauss = np.random.normal(mean, sigma, (row, col, channels))
        gauss = gauss.reshape(row, col, channels)
        noisy_image = img + gauss
    else:
        row, col = img.shape
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy_image = img + gauss

    return noisy_image


# blurring
def img_blur(img, sigma=1):

    blurred_img = scipy.ndimage.gaussian_filter(img, sigma=sigma)
    return blurred_img


class GenerateDataset(object):
    def __init__(self, dataset_size, save_path, nr_of_operation_per_image=3):
        self.dataset_size = dataset_size
        self.save_path = save_path
        self.nr_op = nr_of_operation_per_image
        self.nr_possible_op = 5
        self.operations = {
            0: img_grayscale,
            1: img_shift,
            2: img_flip,
            3: img_blur,
            4: img_noise,
            5: img_rotate
        }

    def generate_dataset(self, img, image_prefix=None, save=False):
        dataset = []
        for i in range(self.dataset_size):
            augmented_img = copy.copy(img)
            for j in range(self.nr_op):
                operation = random.randint(0, self.nr_possible_op)
                augmented_img = self.operations[operation](img)

            if save is True:
                if image_prefix is not None:
                    img_name = self.save_path + str(image_prefix) + '_augmented_' + str(i) + '.jpg'
                else:
                    img_name = self.save_path + 'augmented_' + str(i) + '.jpg'
                cv2.imwrite(img_name, augmented_img)

            dataset.append(augmented_img)

        return dataset


# if __name__ == '__main__':
#     placeholder = np.zeros([255, 255, 3], dtype=np.uint8)
#     placeholder.fill(0)
#     print(img_flip(placeholder).shape)
#     print(img_rotate(placeholder).shape)
#     print(img_shift(img_grayscale(placeholder)).shape)
#     print(img_shift(placeholder).shape)
#     print(img_noise(placeholder).shape)
#     print(img_blur(placeholder).shape)
#     cv2.imshow('test', placeholder)
#     cv2.waitKey(0)
