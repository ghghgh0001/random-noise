import cv2
from glob2 import glob
import numpy as np
import os


def add_gaussian_noise(image_in, noise_sigma=20):
    num = 0
    img_ori = np.float64(np.copy(image_in))
    img = np.float64(np.copy(image_in))

    h = img.shape[0]
    w = img.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(img.shape, np.float64)
    if len(img.shape) == 2:
        noisy_image = img + noise
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + noise
        noisy_image[:, :, 1] = img[:, :, 1] + noise
        noisy_image[:, :, 2] = img[:, :, 2] + noise
    noisy_image[noisy_image < 0] = 0
    noisy_image[noisy_image > 255] = 255
    err = cv2.absdiff(noisy_image, img_ori)
    ori_num = np.sum(img_ori**2)
    num = np.sum(err**2)
    snr = num/ori_num
    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """
    return noisy_image, snr

def add_tri_noise(image_in, noise_sigma=10):
    img_ori = np.float64(np.copy(image_in))
    img = np.float64(np.copy(image_in))
    num = 0

    h = img.shape[0]
    w = img.shape[1]
    num1 = int(h * w * noise_sigma / 40000)  # 多少个像素点添加椒盐噪声
    for i in range(num1):
        w1 = np.random.randint(2, w - 3)
        h1 = np.random.randint(1, h - 2)
        # img[h1, w1] = [255, 255, 255]
        if np.random.randint(2) == 0:
            img[h1-1, w1] = [0, 0, 0]
            img[h1, w1-1] = img[h1, w1+1] = [0, 0, 0]
            img[h1+1, w1-2:w1+3] = [0, 0, 0]
        else:
            img[h1-1, w1] = [255, 255, 255]
            img[h1, w1-1] = img[h1, w1+1] = [255, 255, 255]
            img[h1+1, w1-2:w1+3] = [255, 255, 255]
    err = cv2.absdiff(img, img_ori)
    ori_num = np.sum(img_ori**2)
    num = np.sum(err**2)
    snr = num/ori_num
    return img, snr

def add_lingxing_noise(image_in, noise_sigma=10):
    img_ori = np.float64(np.copy(image_in))
    img = np.float64(np.copy(image_in))
    num = 0

    h = img.shape[0]
    w = img.shape[1]
    num1 = int(h * w * noise_sigma / 20000)  # 多少个像素点添加椒盐噪声
    for i in range(num1):
        w1 = np.random.randint(1, w - 2)
        h1 = np.random.randint(1, h - 2)
        # img[h1, w1] = [255, 255, 255]
        if np.random.randint(2) == 0:
            img[h1-1, w1] = img[h1, w1-1] = img[h1, w1+1] = img[h1+1, w1] = [0, 0, 0]
        else:
            img[h1 - 1, w1] = img[h1, w1 - 1] = img[h1, w1 + 1] = img[h1 + 1, w1] = [255, 255, 255]
    err = cv2.absdiff(img, img_ori)
    ori_num = np.sum(img_ori**2)
    num = np.sum(err**2)
    snr = num/ori_num
    return img, snr


def add_square_noise(image_in, noise_sigma=10):
    img_ori = np.float64(np.copy(image_in))
    img = np.float64(np.copy(image_in))
    num = 0

    h = img.shape[0]
    w = img.shape[1]
    num1 = int(h * w * noise_sigma / 40000)  # 多少个像素点添加椒盐噪声
    for i in range(num1):
        w1 = np.random.randint(1, w - 2)
        h1 = np.random.randint(1, h - 2)
        # img[h1, w1] = [255, 255, 255]
        if np.random.randint(2) == 0:
            img[h1-1, w1-1:w1+2] = [0, 0, 0]
            img[h1, w1-1] = img[h1, w1+1] = [0, 0, 0]
            img[h1+1, w1-1:w1+2] = [0, 0, 0]
        else:
            img[h1-1, w1-1:w1+2] = [255, 255, 255]
            img[h1, w1-1] = img[h1, w1+1] = [255, 255, 255]
            img[h1+1, w1-1:w1+2] = [255, 255, 255]
    err = cv2.absdiff(img, img_ori)
    ori_num = np.sum(img_ori**2)
    num = np.sum(err**2)
    snr = num/ori_num
    return img, snr


def add_alpharnd_noise(image_in, a, noise_sigma=10):
    img_ori = np.float64(np.copy(image_in))
    img = np.float64(np.copy(image_in))
    hight = img.shape[0]
    width = img.shape[1]
    n = int(hight) * int(width)
    y = np.zeros(n)
    for i in range(n):
        cc0 = 31
        cc = 0
        a1 = a - 1
        while(cc0 > 30):
            while(cc < 0.000001):
                v = np.pi * (np.random.random() - 0.5)
                cc1 = np.log(np.cos(a1 * v))
                cc = np.cos(v)
            w = -np.log(np.random.random())
            cc = np.log(cc)
            cc0 = a1 * (np.log(w) - cc1) -cc
        x = abs(np.sin(a * v)) * np.exp(cc0/a)
        if np.random.random() < 0.5:
            x = -x
        y[i] = x * noise_sigma
    noise = np.reshape(np.array(y), (hight, width))
    noisy_image = np.zeros(img.shape, np.float64)
    if len(img.shape) == 2:
        noisy_image = img + noise
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + noise
        noisy_image[:, :, 1] = img[:, :, 1] + noise
        noisy_image[:, :, 2] = img[:, :, 2] + noise
    noisy_image[noisy_image < 0] = 0
    noisy_image[noisy_image > 255] = 255
    err = cv2.absdiff(noisy_image, img_ori)
    ori_num = np.sum(img_ori**2)
    num = np.sum(err**2)
    snr = num/ori_num
    return noisy_image, snr


def process_img(acc):
    NUM = 0
    for fn in glob(r'./data/flower_photos/daisy_224/*.jpg'):
        img = cv2.imread(fn)
        splitName = fn.split("daisy_224\\")
        # newName = splitName[0]
        id = splitName[1]
        for i in range(10):
            img_add, num = add_alpharnd_noise(img, 0.9, acc)
            i_str = str(i)
            location = r'./data/flower_photos/alpha0.9/' + str(acc) + '/' + i_str + '_' + id
            cv2.imwrite(location, img_add)
        # img, num = add_salt_noise(img, 0.05)
        # cv2.imwrite(r'./data/flower_photos/daisy_circles2/' + id, img)
            NUM += num
    return NUM/6330


changed_pixels = []


for i in range(9):
    path = r'./data/flower_photos/alpha0.9/' + str((i + 1) / 1000)
    os.makedirs(path)
    NUM = process_img((i + 1) / 1000)
    changed_pixels.append(NUM)
    print(changed_pixels.__len__(), changed_pixels)
listText = open('./data/flower_photos/alpha_0.9_list.txt', 'a')
list_write = str(changed_pixels) + '\n'
listText.write(list_write)
listText.close()
