import glob
import itertools
import os
from pathlib import Path
import numpy as np
import cv2
from skimage.io import imread
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, ColorJitter, Resize, FiveCrop, RandomResizedCrop,
                                    RandomHorizontalFlip, RandomRotation, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

Unknown = np.array([0, 0, 0])  # label 0
Agri = np.array([51, 255, 51]) # label 1
Road = np.array([255, 255, 255]) # label 2
Water = np.array([0, 128, 255]) # label 3
Veg = np.array([0, 102, 51]) # label 4
Builtup = np.array([128, 128, 128]) # label 5
Bareland = np.array([255, 128, 0]) # label 6

num_classes = 7

# split huge RS image to small patches
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/openearth/OpenEarthMap_wo_xBD")
    parser.add_argument("--output-img-dir", default="data/openearth/train/images")
    parser.add_argument("--output-mask-dir", default="data/openearth/train/masks")
    parser.add_argument("--gt", action='store_true')  # output RGB mask
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--split-size-h", type=int, default=1024)
    parser.add_argument("--split-size-w", type=int, default=1024)
    parser.add_argument("--stride-h", type=int, default=1024)
    parser.add_argument("--stride-w", type=int, default=1024)
    return parser.parse_args()

def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = Unknown
    mask_rgb[np.all(mask_convert == 1, axis=0)] = Agri
    mask_rgb[np.all(mask_convert == 2, axis=0)] = Road
    mask_rgb[np.all(mask_convert == 3, axis=0)] = Water
    mask_rgb[np.all(mask_convert == 4, axis=0)] = Veg
    mask_rgb[np.all(mask_convert == 5, axis=0)] = Builtup
    mask_rgb[np.all(mask_convert == 6, axis=0)] = Bareland
    return mask_rgb

def rgb2label(label):
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Unknown, axis=-1)] = 0
    label_seg[np.all(label == Agri, axis=-1)] = 1
    label_seg[np.all(label == Road, axis=-1)] = 2
    label_seg[np.all(label == Water, axis=-1)] = 3
    label_seg[np.all(label == Veg, axis=-1)] = 4
    label_seg[np.all(label == Builtup, axis=-1)] = 5
    label_seg[np.all(label == Bareland, axis=-1)] = 6
    return label_seg


def image_augment(image, mask, mode='train'):
    image_list = []
    mask_list = []
    image_width, image_height = image.shape[1], image.shape[0]
    mask_width, mask_height = mask.shape[1], mask.shape[0]
    assert image_height == mask_height and image_width == mask_width
    if mode == 'train':
        image_list_train = [image]
        mask_list_train = [mask]
        for i in range(len(image_list_train)):
            # mask_tmp = rgb2label(mask_list_train[i])
            mask_tmp = mask_list_train[i]
            image_list.append(image_list_train[i])
            mask_list.append(mask_tmp)
    else:
        # mask = rgb2label(mask.copy())
        mask = mask.copy()
        image_list.append(image)
        mask_list.append(mask)
    return image_list, mask_list

def padifneeded(image, mask):
    pad = albu.PadIfNeeded(min_height=1024, min_width=1024, position='bottom_right',
                           border_mode=0, value=[0, 0, 0], mask_value=[0, 0, 0])(image=image, mask=mask)
    # pad = albu.PadIfNeeded(min_height=h, min_width=w)(image=image, mask=mask)
    img_pad, mask_pad = pad['image'], pad['mask']
    assert img_pad.shape[0] == 1024 or img_pad.shape[1] == 1024, print(img_pad.shape)
    # print(img_pad.shape)
    return img_pad, mask_pad

def patch_format(inp):  # sourcery skip: avoid-builtin-shadow, low-code-quality
    (input_dir, seq, imgs_output_dir, masks_output_dir, mode, split_size, stride) = inp
    img_paths = glob.glob(os.path.join(input_dir, str(seq), 'images',  "*.tif"))
    list_file = os.path.join(input_dir, f'{mode}.txt')
    img_paths = [f for f in img_paths if f.split('/')[-1] in np.loadtxt(list_file, dtype=str)]

    # print(img_paths)
    mask_paths = [f.replace("/images/", "/relabelled/") for f in img_paths]
    for img_path, mask_path in zip(img_paths, mask_paths):
        # print(img_path.split('/')[-1])
        # print(mode)

        img = imread(img_path)
        mask = imread(mask_path)

        id = os.path.splitext(os.path.basename(img_path))[0]
        # print(img.shape)
        # assert img.shape == mask.shape and img.shape[0] == 1024, print(img.shape)
        # assert img.shape[1] in [1024, 1024], print(img.shape)
        img, mask = padifneeded(img.copy(), mask.copy())

        # print(img_path)
        # print(img.size, mask.size)
        # img and mask shape: WxHxC
        image_list, mask_list = image_augment(image=img.copy(), mask=mask.copy(), mode=mode)

        assert len(image_list) == len(mask_list)
        for m in range(len(image_list)):
            img = image_list[m]
            mask = mask_list[m]
            img, mask = img[-1024:, -1024:, :], mask[-1024:, -1024:]
            assert img.shape[0] == mask.shape[0] and img.shape[1] == mask.shape[1]
            for k, (y, x) in enumerate(itertools.product(range(0, img.shape[0], stride[0]), range(0, img.shape[1], stride[1]))):
                img_tile_cut = img[y:y + split_size[0], x:x + split_size[1]]
                mask_tile_cut = mask[y:y + split_size[0], x:x + split_size[1]]
                img_tile, mask_tile = img_tile_cut, mask_tile_cut

                if img_tile.shape[0] == split_size[0] and img_tile.shape[1] == split_size[1] \
                        and mask_tile.shape[0] == split_size[0] and mask_tile.shape[1] == split_size[1]:
                    if mode == 'train':
                        out_img_path = os.path.join(imgs_output_dir, f"{seq}_{id}_{m}_{k}.png")
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                    else:
                        img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                        out_img_path = os.path.join(imgs_output_dir, f"{seq}_{id}_{m}_{k}.png")
                    cv2.imwrite(out_img_path, img_tile)
                        # print(img_tile.shape)

                    out_mask_path = os.path.join(masks_output_dir, f"{seq}_{id}_{m}_{k}.png")
                    cv2.imwrite(out_mask_path, mask_tile)

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    input_dir = args.input_dir
    imgs_output_dir = args.output_img_dir
    masks_output_dir = args.output_mask_dir
    mode = args.mode
    split_size_h = args.split_size_h
    split_size_w = args.split_size_w
    split_size = (split_size_h, split_size_w)
    stride_h = args.stride_h
    stride_w = args.stride_w
    stride = (stride_h, stride_w)
    seqs = os.listdir(input_dir)

    if not os.path.exists(imgs_output_dir):
        os.makedirs(imgs_output_dir)
    if not os.path.exists(masks_output_dir):
        os.makedirs(masks_output_dir)

    inp = [(input_dir, seq, imgs_output_dir, masks_output_dir,  mode, split_size, stride)
           for seq in seqs]

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(patch_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print(f'images spliting spends: {split_time} s')


