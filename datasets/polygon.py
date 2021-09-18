import os
import cv2
import glob
import pyclipper
import warnings
import numpy as np

import torch
from torch.utils.data import Dataset

from datasets.utils import get_img
from datasets.utils import make_segment_label

train_root_dir = '/data/weixianwei/psenet/data/MSRA-TD500_v1.2.0/'
# train_root_dir = '/Users/weixianwei/Dataset/open/MSRA-TD500/'
train_data_dir = os.path.join(train_root_dir, 'train')
train_gt_dir = os.path.join(train_root_dir, 'train')

test_root_dir = '/data/weixianwei/psenet/data/MSRA-TD500_v1.2.0/'
# test_root_dir = '/Users/weixianwei/Dataset/open/MSRA-TD500/'
test_data_dir = os.path.join(train_root_dir, 'test')
test_gt_dir = os.path.join(train_root_dir, 'test')

img_postfix = "JPG"
gt_postfix = "TXT"


def get_ann(img, gt_path):
    """
    如果img is None, 则返回像素坐标
    """
    if img is None:
        h, w = 1.0, 1.0
    else:
        h, w = img.shape[0:2]
    lines = open(gt_path, 'r').readlines()
    text_regions = []
    words = []
    for idx, line in enumerate(lines):
        sp = line.strip().split(',')
        word = sp[-1]
        if word[0] == '#':
            words.append('###')
        else:
            words.append(word)
        location = [int(x) for x in sp[:-1]]
        location = np.array(location) / ([w * 1.0, h * 1.0] * int(len(location) / 2))
        text_regions.append(location)
    return text_regions, words


def get_distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    # 计算像素的各个点到多边形任意两个点构成的直线的距离.
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    # 点1到各个像素点的距离的平方 (H, W)
    square_distance_1 = np.square(xs - point_1[0]) + np.square(ys - point_1[1])
    # 点2到各个像素点的距离的平方 (H, W)
    square_distance_2 = np.square(xs - point_2[0]) + np.square(ys - point_2[1])
    # 点1到点2的距离的平方 一个值
    square_distance = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])
    # 点1点2的距离为c, 某个像素的距点1的距离为a, 距离点2的距离为b
    # 那么: cosin = (c^2 - (a^2+b^2)) / 2ab
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cosin = (square_distance - square_distance_1 - square_distance_2) / \
                (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
    # 根据三角形面积计算高,高即为点到直线的距离
    result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)
    #
    result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
    # self.extend_line(point_1, point_2, result)
    return result


def make_border_label(img, text_regions, min_scale):
    """

    :param img:
    :param text_regions:  像素坐标
    :param min_scale:
    :return:
    """
    height, width = img.shape[:2]

    canvas = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)

    for polygon in text_regions:
        area = cv2.contourArea(polygon)
        length = cv2.arcLength(polygon, True)
        distance = area * (1 - min_scale ** 2) / length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(polygon,
                    pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
        dilated = np.reshape(pco.Execute(distance)[0], (-1, 2)).astype(int)
        cv2.fillPoly(mask, [dilated], 1.0)

        # 生成高斯边缘阈值
        xmin = dilated[:, 0].min()
        xmax = dilated[:, 0].max()
        ymin = dilated[:, 1].min()
        ymax = dilated[:, 1].max()

        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = get_distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid - ymin:ymax_valid - ymax + height,
                xmin_valid - xmin:xmax_valid - xmax + width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])
    return canvas, mask


class PolygonDataSet(Dataset):
    def __init__(self, cfg, data_type='train'):
        self.short_size = cfg.short_size
        self.kernel_num = cfg.kernel_num
        self.min_scale = cfg.min_scale
        self.use_mosaic = cfg.use_mosaic
        self.data_type = data_type

        if self.data_type == 'train':
            data_pattern = os.path.join(train_data_dir, f"*.{img_postfix}")
            gt_pattern = os.path.join(train_gt_dir, f"*.{gt_postfix}")
        elif self.data_type == 'test':
            data_pattern = os.path.join(test_data_dir, f"*.{img_postfix}")
            gt_pattern = os.path.join(test_gt_dir, f"*.{gt_postfix}")
        else:
            print('Error: data_type must be train or test!')
            raise
        self.img_paths = glob.glob(data_pattern)
        self.img_paths.sort()
        self.gt_paths = glob.glob(gt_pattern)
        self.gt_paths.sort()
        assert len(self.gt_paths) == len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        img = get_img(img_path)  # (h,w,c-rgb)
        text_regions, words = get_ann(img, gt_path)
        img = cv2.resize(img, (self.short_size, self.short_size))
        # gt mask
        shrunk_segment, train_mask, text_regions, _ = make_segment_label(img, text_regions, words, self.min_scale)
        # canvas, mask
        threshold, dilated_segment = make_border_label(img, text_regions, self.min_scale)

        # ==========
        img = img.astype(np.float32) / 127.5 - 1
        img = np.transpose(img, (2, 0, 1))
        data = dict(
            img=torch.from_numpy(img),
            dilated_segment=torch.from_numpy(dilated_segment),
            shrunk_segment=torch.from_numpy(shrunk_segment),
            threshold=torch.from_numpy(threshold),
            train_mask=torch.from_numpy(train_mask),
        )
        return data


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    cfg = EasyDict(yaml.safe_load(open('../config/db_v1.0.0.yaml')))
    dataset = PolygonDataSet(cfg.data, data_type='train')
    batch_size = 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    for data in loader:
        for i in range(batch_size):
            img = data['img'].numpy()[i].transpose((1, 2, 0))
            dilated_segment = data['dilated_segment'].numpy()[i]
            shrunk_segment = data['shrunk_segment'].numpy()[i]
            threshold = data['threshold'].numpy()[i]
            train_mask = data['train_mask'].numpy()[i]
            concat = [img]
            for x in [dilated_segment, shrunk_segment, threshold, train_mask]:
                x = (x * 255).astype(np.uint8)
                concat.append(np.stack([x] * 3, -1))
            concat = np.concatenate(concat, 1)
            cv2.imshow('img', concat)
            cv2.waitKey(0)
