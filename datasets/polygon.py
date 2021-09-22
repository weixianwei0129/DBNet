import os
import cv2
import glob
import pyclipper
import warnings
import numpy as np

import torch
from torch.utils.data import Dataset

from datasets.utils import get_img, letterbox
from datasets.utils import random_color_aug, random_rotate
from datasets.utils import make_segment_label, scale_aligned_short

# train_root_dir = '/data/weixianwei/psenet/data/MSRA-TD500_v1.2.0/'
train_root_dir = '/Users/weixianwei/Dataset/open/MSRA-TD500/'
train_data_dir = os.path.join(train_root_dir, 'train')
train_gt_dir = os.path.join(train_root_dir, 'train')

# test_root_dir = '/data/weixianwei/psenet/data/MSRA-TD500_v1.2.0/'
test_root_dir = '/Users/weixianwei/Dataset/open/MSRA-TD500/'
test_data_dir = os.path.join(train_root_dir, 'test')
test_gt_dir = os.path.join(train_root_dir, 'test')

img_postfix = "JPG"
gt_postfix = "TXT"


def get_ann(img, gt_path, ph, pw):
    """
    如果img is None, 则返回像素坐标
    """
    if img is None:
        h, w = 1.0, 1.0
    else:
        h, w = img.shape[:2]
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
        location = np.reshape(location, (-1, 2))
        location = location + np.array([pw, ph])
        location = location / np.array([w, h])
        location = location.flatten()
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
        self.argument = "norm"
        self.argument_method = [
            "norm", "tradition",
            "mosaic",
            # "mix-up",
            "random_crop"
        ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        if self.data_type == "train":
            self.argument = np.random.choice(self.argument_method)
        if self.argument == "mosaic":
            img, text_regions, words = self.mosaic(index)
        elif self.argument == "mix-up":
            img, text_regions, words = self.mix_up(index)
        elif self.argument == "random_crop":
            img, text_regions, words = self.random_crop(index)
        else:
            img = get_img(img_path)  # (h,w,c-rgb)
            img, ratio, (ph, pw) = letterbox(img, self.short_size, 127.5, 32)
            text_regions, words = get_ann(img, gt_path, ph, pw)
            img = cv2.resize(img, (self.short_size, self.short_size))

        # gt mask
        shrunk_segment, train_mask, text_regions, _ = make_segment_label(img, text_regions, words, self.min_scale)
        # canvas, mask
        threshold, dilated_segment = make_border_label(img, text_regions, self.min_scale)

        if self.argument == "tradition":
            if np.random.uniform(0, 10) > 5:
                img = random_color_aug(img)
            collector = [img, shrunk_segment, threshold, train_mask]
            img, shrunk_segment, threshold, train_mask = random_rotate(collector, 3)

        # ==========
        img = img.astype(np.float32) / 127.5 - 1
        img = np.transpose(img, (2, 0, 1))
        data = dict(
            img=torch.from_numpy(img),
            # dilated_segment=torch.from_numpy(dilated_segment),
            shrunk_segment=torch.from_numpy(shrunk_segment),
            threshold=torch.from_numpy(threshold),
            train_mask=torch.from_numpy(train_mask),
        )
        return data

    def mix_up(self, index):
        indexes = np.random.choice(range(len(self.img_paths)), size=(2,)).tolist()
        if index not in indexes:
            indexes[0] = index
        img1 = get_img(self.img_paths[indexes[0]])
        texts_regions1, words1 = get_ann(img1, self.gt_paths[indexes[0]], 0, 0)
        img1 = cv2.resize(img1, (self.short_size, self.short_size))
        img2 = get_img(self.img_paths[indexes[1]])
        texts_regions2, words2 = get_ann(img2, self.gt_paths[indexes[1]], 0, 0)
        img2 = cv2.resize(img2, (self.short_size, self.short_size))
        r = np.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
        img = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        return img, texts_regions1 + texts_regions2, words1 + words2

    def mosaic(self, index):
        """
        田字格画布用a表示
        4张图片用b表示

        """
        text_regions4 = []
        words4 = []
        s = self.short_size
        xc, yc = np.random.uniform(s // 2, s * 3 // 2, size=[2, ]).astype(int)
        indices = [index] + np.random.choice(range(len(self.img_paths)), size=(3,)).tolist()
        for i, index in enumerate(indices):
            img = get_img(self.img_paths[index])
            texts_regions, words = get_ann(img, self.gt_paths[index], 0, 0)
            img = scale_aligned_short(img, int(self.short_size * np.random.uniform(0.9, 2)), 32)
            h, w, c = img.shape

            if i == 0:  # a图的左上图, 图b右下角与中心对齐
                img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(0, xc - w), max(0, yc - h), xc, yc
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = w - wa, h - ha, w, h
            elif i == 1:  # a图的右上图, 图片b左下角与中心对齐
                x1a, y1a, x2a, y2a = xc, max(0, yc - h), min(s * 2, xc + w), yc
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = 0, h - ha, min(w, wa), h
            elif i == 2:  # a图的左下图, 图片b的右上角与中心对齐
                x1a, y1a, x2a, y2a = max(0, xc - w), yc, xc, min(s * 2, yc + h)
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = w - wa, 0, w, min(ha, h)
            elif i == 3:  # a图右下图, 图b的左上角与中心对齐
                x1a, y1a, x2a, y2a = xc, yc, min(s * 2, xc + w), min(s * 2, yc + h)
                wa, ha = x2a - x1a, y2a - y1a
                x1b, y1b, x2b, y2b = 0, 0, min(w, wa), min(h, ha)
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            for points, word in zip(texts_regions, words):
                num = points.shape[0] // 2
                points = points * np.array([w, h] * num) + np.array([padw, padh] * num)
                points = points / np.array([s * 2, s * 2] * num)
                text_regions4.append(points)
                words4.append(word)
        img4 = cv2.resize(img4, (self.short_size, self.short_size))
        return img4, text_regions4, words4

    def random_crop(self, index):
        img = get_img(self.img_paths[index])
        h, w = img.shape[:2]
        if min(h, w) < self.short_size:
            text_regions, words = get_ann(img, self.gt_paths[index], 0, 0)
        else:
            bh = np.random.randint(0, h - self.short_size)
            bw = np.random.randint(0, w - self.short_size)
            img = img[bh:bh + self.short_size, bw:bw + self.short_size, :]
            text_regions, words = get_ann(img, self.gt_paths[index], -bh, -bw)
        return img, text_regions, words


if __name__ == '__main__':
    import yaml
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    cfg = EasyDict(yaml.safe_load(open('../config/db_v1.0.0.yaml')))
    dataset = PolygonDataSet(cfg.data, data_type='train')
    batch_size = 3
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    for data in loader:
        for i in range(batch_size):
            img = data['img'].numpy()[i].transpose((1, 2, 0))
            shrunk_segment = data['shrunk_segment'].numpy()[i]
            threshold = data['threshold'].numpy()[i]
            train_mask = data['train_mask'].numpy()[i]
            concat = [img]
            for x in [shrunk_segment, threshold, train_mask]:
                x = (x * 255).astype(np.uint8)
                concat.append(np.stack([x] * 3, -1))
            concat = np.concatenate(concat, 1)
            cv2.imshow('img', concat)
            cv2.waitKey(0)
