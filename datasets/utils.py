import cv2
import pyclipper
import numpy as np
import imgaug.augmenters as iaa


def scale_aligned_short(img, short_size, strid):
    """根据短边进行resize,并调整为32的倍数"""
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % strid != 0:
        h = h + (strid - h % strid)
    if w % strid != 0:
        w = w + (strid - w % strid)
    img = cv2.resize(img, dsize=(w, h))
    return img


def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def letterbox(im, new_shape, color, stride):
    if not isinstance(new_shape, tuple):
        new_shape = (new_shape, new_shape)
    if isinstance(color, int):
        color = (color, color, color)
    shape = im.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    rate = r, r
    pad = int(round(shape[0] * r)), int(round(shape[1] * r))
    pw, ph = new_shape[1] - pad[1], new_shape[0] - pad[0]
    pw, ph = np.mod(pw, stride) / 2, np.mod(ph, stride) / 2

    # if shape != pad:
    #     im = cv2.resize(im, pad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(ph - 0.1)), int(round(ph + 0.1))
    left, right = int(round(pw - 0.1)), int(round(pw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, rate, (top, left)


def get_ann(gt_path):
    """
    如果img is None, 则返回像素坐标
    """
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
        text_regions.append(location)
    return text_regions, words


def clip_polygon(subj, h, w):
    # 重要的点，有的点可能会超出边界
    subjs = []
    for sub in subj:
        sub = sub * np.array([w, h] * (len(sub) // 2))
        sub = sub.reshape((-1, 2))
        subjs.append(sub.astype(int).tolist())
    ori_area = cv2.contourArea(np.reshape(subjs, (-1, 2)))

    # 边界
    clip = ((0, h), (w, h), (w, 0), (0, 0))

    pc = pyclipper.Pyclipper()

    pc.AddPath(clip, pyclipper.PT_CLIP, True)
    pc.AddPaths(subjs, pyclipper.PT_SUBJECT, True)

    solution = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD)
    res = []
    for s in solution:
        s = np.array(s) / np.array([w, h]).T
        res.append(s.flatten())
    return ori_area, res


def make_segment_label(img, text_regions, words, min_scale):
    """

    :param img:
    :param text_regions: 归一化坐标
    :param words:
    :param min_scale:
    :return:
    """
    height, width = img.shape[:2]
    gt = np.zeros((height, width), dtype=np.float32)
    mask = np.ones((height, width), dtype=np.float32)
    selected_polygons = []
    ignore_polygons = []
    for i in range(len(text_regions)):
        ignore = words[i] == '###'
        polygon = text_regions[i]
        original_area, polygon = clip_polygon([polygon], height, width)
        if not len(polygon):
            continue
        polygon = np.reshape(polygon, (-1, 2)) * np.array([width, height]).T
        polygon = np.int32(polygon)
        # 如果文本面积很小并且的长宽太小(看不清楚),那么不要
        area = cv2.contourArea(polygon)
        if area / original_area < 0.1:
            ignore = True
        shape = cv2.minAreaRect(polygon)[1]
        if area < 300 and min(shape) < 10:
            ignore = True
        # 制作label
        if ignore:
            cv2.fillPoly(mask, [polygon], 0)
            ignore_polygons.append(polygon)
        else:
            length = cv2.arcLength(polygon, True)
            distance = area * (1 - min_scale ** 2) / length
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(polygon,
                        pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
            shrunk = pco.Execute(-distance)
            if not shrunk:
                cv2.fillPoly(mask, [polygon], 0)
                ignore_polygons.append(polygon)
            else:
                shrunk = np.reshape(shrunk, (-1, 2)).astype(int)
                cv2.fillPoly(gt, [shrunk], 1)
                selected_polygons.append(shrunk)
    return gt, mask, selected_polygons, ignore_polygons


# Random aug
def random_color_aug(image):
    """
    输入uint8数据[0,255]
    输出uint8数据[0,255]
    """
    # jpeg 图像质量
    sometimes = lambda aug: iaa.Sometimes(0.2, aug)
    image = sometimes(iaa.JpegCompression(compression=(60, 80)))(image=image)
    image = sometimes(iaa.AddToHueAndSaturation((-60, 60)))(image=image)
    k = np.random.randint(3, 8)
    image = sometimes(iaa.MotionBlur(k, angle=[-90, 90]))(image=image)
    image = image.astype(np.uint8)
    return image


def random_rotate(imgs, one_index):
    """根据图像中心随机旋转 [-10,10]度"""
    max_angle = 10
    angle = np.random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        if i == one_index:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST, borderValue=1)
        else:
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST, borderValue=0)
        imgs[i] = img_rotation
    return imgs
