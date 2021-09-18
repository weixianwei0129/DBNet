import cv2
import numpy as np
import pyclipper


def get_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


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
    return res


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
        polygon = clip_polygon([polygon], height, width)
        if not len(polygon):
            continue
        polygon = np.reshape(polygon, (-1, 2)) * np.array([width, height]).T
        polygon = np.int32(polygon)
        # 如果文本面积很小并且的长宽太小(看不清楚),那么不要
        area = cv2.contourArea(polygon)
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
