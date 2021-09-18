import cv2
import numpy as np
import pyclipper


class SynthResult(object):
    def __init__(self, thresh=0.2, box_thresh=0.7, max_candidates=1000, clipper_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.clipper_ratio = clipper_ratio

    def __call__(self, original_shapes, pred, is_output_polygon=False):
        '''
        original_shapes: [[height, width],[height, width]]
        pred:
            binary: text region segmentation map, with shape (N, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, H, W)
        '''
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh
        boxes_batch = []
        scores_batch = []
        for batch_index in range(pred.shape[0]):
            height, width = original_shapes[batch_index]
            if is_output_polygon:
                boxes, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self.boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap  # .cpu().numpy()  # The first channel
        pred = pred  # .cpu().detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours[:self.max_candidates]:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(pred, contour.squeeze(1))
            if score < self.box_thresh:
                continue

            if points.shape[0] > 2:
                box = dilate_polygon(points, clipper_ratio=self.clipper_ratio)
                if len(box) > 1:
                    continue
            else:
                continue
            box = box.reshape(-1, 2)
            _, side = get_mini_boxes(box.reshape((-1, 1, 2)))
            if side < self.min_size + 2:
                continue

            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''

        assert len(_bitmap.shape) == 2
        bitmap = _bitmap  # .cpu().numpy()  # The first channel
        pred = pred  # .cpu().detach().numpy()
        height, width = bitmap.shape
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)

        for index in range(num_contours):
            contour = contours[index].squeeze(1)
            points, side = get_mini_boxes(contour)
            if side < self.min_size:
                continue
            points = np.array(points)
            score = box_score_fast(pred, contour)
            if self.box_thresh > score:
                continue

            box = dilate_polygon(points, clipper_ratio=self.clipper_ratio).reshape(-1, 1, 2)
            box, side = get_mini_boxes(box)
            if side < self.min_size + 2:
                continue
            box = np.array(box)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()

            box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
        return boxes, scores


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2], points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(np.min(box[:, 0])).astype(int), 0, w - 1)
    xmax = np.clip(np.ceil(np.max(box[:, 0])).astype(int), 0, w - 1)
    ymin = np.clip(np.floor(np.min(box[:, 1])).astype(int), 0, h - 1)
    ymax = np.clip(np.ceil(np.max(box[:, 1])).astype(int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def dilate_polygon(box, clipper_ratio=1.5):
    area = cv2.contourArea(box)
    length = cv2.arcLength(box, True)
    distance = area * clipper_ratio / length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


if __name__ == '__main__':
    from datasets.polygon import PolygonDataSet

    import yaml
    from easydict import EasyDict
    from torch.utils.data import DataLoader

    post_process = SynthResult(clipper_ratio=1.5)

    cfg = EasyDict(yaml.safe_load(open('../config/db_v1.0.0.yaml')))
    dataset = PolygonDataSet(cfg.data, data_type='train')
    batch_size = 2
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
    )
    norm = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-4) * 255).astype(np.uint8)
    for data in loader:
        for i in range(batch_size):
            img = data['img'].numpy()[i].transpose((1, 2, 0))
            dilated_segment = data['dilated_segment'].numpy()[i]
            shrunk_segment = data['shrunk_segment'].numpy()[i]
            threshold = data['threshold'].numpy()[i]
            train_mask = data['train_mask'].numpy()[i]
            concat = [img]
            for x in [dilated_segment, shrunk_segment, threshold, train_mask]:
                x = norm(x)
                concat.append(np.stack([x] * 3, -1))
            concat = np.concatenate(concat, 1)
            # 测试后处理
            pred = np.stack([shrunk_segment, threshold], axis=0)[None, ...]
            boxes, scores = post_process([[640, 640]], pred, is_output_polygon=False)
            boxes = np.array(boxes)
            scores = np.array(scores)
            img = norm(img)
            img = np.ascontiguousarray(img)
            mask = np.zeros_like(img, np.uint8)
            for j in range(boxes.shape[1]):
                box = np.reshape(boxes[0, j], (1, -1, 2)).astype(int)
                cv2.fillPoly(mask, box, (0, 0, 222))
            img = np.clip(mask * .5 + img * .5, 0, 255).astype(np.uint8)
            cv2.imshow("img", img)
            cv2.imshow('concat', concat)
            cv2.waitKey(0)
