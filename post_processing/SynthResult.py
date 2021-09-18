import cv2
import numpy as np
from post_processing.common import draw_boxes_on_img, norm
from post_processing.common import get_mini_boxes, box_score_fast, dilate_polygon


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
                boxes, scores = self._polygons_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            else:
                boxes, scores = self._boxes_from_bitmap(pred[batch_index], segmentation[batch_index], width, height)
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def _polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
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

    def _boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
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
    for data in loader:
        for i in range(batch_size):
            img = data['img'].numpy()[i].transpose((1, 2, 0))
            dilated_segment = data['dilated_segment'].numpy()[i]
            shrunk_segment = data['shrunk_segment'].numpy()[i]
            threshold = data['threshold'].numpy()[i]
            train_mask = data['train_mask'].numpy()[i]

            # 确认输入图片没有问题
            concat = [img]
            for x in [dilated_segment, shrunk_segment, threshold, train_mask]:
                x = norm(x)
                concat.append(np.stack([x] * 3, -1))
            concat = np.concatenate(concat, 1)

            # 测试后处理
            pred = np.stack([shrunk_segment, threshold], axis=0)[None, ...]
            boxes, scores = post_process([[640, 640]], pred, is_output_polygon=False)

            # 确认后处理没有问题
            img = draw_boxes_on_img(img, boxes)
            cv2.imshow("img", img)
            cv2.imshow('concat', concat)
            cv2.waitKey(0)
