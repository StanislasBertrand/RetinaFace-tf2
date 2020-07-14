from __future__ import print_function
import numpy as np
import cv2
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from networks.retinaface_network import RetinaFaceNetwork
import time

class RetinaFace:
    def __init__(self, model_weights, use_gpu_nms=True, nms=0.4, decay4=0.5):
        self.decay4 = decay4
        self.nms_threshold = nms
        self.fpn_keys = []
        self.anchor_cfg = None
        pixel_means=[0.0, 0.0, 0.0]
        pixel_stds=[1.0, 1.0, 1.0]
        pixel_scale = 1.0
        self.preprocess = False
        _ratio = (1.,)
        self._feat_stride_fpn = [32, 16, 8]
        self.anchor_cfg = {
          '32': {'SCALES': (32,16), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '16': {'SCALES': (8,4), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
          '8': {'SCALES': (2,1), 'BASE_SIZE': 16, 'RATIOS': _ratio, 'ALLOWED_BORDER': 9999},
        }
        for s in self._feat_stride_fpn:
            self.fpn_keys.append('stride%s'%s)
        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=False, cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
            v = self._anchors_fpn[k].astype(np.float32)
            self._anchors_fpn[k] = v

        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        if use_gpu_nms:
            self.nms = gpu_nms_wrapper(self.nms_threshold, 0)
        else:
            self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)
        self.use_landmarks = True
        self.cascade = 0
        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]


        self.model = RetinaFaceNetwork(model_weights).model


    def detect(self, img, threshold=0.5, im_scale = 1.0):
        proposals_list = []
        scores_list = []
        landmarks_list = []

        if im_scale != 1.0:
            img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        im_info = [img.shape[0], img.shape[1]]
        im_tensor = np.zeros((1, 3, img.shape[0], img.shape[1]))
        for i in range(3):
            im_tensor[0, i, :, :] = (img[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[
                2 - i]
        net_out = self.model.predict(im_tensor.transpose(0, 2, 3, 1))
        net_out_reshaped = []
        for elt in net_out:
            net_out_reshaped.append(elt.transpose(0, 3, 1, 2))
        net_out = net_out_reshaped
        sym_idx = 0

        for _idx,s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s'%s
            stride = int(s)
            scores = net_out[sym_idx]
            scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[2], bbox_deltas.shape[3]

            A = self._num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, stride, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1))
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * self.bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * self.bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * self.bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * self.bbox_stds[3]
            proposals = self.bbox_pred(anchors, bbox_deltas)

            proposals = clip_boxes(proposals, im_info[:2])


            if stride==4 and self.decay4<1.0:
                scores *= self.decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= im_scale
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[1]//A
            landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
            landmarks = self.landmark_pred(anchors, landmark_deltas)
            landmarks = landmarks[order, :]

            landmarks[:, :, 0:2] /= im_scale
            landmarks_list.append(landmarks)
            sym_idx += 3

        proposals = np.vstack(proposals_list)
        if proposals.shape[0]==0:
            landmarks = np.zeros( (0,5,2) )
            return np.zeros( (0,5) ), landmarks
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        proposals = proposals[order, :]
        scores = scores[order]
        landmarks = np.vstack(landmarks_list)
        landmarks = landmarks[order].astype(np.float32, copy=False)

        pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
        keep = self.nms(pre_det)
        det = np.hstack( (pre_det, proposals[:,4:]) )
        det = det[keep, :]
        landmarks = landmarks[keep]

        return det, landmarks

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _filter_boxes2(boxes, max_size, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size>0:
            keep = np.where( np.minimum(ws, hs)<max_size )[0]
        elif min_size>0:
            keep = np.where( np.maximum(ws, hs)>min_size )[0]
        return keep

    @staticmethod
    def _clip_pad(tensor, pad_shape):
        """
        Clip boxes of the pad area.
        :param tensor: [n, c, H, W]
        :param pad_shape: [h, w]
        :return: [n, c, h, w]
        """
        H, W = tensor.shape[2:]
        h, w = pad_shape

        if h < H or w < W:
            tensor = tensor[:, :, :h, :w].copy()

        return tensor

    @staticmethod
    def bbox_pred(boxes, box_deltas):
        """
        Transform the set of class-agnostic boxes into class-specific boxes
        by applying the predicted offsets (box_deltas)
        :param boxes: !important [N 4]
        :param box_deltas: [N, 4 * num_classes]
        :return: [N 4 * num_classes]
        """
        if boxes.shape[0] == 0:
            return np.zeros((0, box_deltas.shape[1]))

        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)

        dx = box_deltas[:, 0:1]
        dy = box_deltas[:, 1:2]
        dw = box_deltas[:, 2:3]
        dh = box_deltas[:, 3:4]

        pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
        pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
        pred_w = np.exp(dw) * widths[:, np.newaxis]
        pred_h = np.exp(dh) * heights[:, np.newaxis]

        pred_boxes = np.zeros(box_deltas.shape)
        # x1
        pred_boxes[:, 0:1] = pred_ctr_x - 0.5 * (pred_w - 1.0)
        # y1
        pred_boxes[:, 1:2] = pred_ctr_y - 0.5 * (pred_h - 1.0)
        # x2
        pred_boxes[:, 2:3] = pred_ctr_x + 0.5 * (pred_w - 1.0)
        # y2
        pred_boxes[:, 3:4] = pred_ctr_y + 0.5 * (pred_h - 1.0)

        if box_deltas.shape[1]>4:
            pred_boxes[:,4:] = box_deltas[:,4:]

        return pred_boxes

    @staticmethod
    def landmark_pred(boxes, landmark_deltas):
        if boxes.shape[0] == 0:
          return np.zeros((0, landmark_deltas.shape[1]))
        boxes = boxes.astype(np.float, copy=False)
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * (widths - 1.0)
        ctr_y = boxes[:, 1] + 0.5 * (heights - 1.0)
        pred = landmark_deltas.copy()
        for i in range(5):
            pred[:,i,0] = landmark_deltas[:,i,0]*widths + ctr_x
            pred[:,i,1] = landmark_deltas[:,i,1]*heights + ctr_y
        return pred


    def bbox_vote(self, det):
        if det.shape[0] == 0:
            return np.zeros( (0, 5) )
        dets = None
        while det.shape[0] > 0:
            if dets is not None and dets.shape[0]>=750:
                break
            # IOU
            area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
            xx1 = np.maximum(det[0, 0], det[:, 0])
            yy1 = np.maximum(det[0, 1], det[:, 1])
            xx2 = np.minimum(det[0, 2], det[:, 2])
            yy2 = np.minimum(det[0, 3], det[:, 3])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            o = inter / (area[0] + area[:] - inter)

            # nms
            merge_index = np.where(o >= self.nms_threshold)[0]
            det_accu = det[merge_index, :]
            det = np.delete(det, merge_index, 0)
            if merge_index.shape[0] <= 1:
                if det.shape[0] == 0:
                    try:
                        dets = np.row_stack((dets, det_accu))
                    except:
                        dets = det_accu
                        continue
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4],
                                        axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            if dets is None:
                dets = det_accu_sum
            else:
                dets = np.row_stack((dets, det_accu_sum))
        dets = dets[0:750, :]
        return dets