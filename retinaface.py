from __future__ import print_function
import numpy as np
import tensorflow as tf

import cv2
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from networks.retinaface_network import RetinaFaceNetwork

class RetinaFace:
    def __init__(self, model_weights, use_gpu_nms=True, nms=0.4, decay4=0.5):
        self.decay4 = decay4
        self.nms_threshold = nms
        self.fpn_keys = []
        self.anchor_cfg = None
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
        pixel_means=[0.0, 0.0, 0.0]
        pixel_stds=[1.0, 1.0, 1.0]
        pixel_scale = 1.0
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)
        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]
        self.scales = [1024, 1980]
        self.model = tf.function(
            RetinaFaceNetwork(model_weights).model,
            input_signature=(tf.TensorSpec(shape=[None, None, None, 3], dtype=np.float32),)
        )

    def _resize_image(self, img):
        img_w, img_h = img.shape[0:2]
        target_size = self.scales[0]
        max_size = self.scales[1]

        if img_w > img_h:
            im_size_min, im_size_max = img_h, img_w
        else:
            im_size_min, im_size_max = img_w, img_h

        im_scale = target_size / float(im_size_min)

        if np.round(im_scale * im_size_max) > max_size:
            im_scale = max_size / float(im_size_max)

        if im_scale != 1.0:
            img = cv2.resize(
                img,
                None,
                None,
                fx=im_scale,
                fy=im_scale,
                interpolation=cv2.INTER_LINEAR
            )

        return img, im_scale

    def _preprocess_image(self, img):
        """
        Preprocess an image applying resizing, scaling and channel conversion
        :param img: np.ndarray
            A BGR image to preprocess with shape (H, W, C)
        :return: Tuple[np.ndarray, Tuple[int], float]
            A tuple that contains 3 elements:
            * The preprocessed image with shape (1, H, W, C).
              This image has now RBG channels' format
            * The image resized height and width
            * The resized scale
        """
        img, im_scale = self._resize_image(img)
        img = img.astype(np.float32)
        im_tensor = np.zeros((1, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        # Make image scaling + BGR2RGB conversion + transpose (N,H,W,C) to (N,C,H,W)
        for i in range(3):
            im_tensor[0, :, :, i] = (img[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / self.pixel_stds[2 - i]

        return im_tensor, img.shape[0:2], im_scale

    def detect(self, img, threshold=0.5):
        """
        Detect all the faces and landmarks in an image
        :param img: input image
        :param threshold: detection threshold
        :return: tuple faces, landmarks
        """
        proposals_list = []
        scores_list = []
        landmarks_list = []
        im_tensor, im_info, im_scale = self._preprocess_image(img)
        net_out = self.model(im_tensor)
        net_out = [elt.numpy() for elt in net_out]
        sym_idx = 0

        for _idx, s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s'%s
            scores = net_out[sym_idx]
            scores = scores[:, :, :, self._num_anchors['stride%s'%s]:]

            bbox_deltas = net_out[sym_idx + 1]
            height, width = bbox_deltas.shape[1], bbox_deltas.shape[2]

            A = self._num_anchors['stride%s'%s]
            K = height * width
            anchors_fpn = self._anchors_fpn['stride%s'%s]
            anchors = anchors_plane(height, width, s, anchors_fpn)
            anchors = anchors.reshape((K * A, 4))
            scores = scores.reshape((-1, 1))

            bbox_deltas = bbox_deltas
            bbox_pred_len = bbox_deltas.shape[3]//A
            bbox_deltas = bbox_deltas.reshape((-1, bbox_pred_len))
            bbox_deltas[:, 0::4] = bbox_deltas[:,0::4] * self.bbox_stds[0]
            bbox_deltas[:, 1::4] = bbox_deltas[:,1::4] * self.bbox_stds[1]
            bbox_deltas[:, 2::4] = bbox_deltas[:,2::4] * self.bbox_stds[2]
            bbox_deltas[:, 3::4] = bbox_deltas[:,3::4] * self.bbox_stds[3]
            proposals = self.bbox_pred(anchors, bbox_deltas)

            proposals = clip_boxes(proposals, im_info[:2])


            if s==4 and self.decay4<1.0:
                scores *= self.decay4

            scores_ravel = scores.ravel()
            order = np.where(scores_ravel>=threshold)[0]
            proposals = proposals[order, :]
            scores = scores[order]

            proposals[:, 0:4] /= im_scale
            proposals_list.append(proposals)
            scores_list.append(scores)

            landmark_deltas = net_out[sym_idx + 2]
            landmark_pred_len = landmark_deltas.shape[3]//A
            landmark_deltas = landmark_deltas.reshape((-1, 5, landmark_pred_len//5))
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