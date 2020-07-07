from __future__ import print_function
import numpy as np
from rcnn.processing.bbox_transform import clip_boxes
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from networks.retinaface_network import RetinaFaceNetwork

class RetinaFace:
    def __init__(self, weights_path, img_size, nms=0.4, nocrop=False, decay4 = 0.5, vote=False):
        self.decay4 = decay4
        self.nms_threshold = nms
        self.vote = vote
        self.nocrop = nocrop
        self.debug = False
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
        dense_anchor = False
        self._anchors_fpn = dict(zip(self.fpn_keys, generate_anchors_fpn(dense_anchor=dense_anchor, cfg=self.anchor_cfg)))
        for k in self._anchors_fpn:
          v = self._anchors_fpn[k].astype(np.float32)
          self._anchors_fpn[k] = v

        self._num_anchors = dict(zip(self.fpn_keys, [anchors.shape[0] for anchors in self._anchors_fpn.values()]))
        # if use_gpu:
        #   self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)
        # else:
        #   self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.nms = cpu_nms_wrapper(self.nms_threshold)
        self.pixel_means = np.array(pixel_means, dtype=np.float32)
        self.pixel_stds = np.array(pixel_stds, dtype=np.float32)
        self.pixel_scale = float(pixel_scale)
        self.use_landmarks = True
        self.cascade = 0
        self.bbox_stds = [1.0, 1.0, 1.0, 1.0]
        self.landmark_std = 1.0
        self.model = RetinaFaceNetwork(weights_path).model


    def detect(self, img, threshold=0.5):
        proposals_list = []
        scores_list = []
        landmarks_list = []
        strides_list = []
        im = img.copy()
        im = im.astype(np.float32)
        im_info = [im.shape[0], im.shape[1]]
        im_tensor = img[np.newaxis, :, :, :]

        net_out = self.model.predict(im_tensor)
        net_out_reshaped = []
        for elt in net_out:
            net_out_reshaped.append(elt.transpose(0,3,1,2))

        net_out = net_out_reshaped
        sym_idx = 0

        for _idx,s in enumerate(self._feat_stride_fpn):
            _key = 'stride%s'%s
            stride = int(s)

            scores = net_out[sym_idx]

            scores = scores[:, self._num_anchors['stride%s'%s]:, :, :]

            bbox_deltas = net_out[sym_idx+1]

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


            proposals[:,0:4] /= 1.0

            proposals_list.append(proposals)
            scores_list.append(scores)
            if self.nms_threshold<0.0:
                _strides = np.empty(shape=(scores.shape), dtype=np.float32)
                _strides.fill(stride)
                strides_list.append(_strides)

            if not self.vote and self.use_landmarks:
                landmark_deltas = net_out[sym_idx+2]
                landmark_pred_len = landmark_deltas.shape[1]//A
                landmark_deltas = landmark_deltas.transpose((0, 2, 3, 1)).reshape((-1, 5, landmark_pred_len//5))
                landmark_deltas *= self.landmark_std
                landmarks = self.landmark_pred(anchors, landmark_deltas)
                landmarks = landmarks[order, :]


            landmarks[:,:,0:2] /= 1.0

            landmarks_list.append(landmarks)
            if self.use_landmarks:
                sym_idx += 3
            else:
                sym_idx += 2


        proposals = np.vstack(proposals_list)
        landmarks = None
        if proposals.shape[0]==0:
            if self.use_landmarks:
                landmarks = np.zeros( (0,5,2) )
            if self.nms_threshold<0.0:
                return np.zeros( (0,6) ), landmarks
            else:
                return np.zeros( (0,5) ), landmarks

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        proposals = proposals[order, :]
        scores = scores[order]
        if self.nms_threshold<0.0:
            strides = np.vstack(strides_list)
            strides = strides[order]
        if not self.vote and self.use_landmarks:
            landmarks = np.vstack(landmarks_list)
            landmarks = landmarks[order].astype(np.float32, copy=False)

        if self.nms_threshold>0.0:
            pre_det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)
            if not self.vote:
                keep = self.nms(pre_det)
                det = np.hstack( (pre_det, proposals[:,4:]) )
                det = det[keep, :]
            if self.use_landmarks:
                landmarks = landmarks[keep]
            else:
                det = np.hstack( (pre_det, proposals[:,4:]) )
                det = self.bbox_vote(det)
        elif self.nms_threshold<0.0:
            det = np.hstack((proposals[:,0:4], scores, strides)).astype(np.float32, copy=False)
        else:
            det = np.hstack((proposals[:,0:4], scores)).astype(np.float32, copy=False)


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