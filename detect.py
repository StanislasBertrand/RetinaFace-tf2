import cv2
import numpy as np
from absl import app, flags
from absl.flags import FLAGS
from src.retinafacetf2.retinaface import RetinaFace

flags.DEFINE_string('sample_img', './sample-images/WC_FR.jpeg', 'image to test on')
flags.DEFINE_string('save_destination', 'retinaface_tf2_output.jpg', "destination image")
flags.DEFINE_float('det_thresh', 0.9, "detection threshold")
flags.DEFINE_float('nms_thresh', 0.4, "nms threshold")
flags.DEFINE_bool('use_gpu_nms', False, "whether to use gpu for nms")


def _main(_argv):
    detector = RetinaFace(FLAGS.use_gpu_nms, FLAGS.nms_thresh)
    img = cv2.imread(FLAGS.sample_img)
    faces, landmarks = detector.detect(img, FLAGS.det_thresh)
    if faces is not None:
        print('found', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 1)

    cv2.imwrite(FLAGS.save_destination, img)


if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass
