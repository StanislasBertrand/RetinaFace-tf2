import cv2
import numpy as np
from retinaface import RetinaFace

thresh = 0.85
detector = RetinaFace("data/retinafaceweights.npy", None)
img = cv2.imread('sample-images/random_internet_selfie.jpg')
faces, landmarks = detector.detect(img, thresh)

if faces is not None:
  print('found', faces.shape[0], 'faces')
  for i in range(faces.shape[0]):
    box = faces[i].astype(np.int)
    color = (0,0,255)
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 1)
    if landmarks is not None:
      landmark5 = landmarks[i].astype(np.int)
      for l in range(landmark5.shape[0]):
        color = (0,0,255)
        if l==0 or l==3:
          color = (0,255,0)
        cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 1)

  filename = './retinaface_tf2_output.jpg'
  print('writing output to ', filename)
  cv2.imwrite(filename, img)

