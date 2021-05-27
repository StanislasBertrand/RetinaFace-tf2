import os
import cv2
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
from retinaface import RetinaFace

flags.DEFINE_string('widerface_data_dir', '/home/bertrans/Downloads/WIDER_val/images/', 'data directory of widerface test set')
flags.DEFINE_string('save_folder', './WiderFace-Evaluation/results_val/',
                    'folder path to save evaluate results')


def _main(_argv):
    detector = RetinaFace(use_gpu_nms = False)
    if not os.path.isdir(FLAGS.save_folder):
        os.mkdir(FLAGS.save_folder)
    subdirs = [x[0] for x in os.walk(FLAGS.widerface_data_dir)][1:]
    save_dir = FLAGS.save_folder
    for subdir in subdirs:
        print(subdir)
        output_dir = os.path.join(save_dir, subdir.split("/")[-1])
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for file in os.listdir(subdir):
            if os.path.isfile(os.path.join(output_dir, file.replace("jpg", "txt"))):
                continue
            img = cv2.imread(os.path.join(subdir, file))
            faces, ldmks = detector.detect(img, 0.01)
            with open(os.path.join(output_dir, file.replace("jpg", "txt")), "w+") as f:
                f.write(file.split("/")[-1].split(".")[0] + "\n")
                f.write(str(len(faces)) + "\n")
                for face in faces:
                    f.write(str(int(face[0]))
                            + " "
                            + str(int(face[1]))
                            + " "
                            + str(int(face[2]) - int(face[0]))
                            + " "
                            + str(int(face[3]) - int(face[1]))
                            + " "
                            + str(face[4])
                            + "\n")

if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass