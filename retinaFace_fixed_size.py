import tensorflow as tf
import numpy as np


def load_weights(weight_file):
    import numpy as np
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict


class RetinaFace(object):
    def __init__(self, weights_path, size):
        assert size%32==0
        self.size = size
        self.weights_dict = load_weights(weights_path)
        self.model = self.load_model()

    def load_model(self):
        data = tf.keras.Input(dtype=tf.float32, shape=(self.size, self.size, 3), name='data')
        bn_data = self.batch_normalization(data, variance_epsilon=1.9999999494757503e-05, name='bn_data')
        conv0_pad = self.pad(bn_data, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
        conv0 = self.convolution(conv0_pad, self.weights_dict, strides=[2, 2], padding='VALID', name='conv0')
        bn0 = self.batch_normalization(conv0, variance_epsilon=1.9999999494757503e-05, name='bn0')
        relu0 = self.relu(bn0, name='relu0')
        pooling0_pad = self.pad(relu0, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        pooling0 = tf.keras.layers.MaxPool2D((3, 3), (2, 2), padding='VALID', name='pooling0')(pooling0_pad)
        stage1_unit1_bn1 = self.batch_normalization(pooling0, variance_epsilon=1.9999999494757503e-05, name='stage1_unit1_bn1')
        stage1_unit1_relu1 = self.relu(stage1_unit1_bn1, name='stage1_unit1_relu1')
        stage1_unit1_conv1 = self.convolution(stage1_unit1_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit1_conv1')
        stage1_unit1_sc = self.convolution(stage1_unit1_relu1, self.weights_dict, strides=[1, 1], padding='VALID', name='stage1_unit1_sc')
        stage1_unit1_bn2 = self.batch_normalization(stage1_unit1_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage1_unit1_bn2')
        stage1_unit1_relu2 = self.relu(stage1_unit1_bn2, name='stage1_unit1_relu2')
        stage1_unit1_conv2_pad = self.pad(stage1_unit1_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage1_unit1_conv2 = self.convolution(stage1_unit1_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit1_conv2')
        stage1_unit1_bn3 = self.batch_normalization(stage1_unit1_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage1_unit1_bn3')
        stage1_unit1_relu3 = self.relu(stage1_unit1_bn3, name='stage1_unit1_relu3')
        stage1_unit1_conv3 = self.convolution(stage1_unit1_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit1_conv3')
        plus0_v1 = stage1_unit1_conv3 + stage1_unit1_sc
        stage1_unit2_bn1 = self.batch_normalization(plus0_v1, variance_epsilon=1.9999999494757503e-05, name='stage1_unit2_bn1')
        stage1_unit2_relu1 = self.relu(stage1_unit2_bn1, name='stage1_unit2_relu1')
        stage1_unit2_conv1 = self.convolution(stage1_unit2_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit2_conv1')
        stage1_unit2_bn2 = self.batch_normalization(stage1_unit2_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage1_unit2_bn2')
        stage1_unit2_relu2 = self.relu(stage1_unit2_bn2, name='stage1_unit2_relu2')
        stage1_unit2_conv2_pad = self.pad(stage1_unit2_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage1_unit2_conv2 = self.convolution(stage1_unit2_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit2_conv2')
        stage1_unit2_bn3 = self.batch_normalization(stage1_unit2_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage1_unit2_bn3')
        stage1_unit2_relu3 = self.relu(stage1_unit2_bn3, name='stage1_unit2_relu3')
        stage1_unit2_conv3 = self.convolution(stage1_unit2_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit2_conv3')
        plus1_v2 = stage1_unit2_conv3 + plus0_v1
        stage1_unit3_bn1 = self.batch_normalization(plus1_v2, variance_epsilon=1.9999999494757503e-05, name='stage1_unit3_bn1')
        stage1_unit3_relu1 = self.relu(stage1_unit3_bn1, name='stage1_unit3_relu1')
        stage1_unit3_conv1 = self.convolution(stage1_unit3_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit3_conv1')
        stage1_unit3_bn2 = self.batch_normalization(stage1_unit3_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage1_unit3_bn2')
        stage1_unit3_relu2 = self.relu(stage1_unit3_bn2, name='stage1_unit3_relu2')
        stage1_unit3_conv2_pad = self.pad(stage1_unit3_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage1_unit3_conv2 = self.convolution(stage1_unit3_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit3_conv2')
        stage1_unit3_bn3 = self.batch_normalization(stage1_unit3_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage1_unit3_bn3')
        stage1_unit3_relu3 = self.relu(stage1_unit3_bn3, name='stage1_unit3_relu3')
        stage1_unit3_conv3 = self.convolution(stage1_unit3_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage1_unit3_conv3')
        plus2 = stage1_unit3_conv3 + plus1_v2
        stage2_unit1_bn1 = self.batch_normalization(plus2, variance_epsilon=1.9999999494757503e-05, name='stage2_unit1_bn1')
        stage2_unit1_relu1 = self.relu(stage2_unit1_bn1, name='stage2_unit1_relu1')
        stage2_unit1_conv1 = self.convolution(stage2_unit1_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit1_conv1')
        stage2_unit1_sc = self.convolution(stage2_unit1_relu1, self.weights_dict, strides=[2, 2], padding='VALID', name='stage2_unit1_sc')
        stage2_unit1_bn2 = self.batch_normalization(stage2_unit1_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit1_bn2')
        stage2_unit1_relu2 = self.relu(stage2_unit1_bn2, name='stage2_unit1_relu2')
        stage2_unit1_conv2_pad = self.pad(stage2_unit1_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage2_unit1_conv2 = self.convolution(stage2_unit1_conv2_pad, self.weights_dict, strides=[2, 2], padding='VALID',
                                         name='stage2_unit1_conv2')
        stage2_unit1_bn3 = self.batch_normalization(stage2_unit1_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit1_bn3')
        stage2_unit1_relu3 = self.relu(stage2_unit1_bn3, name='stage2_unit1_relu3')
        stage2_unit1_conv3 = self.convolution(stage2_unit1_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit1_conv3')
        plus3 = stage2_unit1_conv3 + stage2_unit1_sc
        stage2_unit2_bn1 = self.batch_normalization(plus3, variance_epsilon=1.9999999494757503e-05, name='stage2_unit2_bn1')
        stage2_unit2_relu1 = self.relu(stage2_unit2_bn1, name='stage2_unit2_relu1')
        stage2_unit2_conv1 = self.convolution(stage2_unit2_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit2_conv1')
        stage2_unit2_bn2 = self.batch_normalization(stage2_unit2_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit2_bn2')
        stage2_unit2_relu2 = self.relu(stage2_unit2_bn2, name='stage2_unit2_relu2')
        stage2_unit2_conv2_pad = self.pad(stage2_unit2_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage2_unit2_conv2 = self.convolution(stage2_unit2_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit2_conv2')
        stage2_unit2_bn3 = self.batch_normalization(stage2_unit2_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit2_bn3')
        stage2_unit2_relu3 = self.relu(stage2_unit2_bn3, name='stage2_unit2_relu3')
        stage2_unit2_conv3 = self.convolution(stage2_unit2_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit2_conv3')
        plus4 = stage2_unit2_conv3 + plus3
        stage2_unit3_bn1 = self.batch_normalization(plus4, variance_epsilon=1.9999999494757503e-05, name='stage2_unit3_bn1')
        stage2_unit3_relu1 = self.relu(stage2_unit3_bn1, name='stage2_unit3_relu1')
        stage2_unit3_conv1 = self.convolution(stage2_unit3_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit3_conv1')
        stage2_unit3_bn2 = self.batch_normalization(stage2_unit3_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit3_bn2')
        stage2_unit3_relu2 = self.relu(stage2_unit3_bn2, name='stage2_unit3_relu2')
        stage2_unit3_conv2_pad = self.pad(stage2_unit3_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage2_unit3_conv2 = self.convolution(stage2_unit3_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit3_conv2')
        stage2_unit3_bn3 = self.batch_normalization(stage2_unit3_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit3_bn3')
        stage2_unit3_relu3 = self.relu(stage2_unit3_bn3, name='stage2_unit3_relu3')
        stage2_unit3_conv3 = self.convolution(stage2_unit3_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit3_conv3')
        plus5 = stage2_unit3_conv3 + plus4
        stage2_unit4_bn1 = self.batch_normalization(plus5, variance_epsilon=1.9999999494757503e-05, name='stage2_unit4_bn1')
        stage2_unit4_relu1 = self.relu(stage2_unit4_bn1, name='stage2_unit4_relu1')
        stage2_unit4_conv1 = self.convolution(stage2_unit4_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit4_conv1')
        stage2_unit4_bn2 = self.batch_normalization(stage2_unit4_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit4_bn2')
        stage2_unit4_relu2 = self.relu(stage2_unit4_bn2, name='stage2_unit4_relu2')
        stage2_unit4_conv2_pad = self.pad(stage2_unit4_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage2_unit4_conv2 = self.convolution(stage2_unit4_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit4_conv2')
        stage2_unit4_bn3 = self.batch_normalization(stage2_unit4_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage2_unit4_bn3')
        stage2_unit4_relu3 = self.relu(stage2_unit4_bn3, name='stage2_unit4_relu3')
        stage2_unit4_conv3 = self.convolution(stage2_unit4_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage2_unit4_conv3')
        plus6 = stage2_unit4_conv3 + plus5
        stage3_unit1_bn1 = self.batch_normalization(plus6, variance_epsilon=1.9999999494757503e-05, name='stage3_unit1_bn1')
        stage3_unit1_relu1 = self.relu(stage3_unit1_bn1, name='stage3_unit1_relu1')
        stage3_unit1_conv1 = self.convolution(stage3_unit1_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit1_conv1')
        stage3_unit1_sc = self.convolution(stage3_unit1_relu1, self.weights_dict, strides=[2, 2], padding='VALID', name='stage3_unit1_sc')
        stage3_unit1_bn2 = self.batch_normalization(stage3_unit1_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit1_bn2')
        stage3_unit1_relu2 = self.relu(stage3_unit1_bn2, name='stage3_unit1_relu2')
        stage3_unit1_conv2_pad = self.pad(stage3_unit1_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage3_unit1_conv2 = self.convolution(stage3_unit1_conv2_pad, self.weights_dict, strides=[2, 2], padding='VALID',
                                         name='stage3_unit1_conv2')
        ssh_m1_red_conv = self.convolution(stage3_unit1_relu2, self.weights_dict, strides=[1, 1], padding='VALID', name='ssh_m1_red_conv')
        stage3_unit1_bn3 = self.batch_normalization(stage3_unit1_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit1_bn3')
        ssh_m1_red_conv_bn = self.batch_normalization(ssh_m1_red_conv, variance_epsilon=1.9999999494757503e-05,
                                                 name='ssh_m1_red_conv_bn')
        stage3_unit1_relu3 = self.relu(stage3_unit1_bn3, name='stage3_unit1_relu3')
        ssh_m1_red_conv_relu = self.relu(ssh_m1_red_conv_bn, name='ssh_m1_red_conv_relu')
        stage3_unit1_conv3 = self.convolution(stage3_unit1_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit1_conv3')
        plus7 = stage3_unit1_conv3 + stage3_unit1_sc
        stage3_unit2_bn1 = self.batch_normalization(plus7, variance_epsilon=1.9999999494757503e-05, name='stage3_unit2_bn1')
        stage3_unit2_relu1 = self.relu(stage3_unit2_bn1, name='stage3_unit2_relu1')
        stage3_unit2_conv1 = self.convolution(stage3_unit2_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit2_conv1')
        stage3_unit2_bn2 = self.batch_normalization(stage3_unit2_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit2_bn2')
        stage3_unit2_relu2 = self.relu(stage3_unit2_bn2, name='stage3_unit2_relu2')
        stage3_unit2_conv2_pad = self.pad(stage3_unit2_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage3_unit2_conv2 = self.convolution(stage3_unit2_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit2_conv2')
        stage3_unit2_bn3 = self.batch_normalization(stage3_unit2_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit2_bn3')
        stage3_unit2_relu3 = self.relu(stage3_unit2_bn3, name='stage3_unit2_relu3')
        stage3_unit2_conv3 = self.convolution(stage3_unit2_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit2_conv3')
        plus8 = stage3_unit2_conv3 + plus7
        stage3_unit3_bn1 = self.batch_normalization(plus8, variance_epsilon=1.9999999494757503e-05, name='stage3_unit3_bn1')
        stage3_unit3_relu1 = self.relu(stage3_unit3_bn1, name='stage3_unit3_relu1')
        stage3_unit3_conv1 = self.convolution(stage3_unit3_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit3_conv1')
        stage3_unit3_bn2 = self.batch_normalization(stage3_unit3_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit3_bn2')
        stage3_unit3_relu2 = self.relu(stage3_unit3_bn2, name='stage3_unit3_relu2')
        stage3_unit3_conv2_pad = self.pad(stage3_unit3_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage3_unit3_conv2 = self.convolution(stage3_unit3_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit3_conv2')
        stage3_unit3_bn3 = self.batch_normalization(stage3_unit3_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit3_bn3')
        stage3_unit3_relu3 = self.relu(stage3_unit3_bn3, name='stage3_unit3_relu3')
        stage3_unit3_conv3 = self.convolution(stage3_unit3_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit3_conv3')
        plus9 = stage3_unit3_conv3 + plus8
        stage3_unit4_bn1 = self.batch_normalization(plus9, variance_epsilon=1.9999999494757503e-05, name='stage3_unit4_bn1')
        stage3_unit4_relu1 = self.relu(stage3_unit4_bn1, name='stage3_unit4_relu1')
        stage3_unit4_conv1 = self.convolution(stage3_unit4_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit4_conv1')
        stage3_unit4_bn2 = self.batch_normalization(stage3_unit4_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit4_bn2')
        stage3_unit4_relu2 = self.relu(stage3_unit4_bn2, name='stage3_unit4_relu2')
        stage3_unit4_conv2_pad = self.pad(stage3_unit4_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage3_unit4_conv2 = self.convolution(stage3_unit4_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit4_conv2')
        stage3_unit4_bn3 = self.batch_normalization(stage3_unit4_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit4_bn3')
        stage3_unit4_relu3 = self.relu(stage3_unit4_bn3, name='stage3_unit4_relu3')
        stage3_unit4_conv3 = self.convolution(stage3_unit4_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit4_conv3')
        plus10 = stage3_unit4_conv3 + plus9
        stage3_unit5_bn1 = self.batch_normalization(plus10, variance_epsilon=1.9999999494757503e-05, name='stage3_unit5_bn1')
        stage3_unit5_relu1 = self.relu(stage3_unit5_bn1, name='stage3_unit5_relu1')
        stage3_unit5_conv1 = self.convolution(stage3_unit5_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit5_conv1')
        stage3_unit5_bn2 = self.batch_normalization(stage3_unit5_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit5_bn2')
        stage3_unit5_relu2 = self.relu(stage3_unit5_bn2, name='stage3_unit5_relu2')
        stage3_unit5_conv2_pad = self.pad(stage3_unit5_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage3_unit5_conv2 = self.convolution(stage3_unit5_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit5_conv2')
        stage3_unit5_bn3 = self.batch_normalization(stage3_unit5_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit5_bn3')
        stage3_unit5_relu3 = self.relu(stage3_unit5_bn3, name='stage3_unit5_relu3')
        stage3_unit5_conv3 = self.convolution(stage3_unit5_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit5_conv3')
        plus11 = stage3_unit5_conv3 + plus10
        stage3_unit6_bn1 = self.batch_normalization(plus11, variance_epsilon=1.9999999494757503e-05, name='stage3_unit6_bn1')
        stage3_unit6_relu1 = self.relu(stage3_unit6_bn1, name='stage3_unit6_relu1')
        stage3_unit6_conv1 = self.convolution(stage3_unit6_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit6_conv1')
        stage3_unit6_bn2 = self.batch_normalization(stage3_unit6_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit6_bn2')
        stage3_unit6_relu2 = self.relu(stage3_unit6_bn2, name='stage3_unit6_relu2')
        stage3_unit6_conv2_pad = self.pad(stage3_unit6_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage3_unit6_conv2 = self.convolution(stage3_unit6_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit6_conv2')
        stage3_unit6_bn3 = self.batch_normalization(stage3_unit6_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage3_unit6_bn3')
        stage3_unit6_relu3 = self.relu(stage3_unit6_bn3, name='stage3_unit6_relu3')
        stage3_unit6_conv3 = self.convolution(stage3_unit6_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage3_unit6_conv3')
        plus12 = stage3_unit6_conv3 + plus11
        stage4_unit1_bn1 = self.batch_normalization(plus12, variance_epsilon=1.9999999494757503e-05, name='stage4_unit1_bn1')
        stage4_unit1_relu1 = self.relu(stage4_unit1_bn1, name='stage4_unit1_relu1')
        stage4_unit1_conv1 = self.convolution(stage4_unit1_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit1_conv1')
        stage4_unit1_sc = self.convolution(stage4_unit1_relu1, self.weights_dict, strides=[2, 2], padding='VALID', name='stage4_unit1_sc')
        stage4_unit1_bn2 = self.batch_normalization(stage4_unit1_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage4_unit1_bn2')
        stage4_unit1_relu2 = self.relu(stage4_unit1_bn2, name='stage4_unit1_relu2')
        stage4_unit1_conv2_pad = self.pad(stage4_unit1_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage4_unit1_conv2 = self.convolution(stage4_unit1_conv2_pad, self.weights_dict, strides=[2, 2], padding='VALID',
                                         name='stage4_unit1_conv2')
        ssh_c2_lateral = self.convolution(stage4_unit1_relu2, self.weights_dict, strides=[1, 1], padding='VALID', name='ssh_c2_lateral')
        stage4_unit1_bn3 = self.batch_normalization(stage4_unit1_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage4_unit1_bn3')
        ssh_c2_lateral_bn = self.batch_normalization(ssh_c2_lateral, variance_epsilon=1.9999999494757503e-05,
                                                name='ssh_c2_lateral_bn')
        stage4_unit1_relu3 = self.relu(stage4_unit1_bn3, name='stage4_unit1_relu3')
        ssh_c2_lateral_relu = self.relu(ssh_c2_lateral_bn, name='ssh_c2_lateral_relu')
        stage4_unit1_conv3 = self.convolution(stage4_unit1_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit1_conv3')
        plus13 = stage4_unit1_conv3 + stage4_unit1_sc
        stage4_unit2_bn1 = self.batch_normalization(plus13, variance_epsilon=1.9999999494757503e-05, name='stage4_unit2_bn1')
        stage4_unit2_relu1 = self.relu(stage4_unit2_bn1, name='stage4_unit2_relu1')
        stage4_unit2_conv1 = self.convolution(stage4_unit2_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit2_conv1')
        stage4_unit2_bn2 = self.batch_normalization(stage4_unit2_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage4_unit2_bn2')
        stage4_unit2_relu2 = self.relu(stage4_unit2_bn2, name='stage4_unit2_relu2')
        stage4_unit2_conv2_pad = self.pad(stage4_unit2_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage4_unit2_conv2 = self.convolution(stage4_unit2_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit2_conv2')
        stage4_unit2_bn3 = self.batch_normalization(stage4_unit2_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage4_unit2_bn3')
        stage4_unit2_relu3 = self.relu(stage4_unit2_bn3, name='stage4_unit2_relu3')
        stage4_unit2_conv3 = self.convolution(stage4_unit2_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit2_conv3')
        plus14 = stage4_unit2_conv3 + plus13
        stage4_unit3_bn1 = self.batch_normalization(plus14, variance_epsilon=1.9999999494757503e-05, name='stage4_unit3_bn1')
        stage4_unit3_relu1 = self.relu(stage4_unit3_bn1, name='stage4_unit3_relu1')
        stage4_unit3_conv1 = self.convolution(stage4_unit3_relu1, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit3_conv1')
        stage4_unit3_bn2 = self.batch_normalization(stage4_unit3_conv1, variance_epsilon=1.9999999494757503e-05,
                                               name='stage4_unit3_bn2')
        stage4_unit3_relu2 = self.relu(stage4_unit3_bn2, name='stage4_unit3_relu2')
        stage4_unit3_conv2_pad = self.pad(stage4_unit3_relu2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        stage4_unit3_conv2 = self.convolution(stage4_unit3_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit3_conv2')
        stage4_unit3_bn3 = self.batch_normalization(stage4_unit3_conv2, variance_epsilon=1.9999999494757503e-05,
                                               name='stage4_unit3_bn3')
        stage4_unit3_relu3 = self.relu(stage4_unit3_bn3, name='stage4_unit3_relu3')
        stage4_unit3_conv3 = self.convolution(stage4_unit3_relu3, self.weights_dict, strides=[1, 1], padding='VALID',
                                         name='stage4_unit3_conv3')
        plus15 = stage4_unit3_conv3 + plus14
        bn1 = self.batch_normalization(plus15, variance_epsilon=1.9999999494757503e-05, name='bn1')
        relu1 = self.relu(bn1, name='relu1')
        ssh_c3_lateral = self.convolution(relu1, self.weights_dict, strides=[1, 1], padding='VALID', name='ssh_c3_lateral')
        ssh_c3_lateral_bn = self.batch_normalization(ssh_c3_lateral, variance_epsilon=1.9999999494757503e-05,name='ssh_c3_lateral_bn')
        ssh_c3_lateral_relu = self.relu(ssh_c3_lateral_bn, name='ssh_c3_lateral_relu')
        ssh_m3_det_conv1_pad = self.pad(ssh_c3_lateral_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m3_det_conv1 = self.convolution(ssh_m3_det_conv1_pad, self.weights_dict, strides=[1, 1], padding='VALID',name='ssh_m3_det_conv1')
        ssh_m3_det_context_conv1_pad = self.pad(ssh_c3_lateral_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m3_det_context_conv1 = self.convolution(ssh_m3_det_context_conv1_pad, self.weights_dict, strides=[1, 1], padding='VALID',name='ssh_m3_det_context_conv1')
        ssh_c3_up = self.upsampling(ssh_c3_lateral_relu, (2, 2), "ssh_c3_up")
        ssh_m3_det_conv1_bn = self.batch_normalization(ssh_m3_det_conv1, variance_epsilon=1.9999999494757503e-05,name='ssh_m3_det_conv1_bn')
        ssh_m3_det_context_conv1_bn = self.batch_normalization(ssh_m3_det_context_conv1, variance_epsilon=1.9999999494757503e-05,name='ssh_m3_det_context_conv1_bn')
        crop0 = tf.identity(ssh_c3_up, "crop0")
        ssh_m3_det_context_conv1_relu = self.relu(ssh_m3_det_context_conv1_bn, name='ssh_m3_det_context_conv1_relu')
        plus0_v2 = ssh_c2_lateral_relu + crop0
        ssh_m3_det_context_conv2_pad = self.pad(ssh_m3_det_context_conv1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m3_det_context_conv2 = self.convolution(ssh_m3_det_context_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                               name='ssh_m3_det_context_conv2')
        ssh_m3_det_context_conv3_1_pad = self.pad(ssh_m3_det_context_conv1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m3_det_context_conv3_1 = self.convolution(ssh_m3_det_context_conv3_1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='ssh_m3_det_context_conv3_1')
        ssh_c2_aggr_pad = self.pad(plus0_v2, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_c2_aggr = self.convolution(ssh_c2_aggr_pad, self.weights_dict, strides=[1, 1], padding='VALID', name='ssh_c2_aggr')
        ssh_m3_det_context_conv2_bn = self.batch_normalization(ssh_m3_det_context_conv2, variance_epsilon=1.9999999494757503e-05,
                                                          name='ssh_m3_det_context_conv2_bn')
        ssh_m3_det_context_conv3_1_bn = self.batch_normalization(ssh_m3_det_context_conv3_1,
                                                            variance_epsilon=1.9999999494757503e-05,
                                                            name='ssh_m3_det_context_conv3_1_bn')
        ssh_c2_aggr_bn = self.batch_normalization(ssh_c2_aggr, variance_epsilon=1.9999999494757503e-05, name='ssh_c2_aggr_bn')
        ssh_m3_det_context_conv3_1_relu = self.relu(ssh_m3_det_context_conv3_1_bn, name='ssh_m3_det_context_conv3_1_relu')
        ssh_c2_aggr_relu = self.relu(ssh_c2_aggr_bn, name='ssh_c2_aggr_relu')
        ssh_m3_det_context_conv3_2_pad = self.pad(ssh_m3_det_context_conv3_1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m3_det_context_conv3_2 = self.convolution(ssh_m3_det_context_conv3_2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='ssh_m3_det_context_conv3_2')
        ssh_m2_det_conv1_pad = self.pad(ssh_c2_aggr_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m2_det_conv1 = self.convolution(ssh_m2_det_conv1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                       name='ssh_m2_det_conv1')
        ssh_m2_det_context_conv1_pad = self.pad(ssh_c2_aggr_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m2_det_context_conv1 = self.convolution(ssh_m2_det_context_conv1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                               name='ssh_m2_det_context_conv1')
        ssh_m2_red_up = self.upsampling(ssh_c2_aggr_relu, (2, 2), "ssh_m2_red_up")
        ssh_m3_det_context_conv3_2_bn = self.batch_normalization(ssh_m3_det_context_conv3_2,
                                                            variance_epsilon=1.9999999494757503e-05,
                                                            name='ssh_m3_det_context_conv3_2_bn')
        ssh_m2_det_conv1_bn = self.batch_normalization(ssh_m2_det_conv1, variance_epsilon=1.9999999494757503e-05,
                                                  name='ssh_m2_det_conv1_bn')
        ssh_m2_det_context_conv1_bn = self.batch_normalization(ssh_m2_det_context_conv1, variance_epsilon=1.9999999494757503e-05,
                                                          name='ssh_m2_det_context_conv1_bn')
        crop1 = tf.identity(ssh_m2_red_up, "crop1")
        ssh_m3_det_concat = tf.keras.layers.concatenate(
            [ssh_m3_det_conv1_bn, ssh_m3_det_context_conv2_bn, ssh_m3_det_context_conv3_2_bn], 3, name='ssh_m3_det_concat')
        ssh_m2_det_context_conv1_relu = self.relu(ssh_m2_det_context_conv1_bn, name='ssh_m2_det_context_conv1_relu')
        plus1_v1 = ssh_m1_red_conv_relu + crop1
        ssh_m3_det_concat_relu = self.relu(ssh_m3_det_concat, name='ssh_m3_det_concat_relu')
        ssh_m2_det_context_conv2_pad = self.pad(ssh_m2_det_context_conv1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m2_det_context_conv2 = self.convolution(ssh_m2_det_context_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                               name='ssh_m2_det_context_conv2')
        ssh_m2_det_context_conv3_1_pad = self.pad(ssh_m2_det_context_conv1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m2_det_context_conv3_1 = self.convolution(ssh_m2_det_context_conv3_1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='ssh_m2_det_context_conv3_1')
        ssh_c1_aggr_pad = self.pad(plus1_v1, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_c1_aggr = self.convolution(ssh_c1_aggr_pad, self.weights_dict, strides=[1, 1], padding='VALID', name='ssh_c1_aggr')
        face_rpn_cls_score_stride32 = self.convolution(ssh_m3_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                  name='face_rpn_cls_score_stride32')

        face_rpn_cls_score_reshape_stride32 = self.reshape(face_rpn_cls_score_stride32, (int(self.size/16),int(self.size/32),2),
                                                      "face_rpn_cls_score_reshape_stride32")

        face_rpn_bbox_pred_stride32 = self.convolution(ssh_m3_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                  name='face_rpn_bbox_pred_stride32')
        face_rpn_landmark_pred_stride32 = self.convolution(ssh_m3_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                      name='face_rpn_landmark_pred_stride32')
        ssh_m2_det_context_conv2_bn = self.batch_normalization(ssh_m2_det_context_conv2, variance_epsilon=1.9999999494757503e-05,
                                                          name='ssh_m2_det_context_conv2_bn')
        ssh_m2_det_context_conv3_1_bn = self.batch_normalization(ssh_m2_det_context_conv3_1,
                                                            variance_epsilon=1.9999999494757503e-05,
                                                            name='ssh_m2_det_context_conv3_1_bn')
        ssh_c1_aggr_bn = self.batch_normalization(ssh_c1_aggr, variance_epsilon=1.9999999494757503e-05, name='ssh_c1_aggr_bn')
        ssh_m2_det_context_conv3_1_relu = self.relu(ssh_m2_det_context_conv3_1_bn, name='ssh_m2_det_context_conv3_1_relu')
        ssh_c1_aggr_relu = self.relu(ssh_c1_aggr_bn, name='ssh_c1_aggr_relu')

        face_rpn_cls_prob_stride32 = tf.keras.layers.Softmax(name = 'face_rpn_cls_prob_stride32')(face_rpn_cls_score_reshape_stride32)




        face_rpn_cls_prob_reshape_stride32 = self.reshape(face_rpn_cls_prob_stride32, (int(self.size/32), int(self.size/32), 4),
                                                     "face_rpn_cls_prob_reshape_stride32")




        ssh_m2_det_context_conv3_2_pad = self.pad(ssh_m2_det_context_conv3_1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m2_det_context_conv3_2 = self.convolution(ssh_m2_det_context_conv3_2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='ssh_m2_det_context_conv3_2')
        ssh_m1_det_conv1_pad = self.pad(ssh_c1_aggr_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m1_det_conv1 = self.convolution(ssh_m1_det_conv1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                       name='ssh_m1_det_conv1')
        ssh_m1_det_context_conv1_pad = self.pad(ssh_c1_aggr_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m1_det_context_conv1 = self.convolution(ssh_m1_det_context_conv1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                               name='ssh_m1_det_context_conv1')
        ssh_m2_det_context_conv3_2_bn = self.batch_normalization(ssh_m2_det_context_conv3_2,
                                                            variance_epsilon=1.9999999494757503e-05,
                                                            name='ssh_m2_det_context_conv3_2_bn')
        ssh_m1_det_conv1_bn = self.batch_normalization(ssh_m1_det_conv1, variance_epsilon=1.9999999494757503e-05,
                                                  name='ssh_m1_det_conv1_bn')
        ssh_m1_det_context_conv1_bn = self.batch_normalization(ssh_m1_det_context_conv1, variance_epsilon=1.9999999494757503e-05,
                                                          name='ssh_m1_det_context_conv1_bn')
        ssh_m2_det_concat = tf.keras.layers.concatenate(
            [ssh_m2_det_conv1_bn, ssh_m2_det_context_conv2_bn, ssh_m2_det_context_conv3_2_bn], 3, name='ssh_m2_det_concat')
        ssh_m1_det_context_conv1_relu = self.relu(ssh_m1_det_context_conv1_bn, name='ssh_m1_det_context_conv1_relu')
        ssh_m2_det_concat_relu = self.relu(ssh_m2_det_concat, name='ssh_m2_det_concat_relu')
        ssh_m1_det_context_conv2_pad = self.pad(ssh_m1_det_context_conv1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m1_det_context_conv2 = self.convolution(ssh_m1_det_context_conv2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                               name='ssh_m1_det_context_conv2')
        ssh_m1_det_context_conv3_1_pad = self.pad(ssh_m1_det_context_conv1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m1_det_context_conv3_1 = self.convolution(ssh_m1_det_context_conv3_1_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='ssh_m1_det_context_conv3_1')
        face_rpn_cls_score_stride16 = self.convolution(ssh_m2_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                  name='face_rpn_cls_score_stride16')

        input_shape = [tf.shape(face_rpn_cls_score_stride16)[k] for k in range(4)]
        input_shape_1 = tf.dtypes.cast(input_shape[1] * 2, dtype=tf.int32)
        input_shape_2 = tf.dtypes.cast(input_shape[2], dtype=tf.int32)
        input_shape_3 = tf.dtypes.cast(input_shape[3] / 2, dtype=tf.int32)
        face_rpn_cls_score_reshape_stride16 = self.reshape(face_rpn_cls_score_stride16, (int(self.size/8), int(self.size/16), 2),
                                                      "face_rpn_cls_score_reshape_stride16")

        face_rpn_bbox_pred_stride16 = self.convolution(ssh_m2_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                  name='face_rpn_bbox_pred_stride16')
        face_rpn_landmark_pred_stride16 = self.convolution(ssh_m2_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                      name='face_rpn_landmark_pred_stride16')
        ssh_m1_det_context_conv2_bn = self.batch_normalization(ssh_m1_det_context_conv2, variance_epsilon=1.9999999494757503e-05,
                                                          name='ssh_m1_det_context_conv2_bn')
        ssh_m1_det_context_conv3_1_bn = self.batch_normalization(ssh_m1_det_context_conv3_1,
                                                            variance_epsilon=1.9999999494757503e-05,
                                                            name='ssh_m1_det_context_conv3_1_bn')
        ssh_m1_det_context_conv3_1_relu = self.relu(ssh_m1_det_context_conv3_1_bn, name='ssh_m1_det_context_conv3_1_relu')

        face_rpn_cls_prob_stride16 = tf.keras.layers.Softmax(name = 'face_rpn_cls_prob_stride16')(face_rpn_cls_score_reshape_stride16)



        face_rpn_cls_prob_reshape_stride16 = self.reshape(face_rpn_cls_prob_stride16, (int(self.size/16), int(self.size/16), 4),
                                                     "face_rpn_cls_prob_reshape_stride16")




        ssh_m1_det_context_conv3_2_pad = self.pad(ssh_m1_det_context_conv3_1_relu, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]])
        ssh_m1_det_context_conv3_2 = self.convolution(ssh_m1_det_context_conv3_2_pad, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='ssh_m1_det_context_conv3_2')
        ssh_m1_det_context_conv3_2_bn = self.batch_normalization(ssh_m1_det_context_conv3_2,
                                                            variance_epsilon=1.9999999494757503e-05,
                                                            name='ssh_m1_det_context_conv3_2_bn')
        ssh_m1_det_concat = tf.keras.layers.concatenate(
            [ssh_m1_det_conv1_bn, ssh_m1_det_context_conv2_bn, ssh_m1_det_context_conv3_2_bn], 3, name='ssh_m1_det_concat')
        ssh_m1_det_concat_relu = self.relu(ssh_m1_det_concat, name='ssh_m1_det_concat_relu')
        face_rpn_cls_score_stride8 = self.convolution(ssh_m1_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='face_rpn_cls_score_stride8')



        face_rpn_cls_score_reshape_stride8 = self.reshape(face_rpn_cls_score_stride8, (int(self.size/4), int(self.size/8), 2),
                                                     "face_rpn_cls_score_reshape_stride8")

        face_rpn_bbox_pred_stride8 = self.convolution(ssh_m1_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                 name='face_rpn_bbox_pred_stride8')
        face_rpn_landmark_pred_stride8 = self.convolution(ssh_m1_det_concat_relu, self.weights_dict, strides=[1, 1], padding='VALID',
                                                     name='face_rpn_landmark_pred_stride8')


        face_rpn_cls_prob_stride8 = tf.keras.layers.Softmax(name = 'face_rpn_cls_prob_stride8')(face_rpn_cls_score_reshape_stride8)



        face_rpn_cls_prob_reshape_stride8 = self.reshape(face_rpn_cls_prob_stride8, (int(self.size/8), int(self.size/8), 4),
                                                    "face_rpn_cls_prob_reshape_stride8")


        model = tf.keras.models.Model(inputs=data,
                                      outputs=[face_rpn_cls_prob_reshape_stride32,
                                               face_rpn_bbox_pred_stride32,
                                               face_rpn_landmark_pred_stride32,
                                               face_rpn_cls_prob_reshape_stride16,
                                               face_rpn_bbox_pred_stride16,
                                               face_rpn_landmark_pred_stride16,
                                               face_rpn_cls_prob_reshape_stride8,
                                               face_rpn_bbox_pred_stride8,
                                               face_rpn_landmark_pred_stride8
                                               ])
        # model = tf.keras.models.Model(inputs = data, outputs = face_rpn_cls_score_reshape_stride32)
        weights_names = list(self.weights_dict.keys())
        for layer in model.layers:
            if layer.name in weights_names:
                if "bn" in layer.name:
                    mean = self.weights_dict[layer.name]["mean"]
                    var = self.weights_dict[layer.name]["var"]
                    scale = self.weights_dict[layer.name]["scale"] if "scale" in self.weights_dict[layer.name] else np.ones(
                        self.weights_dict[layer.name]["mean"].shape)
                    bias = self.weights_dict[layer.name]["bias"] if "bias" in self.weights_dict[layer.name] else np.zeros(
                        self.weights_dict[layer.name]["mean"].shape)
                    layer.set_weights([scale, bias, mean, var])
                else:
                    if "bias" in self.weights_dict[layer.name].keys():
                        layer.set_weights([self.weights_dict[layer.name]["weights"],self.weights_dict[layer.name]["bias"]])
                    else:
                        layer.set_weights([self.weights_dict[layer.name]["weights"]])
        return model

    @staticmethod
    def relu(input_tensor, name):
        return tf.keras.layers.ReLU(name=name)(input_tensor)

    @staticmethod
    def pad(input_tensor, paddings):
        return tf.keras.layers.ZeroPadding2D(padding=tuple(paddings[1]))(input_tensor)

    @staticmethod
    def convolution(input_tensor, weights_dict, strides, padding, name):
        weights = weights_dict[name]['weights']

        layer = tf.keras.layers.Conv2D(filters=weights.shape[3],
                                       kernel_size=weights.shape[0:2],
                                       name=name,
                                       strides=strides,
                                       padding=padding,
                                       use_bias=True if "bias" in weights_dict[name].keys() else False)(input_tensor)
        return layer

    @staticmethod
    def batch_normalization(input_tensor, variance_epsilon, name):
        return tf.keras.layers.BatchNormalization(epsilon=variance_epsilon, name=name, trainable=False)(input_tensor)

    # @staticmethod
    # def crop(input_tensor, name):
    #     return tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)), name=name)(input_tensor)
    @staticmethod
    def crop(input_tensor, obj_shape_tensor, name):
        x1_shape = tf.shape(input_tensor)
        x2_shape = tf.shape(obj_shape_tensor)
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        cropped = tf.slice(input_tensor, offsets, size, name)
        return cropped


    @staticmethod
    def reshape(input_tensor, shape, name):
        return tf.keras.layers.Reshape(target_shape=shape, name=name)(input_tensor)

    @staticmethod
    def upsampling(input_tensor, size, name):
        return tf.keras.layers.UpSampling2D(size=size, interpolation="nearest", name=name)(input_tensor)
