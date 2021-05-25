import tensorflow as tf
import numpy as np


def load_weights(weight_file):
    """
    Load weights from npy file to map

    :param weight_file: path to npy file
    :return: map layer_name -> map of weights
    """
    import numpy as np
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

def load_weights_to_network(model, weights_dict):
    """
    load pretrained weights to tf keras model

    :param model:
    :param weights_dict:
    :return: tf.keras.model.Model, with loaded weights
    """
    weights_names = list(weights_dict.keys())
    for layer in model.layers:
        if layer.name in weights_names:
            if "bn" in layer.name:
                mean = weights_dict[layer.name]["mean"]
                var = weights_dict[layer.name]["var"]
                scale = weights_dict[layer.name]["scale"] if "scale" in weights_dict[layer.name] else np.ones(
                    weights_dict[layer.name]["mean"].shape)
                bias = weights_dict[layer.name]["bias"] if "bias" in weights_dict[layer.name] else np.zeros(
                    weights_dict[layer.name]["mean"].shape)
                layer.set_weights([scale, bias, mean, var])
            else:
                if "bias" in weights_dict[layer.name].keys():
                    layer.set_weights([weights_dict[layer.name]["weights"], weights_dict[layer.name]["bias"]])
                else:
                    layer.set_weights([weights_dict[layer.name]["weights"]])
    return model

def relu(input_tensor, name):
    """
    Applies tf keras ReLU

    :param input_tensor:
    :param name:
    :return: ReLu tensor applied to input
    """
    return tf.keras.layers.ReLU(name=name)(input_tensor)


def pad(input_tensor, paddings):
    """
    Applies tf keras zero padding, with paddding defined as image dimensions (2nd and 3rd)
    of paddings vector

    :param input_tensor:
    :param paddings: length 4 list of paddings for each dim
    :return: zero padding tensor of input
    """
    assert len(paddings) == 4
    assert paddings[2] == paddings[2]
    return tf.keras.layers.ZeroPadding2D(padding=tuple(paddings[1]))(input_tensor)


def convolution(input_tensor, weights_dict, strides, padding, name):
    """
    Applies tf keras conv2D, with conv parameters defined by weights shape

    :param input_tensor:
    :param weights_dict: weights of all model
    :param strides: conv strides, lenght 2 list
    :param padding: conv padding, 'valid' or 'same'
    :param name:
    :return: convolution tensor of input
    """
    weights = weights_dict[name]['weights']

    layer = tf.keras.layers.Conv2D(filters=weights.shape[3],
                                   kernel_size=weights.shape[0:2],
                                   name=name,
                                   strides=strides,
                                   padding=padding,
                                   use_bias=True if "bias" in weights_dict[name].keys() else False)(input_tensor)
    return layer


def batch_normalization(input_tensor, variance_epsilon, name):
    """
    Appies tf keras BN layer

    :param input_tensor:
    :param variance_epsilon: epsilon to add to variance vector to avoid division by 0
    :param name:
    :return: BN tensor
    """
    return tf.keras.layers.BatchNormalization(epsilon=variance_epsilon, name=name, trainable=False)(input_tensor)


def crop(input_tensor, obj_shape_tensor, name):
    """
    crops tensor input_tensor into the shape of tensor obj_shape_tensor

    :param input_tensor:
    :param obj_shape_tensor: tensor, of shape the shape we want to crop to
    :param name:
    :return: cropped tensor
    """
    x1_shape = tf.shape(input_tensor)
    x2_shape = tf.shape(obj_shape_tensor)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    cropped = tf.slice(input_tensor, offsets, size, name)
    return cropped


def reshape(input_tensor, mode, name):
    """
    Reshapes input tensor.
    if mode is 0, reshape based on input tensor shape by dividing channels dim by 2
    if mode is 1, reshape based on input tensor shape by multiplying channels dim by 2

    :param input_tensor:
    :param mode: reshaping mode, 0 or 1
    :param name:
    :return: reshaped tensor
    """
    input_shape = [tf.shape(input_tensor)[k] for k in range(4)]
    if mode==0:
        input_shape_1 = tf.dtypes.cast(input_shape[1] * 2, dtype=tf.int32)
        input_shape_2 = tf.dtypes.cast(input_shape[2], dtype=tf.int32)
        input_shape_3 = tf.dtypes.cast(input_shape[3] / 2, dtype=tf.int32)
    elif mode==1:
        input_shape_1 = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
        input_shape_2 = tf.dtypes.cast(input_shape[2], dtype=tf.int32)
        input_shape_3 = tf.dtypes.cast(input_shape[3] * 2, dtype=tf.int32)
    return tf.keras.layers.Reshape(target_shape=[input_shape_1, input_shape_2, input_shape_3], name=name)(input_tensor)


def reshape_mxnet_1(input_tensor, name):
    inter_1 = tf.keras.layers.concatenate([input_tensor[:, :, :, 0], input_tensor[:, :, :, 1]], axis=1)
    inter_2 = tf.keras.layers.concatenate([input_tensor[:, :, :, 2], input_tensor[:, :, :, 3]], axis=1)
    final = tf.stack([inter_1, inter_2])
    return tf.transpose(final, (1, 2, 3, 0), name=name)


def reshape_mxnet_2(input_tensor, name):
    input_shape = [tf.shape(input_tensor)[k] for k in range(4)]
    sz = tf.dtypes.cast(input_shape[1] / 2, dtype=tf.int32)
    inter_1 = input_tensor[:, 0:sz, :, 0]
    inter_2 = input_tensor[:, 0:sz, :, 1]
    inter_3 = input_tensor[:, sz:, :, 0]
    inter_4 = input_tensor[:, sz:, :, 1]
    final = tf.stack([inter_1, inter_3, inter_2, inter_4])
    return tf.transpose(final, (1, 2, 3, 0), name=name)


def upsampling(input_tensor, size, name):
    """
    Upsample input tensor with nearest neighbor method by scales size

    :param input_tensor:
    :param size: scales of upsampling, lenght 2 tuple
    :param name:
    :return: upsampled tensor
    """
    return tf.keras.layers.UpSampling2D(size=size, interpolation="nearest", name=name)(input_tensor)
