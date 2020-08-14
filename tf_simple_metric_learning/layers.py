import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
import tensorflow_probability as tfp


class ArcFace(layers.Layer):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """
    def __init__(self, margin=0.5, scale=64, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.scale = scale

    def build(self, input_shape):
        feature_shape, label_shape = input_shape
        self.W = self.add_weight(shape=(feature_shape[-1], label_shape[-1]),
                                 initializer='random_normal')
        # w = np.zeros((feature_shape[-1], label_shape[-1]), dtype=np.float32)
        # w[:, 0] = 1
        # w[:, 2] = -1
        # self.W = tf.Variable(w, trainable=True)        

    def call(self, inputs, training):
        feature, label = inputs

        x = tf.nn.l2_normalize(feature, axis=-1) # (batch_size, nch)
        w = tf.nn.l2_normalize(self.W, axis=0)   # (nch, nlabel)

        # Get cosine correlation: cos(theta)
        cos = tf.matmul(x, w) # (batch_size, nlabel)

        if training:
            # theta = tf.acos(tf.clip_by_value(cos, -1, 1))
            # cos_add = tf.cos(theta+self.margin)
        
            # mask = tf.cast(label, dtype=cos_add.dtype)
            # logits = mask*cos_add + (1-mask)*cos
            # logits *= self.scale

            mask = tf.cast(label, dtype=cos.dtype)

            cos_tar = cos[label == 1]
            theta = tf.acos(tf.clip_by_value(cos_tar, -1, 1))
            cos_tar_add = tf.cos(theta+self.margin)
            diff = tf.expand_dims(cos_tar_add - cos_tar, axis=-1)
            logits = cos + (mask*diff)
            logits = self.scale*logits
        else:
            logits = self.scale*cos

        return {'cos': cos,
                'mask': mask,
                'cos_tar': cos_tar,
                'cos_tar_add': cos_tar_add,
                'diff': diff,
                'logits': logits}
