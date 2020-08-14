import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses
import tensorflow_probability as tfp


class CosineSimilarity(layers.Layer):
    """
    Cosine similarity with classwise weights
    """
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(shape=(input_dim, self.num_classes),
                                 initializer='random_normal')

    def call(self, inputs):
        x = tf.nn.l2_normalize(inputs, axis=-1) # (batch_size, ndim)
        w = tf.nn.l2_normalize(self.W, axis=0)   # (ndim, nclass)
        cos = tf.matmul(x, w) # (batch_size, nclass)
        return cos


def arcface_loss(y_true, y_pred, margin=0.5, scale=64): 
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """   
    theta = tf.acos(tf.clip_by_value(y_pred, -1, 1))
    cos_add = tf.cos(theta + margin)
    
    mask = tf.cast(y_true, dtype=cos_add.dtype)
    logits = mask*cos_add + (1-mask)*y_pred
    logits *= scale
    loss = losses.categorical_crossentropy(y_true, logits, from_logits=True)
    return loss


class ArcFaceLoss(losses.Loss):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """    
    def __init__(self,
                 margin=0.5,
                 scale=64,
                 reduction='auto',
                 name='arcface_loss'):
        super().__init__(reduction=reduction, name=name)
        self.margin = margin
        self.scale = scale
        
    def call(self, y_true, y_pred):
        return arcface_loss(y_true, y_pred, self.margin, self.scale)
