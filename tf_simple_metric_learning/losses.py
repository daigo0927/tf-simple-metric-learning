import tensorflow as tf


def arcface_loss(y_true, y_pred, margin=0.5, scale=64):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf

    Args:
      y_true: one-hot tensor of target labels
      y_pred: predicted tensor, assumed to be cosine values
      margin: a float value, margin for the true label (default 0.5)
      scale: a float value, final scale value (default 64)

    Returns:
      categorical crossentropy for each samples
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
