import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        x = tf.nn.l2_normalize(inputs, axis=-1) # (batch_size, ndim)
        w = tf.nn.l2_normalize(self.W, axis=0)   # (ndim, nclass)
        cos = tf.matmul(x, w) # (batch_size, nclass)
        return cos


class ArcFace(layers.Layer):
    """
    Implementation of https://arxiv.org/pdf/1801.07698.pdf
    """       
    def __init__(self, num_classes, margin=0.5, scale=64, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.cos_similarity = CosineSimilarity(num_classes)

    def call(self, inputs, training):
        # If not training (prediction), labels are ignored
        feature, labels = inputs
        cos = self.cos_similarity(feature)

        if training:
            theta = tf.acos(tf.clip_by_value(cos, -1, 1))
            cos_add = tf.cos(theta + self.margin)
    
            mask = tf.cast(labels, dtype=cos_add.dtype)
            logits = mask*cos_add + (1-mask)*cos
            logits *= self.scale
            return logits
        else:
            return cos


class AdaCos(layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

        self.cos_similarity = CosineSimilarity(num_classes)
        self.scale = tf.Variable(tf.sqrt(2.0)*tf.math.log(num_classes - 1.0),
                                 trainable=False)

    def call(self, inputs, training):
        # In inference, labels are ignored
        feature, labels = inputs
        cos = self.cos_similarity(feature)

        if training:
            mask = tf.cast(labels, dtype=cos.dtype)
        
            # Collect cosine values at only false labels
            B = (1 - mask)*tf.exp(self.scale*cos)
            B_avg = tf.reduce_mean(tf.reduce_sum(B, axis=-1), axis=0)

            theta = tf.acos(tf.clip_by_value(cos, -1, 1))
            # Collect cosine at true labels
            theta_true = tf.reduce_sum(mask*theta, axis=-1)
            # get median (=50-percentile)
            theta_med = tfp.stats.percentile(theta_true, q=50)
 
            scale = tf.math.log(B_avg) / tf.cos(tf.minimum(np.pi/4, theta_med))
            scale = tf.stop_gradient(scale)
            logits = scale*cos
            
            self.scale.assign(scale)
            return logits
        else:
            return cos


class CircleLoss(layers.Layer):
    """
    Implementation of https://arxiv.org/abs/2002.10857 (pair-level label version)
    """
    def __init__(self, margin=0.25, scale=256, **kwargs):
        """
        Args
          margin: a float value, margin for the true label (default 0.25)
          scale: a float value, final scale value,
            stated as gamma in the original paper (default 256)

        Returns:
          a tf.keras.layers.Layer object, outputs logit values of each class

        In the original paper, margin and scale (=gamma) are set depends on tasks
        - Face recognition: m=0.25, scale=256 (default)
        - Person re-identification: m=0.25, scale=256
        - Fine-grained image retrieval: m=0.4, scale=64
        """
        super().__init__(**kwargs)
        self.margin = margin
        self.scale = scale

        self._Op = 1 + margin # O_positive
        self._On = -margin    # O_negative
        self._Dp = 1 - margin # Delta_positive
        self._Dn = margin     # Delta_negative

    def call(self, inputs, training):
        feature, labels = inputs
        x = tf.nn.l2_normalize(feature, axis=-1)
        cos = tf.matmul(x, x, transpose_b=True) # (batch_size, batch_size)

        if training:
            # pairwise version
            mask = tf.cast(labels, dtype=cos.dtype)
            mask_p = tf.matmul(mask, mask, transpose_b=True)
            mask_n = 1 - mask_p
            mask_p = mask_p - tf.eye(mask_p.shape[0])

            logits_p = - self.scale * tf.nn.relu(self._Op - cos) * (cos - self._Dp)
            logits_n = self.scale * tf.nn.relu(cos - self._On) * (cos - self._Dn)

            logits_p = tf.where(mask_p == 1, logits_p, -np.inf)
            logits_n = tf.where(mask_n == 1, logits_n, -np.inf)

            logsumexp_p = tf.reduce_logsumexp(logits_p, axis=-1)
            logsumexp_n = tf.reduce_logsumexp(logits_n, axis=-1)

            mask_p_row = tf.reduce_max(mask_p, axis=-1)
            mask_n_row = tf.reduce_max(mask_n, axis=-1)
            logsumexp_p = tf.where(mask_p_row == 1, logsumexp_p, 0)
            logsumexp_n = tf.where(mask_n_row == 1, logsumexp_n, 0)

            losses = tf.nn.softplus(logsumexp_p + logsumexp_n)

            mask_paired = mask_p_row*mask_n_row
            losses = mask_paired * losses
            return losses
        else:
            return cos
        

class CircleLossCL(layers.Layer):
    """
    Implementation of https://arxiv.org/abs/2002.10857 (class-level label version)
    """    
    def __init__(self, num_classes, margin=0.25, scale=256, **kwargs):
        """
        Args
          num_classes: an int value, number of target classes
          margin: a float value, margin for the true label (default 0.25)
          scale: a float value, final scale value,
            stated as gamma in the original paper (default 256)

        Returns:
          a tf.keras.layers.Layer object, outputs logit values of each class

        In the original paper, margin and scale (=gamma) are set depends on tasks
        - Face recognition: m=0.25, scale=256 (default)
        - Person re-identification: m=0.25, scale=256
        - Fine-grained image retrieval: m=0.4, scale=64
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self._Op = 1 + margin # O_positive
        self._On = -margin    # O_negative
        self._Dp = 1 - margin # Delta_positive
        self._Dn = margin     # Delta_negative

        self.cos_similarity = CosineSimilarity(num_classes)

    def call(self, inputs, training):
        feature, labels = inputs
        cos = self.cos_similarity(feature)
        
        if training:
            # class-lebel version
            mask = tf.cast(labels, dtype=cos.dtype)

            alpha_p = tf.nn.relu(self._Op - cos)
            alpha_n = tf.nn.relu(cos - self._On)

            logits_p = self.scale*alpha_p*(cos - self._Dp)
            logits_n = self.scale*alpha_n*(cos - self._Dn)

            logits = mask*logits_p + (1-mask)*logits_n
            return logits
        else:
            return cos
