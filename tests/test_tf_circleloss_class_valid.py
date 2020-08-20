import numpy as np
import tensorflow as tf
from layers import CircleLossCL

np.random.seed(1)
BATCH_SIZE = 12
NUM_CLASSES = 10
EMBEDDING_SIZE = 32

MARGIN = 0.25
SCALE = 256

import copy


def test():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    w = np.random.normal(0, 1, size=(EMBEDDING_SIZE, NUM_CLASSES))
    print('inputs:', inputs)
    print('labels:', labels)
    print('w:', w)

    circle = CircleLossCL(num_classes=NUM_CLASSES, margin=MARGIN, scale=SCALE)
    print('margin:', circle.margin)
    print('scale:', circle.scale)
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)
    print('one_hot:',labels_onehot)
    _ = circle([inputs_tf, labels_onehot], training=True)

    circle.cos_similarity.W.assign(tf.convert_to_tensor(w, dtype=tf.float32))
    logits = circle([inputs_tf, labels_onehot], training=True)
    print('logits:', logits)
    losses = tf.keras.losses.categorical_crossentropy(labels_onehot, logits,
                                                      from_logits=True)
    print('losses:', losses)

    
if __name__ == "__main__":
    test()
