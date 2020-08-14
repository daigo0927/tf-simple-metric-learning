import numpy as np
import tensorflow as tf
from layers import CosineSimilarity, ArcFaceLoss

np.random.seed(1)
BATCH_SIZE = 2
NUM_CLASSES = 10
EMBEDDING_SIZE= 32


def test():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    w = np.random.normal(0, 1, size=(EMBEDDING_SIZE, NUM_CLASSES))
    print('inputs:', inputs)
    print('labels:', labels)
    print('w:', w)

    cos_layer = CosineSimilarity(NUM_CLASSES)
    arcface = ArcFaceLoss(margin=np.radians(28.6), reduction='none')
    print('margin:', arcface.margin)
    print('scale:', arcface.scale)
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)
    print('one_hot:',labels_onehot)
    cos = cos_layer(inputs_tf)
    _ = arcface(labels_onehot, cos)

    cos_layer.W.assign(tf.convert_to_tensor(w, dtype=tf.float32))
    cos = cos_layer(inputs_tf)
    loss = arcface(labels_onehot, cos)
    print(loss)

    
if __name__ == "__main__":
    test()
