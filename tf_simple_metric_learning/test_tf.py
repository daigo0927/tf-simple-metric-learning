import numpy as np
import tensorflow as tf
from layers import ArcFace

np.random.seed(1)
BATCH_SIZE = 1
NUM_CLASSES = 10
EMBEDDING_SIZE= 32


def test():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    w = np.random.normal(0, 1, size=(EMBEDDING_SIZE, NUM_CLASSES))
    print('inputs:', inputs)
    print('labels:', labels)
    print('w:', w)

    layer = ArcFace(margin=np.radians(28.6))
    print('margin:', layer.margin)
    print('scale:', layer.scale)    
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)
    print('one_hot:',labels_onehot)
    info = layer([inputs_tf, labels_onehot], training=True)

    layer.W.assign(tf.convert_to_tensor(w, dtype=tf.float32))
    info = layer([inputs_tf, labels_onehot], training=True)    

    print('cos:', info['cos'])
    print('logits:', info['logits'])
    loss = tf.keras.losses.categorical_crossentropy(labels_onehot, info['logits'],
                                                    from_logits=True)
    print(tf.reduce_mean(loss))

if __name__ == "__main__":
    test()
