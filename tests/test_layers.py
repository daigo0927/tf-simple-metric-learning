import pytest
import numpy as np
import scipy as sp
import tensorflow as tf

from tf_simple_metric_learning.layers import (
    CosineSimilarity,
    ArcFace,
    AdaCos,
    CircleLoss,
    CircleLossCL
)

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)
BATCH_SIZE = 32
NUM_CLASSES = 10
EMBEDDING_SIZE = 64


def test_cosine_similarity():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    
    cos_similarity = CosineSimilarity(num_classes=NUM_CLASSES)

    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    cos = cos_similarity(inputs_tf)

    inputs_normed = inputs / np.linalg.norm(inputs, ord=2, axis=-1, keepdims=True)
    w = cos_similarity.W.numpy()    
    w_normed = w / np.linalg.norm(w, ord=2, axis=0, keepdims=True)
    cos_valid = np.matmul(inputs_normed, w_normed)

    np.testing.assert_allclose(cos.numpy(), cos_valid, rtol=1e-4)
    

@pytest.mark.parametrize('margin, scale', [(0.5, 64), (1.0, 64), (0.5, 128)])
def test_arcface(margin, scale):
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)

    # Module output
    arcface = ArcFace(num_classes=NUM_CLASSES, margin=margin, scale=scale)
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)

    logits = arcface([inputs_tf, labels_onehot], training=True)

    # Valid output (numpy implementation)
    inputs_normed = inputs / np.linalg.norm(inputs, ord=2, axis=1, keepdims=True)
    w = arcface.cos_similarity.W.numpy()
    w_normed = w / np.linalg.norm(w, ord=2, axis=0, keepdims=True)
    cos = np.matmul(inputs_normed, w_normed)

    acos = np.arccos(np.clip(cos, -1, 1))
    for i, c in enumerate(labels):
        cos[i, c] = np.math.cos(acos[i, c] + margin)
    logits_valid = scale*cos
    
    np.testing.assert_allclose(logits.numpy(), logits_valid, rtol=1e-4)


def test_adacos():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)

    # Module output
    adacos = AdaCos(num_classes=NUM_CLASSES)
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)

    logits = adacos([inputs_tf, labels_onehot], training=True)

    # Valid output (numpy implementation)
    inputs_normed = inputs / np.linalg.norm(inputs, ord=2, axis=1, keepdims=True)
    w = adacos.cos_similarity.W.numpy()
    w_normed = w / np.linalg.norm(w, ord=2, axis=0, keepdims=True)
    cos = np.matmul(inputs_normed, w_normed)

    scale = np.sqrt(2)*np.log(NUM_CLASSES - 1)
    mask = labels_onehot.numpy()
    B = (1 - mask)*np.exp(scale*cos)
    B_avg = np.mean(np.sum(B, axis=-1), axis=0)

    acos = np.arccos(np.clip(cos, -1, 1))
    acos_p = np.sum(mask*acos, axis=-1)
    acos_med = np.median(acos_p)

    scale = np.log(B_avg) / np.cos(np.minimum(np.pi/4, acos_med))
    logits_valid = scale*cos
    
    np.testing.assert_allclose(logits.numpy(), logits_valid, rtol=1e-4)
    np.testing.assert_allclose(adacos.scale.numpy(), scale, rtol=1e-4)
    

@pytest.mark.parametrize('margin, scale', [(0.25, 256), (0.4, 64)])
def test_circleloss(margin, scale):
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)

    # Module output
    circle = CircleLoss(margin=margin, scale=scale)
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)

    losses = circle([inputs_tf, labels_onehot], training=True)

    # Valid output (numpy implementation)
    inputs_normed = inputs / np.linalg.norm(inputs, ord=2, axis=1, keepdims=True)
    cos = np.matmul(inputs_normed, inputs_normed.T)

    Op, On, Dp, Dn = circle._Op, circle._On, circle._Dp, circle._Dn

    mask = labels_onehot.numpy()
    mask_p = np.matmul(mask, mask.T)
    mask_n = 1 - mask_p
    mask_p = mask_p - np.eye(BATCH_SIZE) # ignore indentity element

    logits_p = - scale * np.maximum(Op - cos, 0) * (cos - Dp)
    logits_n = scale * np.maximum(cos - On, 0) * (cos - Dn)

    logits_p[mask_p < 1] = -np.inf
    logits_n[mask_n < 1] = -np.inf

    logsumexp_p = sp.special.logsumexp(logits_p, axis=-1)
    logsumexp_n = sp.special.logsumexp(logits_n, axis=-1)

    mask_pr = np.max(mask_p, axis=-1)
    mask_nr = np.max(mask_n, axis=-1)
    logsumexp_p[mask_pr < 1] = 0
    logsumexp_n[mask_nr < 1] = 0

    losses_valid = np.log(np.exp(logsumexp_p + logsumexp_n) + 1)
    mask_paired = mask_pr*mask_nr
    print(mask_paired)
    losses_valid *= mask_paired

    np.testing.assert_allclose(losses.numpy(), losses_valid, rtol=1e-4)


@pytest.mark.parametrize('margin, scale', [(0.25, 256), (0.4, 64)])
def test_circleloss_cl(margin, scale):
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)

    # Module output
    circle = CircleLossCL(num_classes=NUM_CLASSES, margin=margin, scale=scale)
    
    inputs_tf = tf.convert_to_tensor(inputs, dtype=tf.float32)
    labels_tf = tf.convert_to_tensor(labels, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels_tf, depth=NUM_CLASSES)

    logits = circle([inputs_tf, labels_onehot], training=True)

    # Valid output (numpy implementation)
    inputs_normed = inputs / np.linalg.norm(inputs, ord=2, axis=1, keepdims=True)
    w = circle.cos_similarity.W.numpy()
    w_normed = w / np.linalg.norm(w, ord=2, axis=0, keepdims=True)
    cos = np.matmul(inputs_normed, w_normed)    

    Op, On, Dp, Dn = circle._Op, circle._On, circle._Dp, circle._Dn

    mask = labels_onehot.numpy()
    logits_p = scale * np.maximum(Op - cos, 0) * (cos - Dp)
    logits_n = scale * np.maximum(cos - On, 0) * (cos - Dn)
    logits_valid = mask * logits_p + (1 - mask) * logits_n

    np.testing.assert_allclose(logits.numpy(), logits_valid, rtol=1e-4)
