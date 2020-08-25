# Simple metric learning via tf.keras

This package provides only a few metric learning losses below;
- ArcFace
- AdaCos
- CircleLoss

I have been greatly inspired by [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning).

## Installation

``` shell
$ pip install tf-simple-metric-learning
```

## Usage

Provided layers are implemented via `tf.keras.layers.Layer` API, enables;

``` python
from tf_simple_metric_learning.layers import ArcFace

arcface = ArcFace(num_classes=NUM_CLASSES, margin=MARGIN, scale=SCALE)
```

Example notebook is in [examples](https://github.com/daigo0927/tf-simple-metric-learning/tree/develop/examples) directory. Implement CircleLossCL (Class-level label version) layer top of EfficientNet and train it for [Cars196 dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html);

``` python
import tensorflow as tf
from tf_simple_metric_learning.layers import ArcFace, AdaCos, CircleLossCL

inputs = tf.keras.layers.Input([*IMAGE_SIZE, 3], dtype=tf.uint8)
x = tf.cast(inputs, dtype=tf.float32)
x = tf.keras.applications.efficientnet.preprocess_input(x)

net = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', pooling='avg')
embeds = net(x)

labels = tf.keras.layers.Input([], dtype=tf.int32)
labels_onehot = tf.one_hot(labels, depth=num_classes)

# Create metric learning layer
# metric_layer = ArcFace(num_classes=num_classes, margin=0.5, scale=64)
# metric_layer = AdaCos(num_classes=num_classes)
metric_layer = CircleLossCL(num_classes=num_classes, margin=0.25, scale=256)

logits = metric_layer([embeds, labels_onehot])

model = tf.keras.Model(inputs=[inputs, labels], outputs=logits)
model.summary()
```

**Note that you should feed labels as input** into model in training because these layers require labels to forward.

In evaluation or prediction, above model requires both images and labels but labels is ignored in those metric learning layers. We only need to use dummy labels (ignored) with the target images because we can't access labels in evaluation or prediction.

## References
- https://github.com/KevinMusgrave/pytorch-metric-learning
- https://github.com/scikit-learn-contrib/metric-learn
