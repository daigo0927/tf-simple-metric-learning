import numpy as np
import torch
from pytorch_metric_learning.losses import ArcFaceLoss

np.random.seed(1)
BATCH_SIZE = 1
NUM_CLASSES = 10
EMBEDDING_SIZE= 32


def test():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    # inputs = np.ones((BATCH_SIZE, EMBEDDING_SIZE), dtype=np.float32)
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    w = np.random.normal(0, 1, size=(EMBEDDING_SIZE, NUM_CLASSES))
    # w = np.zeros((EMBEDDING_SIZE, NUM_CLASSES), dtype=np.float32)
    # w[:, 0] = 1
    # w[:, 2] = -1
    print('inputs:', inputs)
    print('labels:', labels)
    print('w:', w)
    
    layer = ArcFaceLoss(num_classes=NUM_CLASSES,
                        embedding_size=EMBEDDING_SIZE)
    layer.W = torch.nn.Parameter(torch.from_numpy(w))
    print('W', layer.W)
    print('margin:', layer.margin)
    print('scale:', layer.scale)

    embeds = torch.from_numpy(inputs)
    lbls = torch.from_numpy(labels)
    loss = layer(torch.from_numpy(inputs), torch.from_numpy(labels))

    # mask = layer.get_target_mask(embeds, lbls)
    # cosine = layer.get_cosine(embeds)
    # cosine_of_target_classes = cosine[mask == 1]
    # modified_cosine_of_target_classes = layer.modify_cosine_of_target_classes(cosine_of_target_classes, cosine, embeds, lbls, mask)
    # diff = (modified_cosine_of_target_classes - cosine_of_target_classes).unsqueeze(1)
    # logits = cosine + (mask*diff)
    
    # logits = layer.scale_logits(logits, embeds)
    # loss = layer.cross_entropy(logits, lbls)

    # print('cos:', cosine)
    # print('logits:', logits)
    print(loss)

if __name__ == "__main__":
    test()
