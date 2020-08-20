import numpy as np
import torch
from pytorch_metric_learning.losses import CircleLoss
from pytorch_metric_learning.reducers import DoNothingReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

np.random.seed(1)
BATCH_SIZE = 12
NUM_CLASSES = 10
EMBEDDING_SIZE= 32

MARGIN = 0.25
SCALE = 256


def test():
    inputs = np.random.normal(0, 1, size=(BATCH_SIZE, EMBEDDING_SIZE))
    labels = np.random.randint(0, NUM_CLASSES, size=BATCH_SIZE)
    w = np.random.normal(0, 1, size=(EMBEDDING_SIZE, NUM_CLASSES))
    print('inputs:', inputs)
    print('labels:', labels)
    print('w:', w)
    
    layer = CircleLoss(m=MARGIN,
                       gamma=SCALE,
                       reducer=DoNothingReducer())
    # print('W', layer.W)
    print('margin:', layer.m)
    print('scale:', layer.gamma)

    inp = torch.from_numpy(inputs)
    lbls = torch.from_numpy(labels)

    # inp_norm = torch.nn.functional.normalize(inp, p=2, dim=1)
    # w_norm = torch.nn.functional.normalize(torch.from_numpy(w), p=2, dim=0)
    # embeds = layer.distance(inp)

    # losses.GenericPairLoss.compute_loss
    indices_tuple = lmu.convert_to_pairs(None, lbls)
    mat = layer.distance(inp) # (batch_size, batch_size) of cosine similarity
    # losses.GenericPairLoss.mat_based_loss
    a1, p, a2, n = indices_tuple
    pos_mask, neg_mask = torch.zeros_like(mat), torch.zeros_like(mat)
    pos_mask[a1, p] = 1 # 1 at positive pair positions
    neg_mask[a2, n] = 1 # 1 at negative pair positions
    print('mask_p:', pos_mask)
    print('mask_n:', neg_mask)
    
    # losses.CircleLoss._compute_loss
    pos_mask_bool = pos_mask.bool()
    neg_mask_bool = neg_mask.bool()
    anchor_positive = mat[pos_mask_bool]
    anchor_negative = mat[neg_mask_bool]
    new_mat = torch.zeros_like(mat)
    print('cos:', mat)

    # -logit of positive pairs at (batch_size, batch_size) matrix
    new_mat[pos_mask_bool] = -layer.gamma * torch.relu(layer.op - anchor_positive.detach()) * (anchor_positive - layer.delta_p)
    # logits of negative pairs at (batch_size, batch_size) matrix
    new_mat[neg_mask_bool] = layer.gamma * torch.relu(anchor_negative.detach() - layer.on) * (anchor_negative - layer.delta_n)

    print('logits_p:', pos_mask*new_mat)
    print('logits_n:', neg_mask*new_mat)

    # logsumexp over axis-1 -> shape (batch_size, 1)
    # 1. put -inf at 0 position by pos/neg_mask_bool
    # 2. torch.logsumexp along axis-1 -> shape (batch_size, 1), -inf shurinks to 0
    # 3. put 0 to the row with no pairs
    #    by output.masked_fill(~torch.any(keep_mask, dim=dim, keepdim=True), 0)
    logsumexp_pos = lmu.logsumexp(new_mat, keep_mask=pos_mask_bool, add_one=False, dim=1)
    logsumexp_neg = lmu.logsumexp(new_mat, keep_mask=neg_mask_bool, add_one=False, dim=1)

    print('logsumexp_pos:', logsumexp_pos)
    print('logsumexp_neg:', logsumexp_neg)
    
    # softplus: log(1 + exp(x)) = log(1+sum(exp(pos))*sum(exp(neg)))
    #           = log(1 + sum(exp(-logit_p)) * sum(exp(logit_n)))
    losses = layer.soft_plus(logsumexp_pos + logsumexp_neg)
    print('losses:', losses)

    zero_rows = torch.where((torch.sum(pos_mask, dim=1)==0) | (torch.sum(neg_mask, dim=1) == 0))[0]
    print('zero_rows:', zero_rows)
    final_mask = torch.ones_like(losses)
    final_mask[zero_rows] = 0
    losses = losses*final_mask
    print('losses_masked:', losses)
    # print(losses)
    # loss = layer(embeds, lbls)

    # print(loss)

if __name__ == "__main__":
    test()
