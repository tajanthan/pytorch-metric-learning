import torch

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss

def sigmoid(tensor, temp=1.0):
    """ temperature controlled sigmoid
    takes as input a torch tensor (tensor) and passes it through a sigmoid, controlled by temperature: temp
    """
    exponent = -tensor / temp
    # clamp the input tensor for stability
    exponent = torch.clamp(exponent, min=-50, max=50)
    y = 1.0 / (1.0 + torch.exp(exponent))
    return y


class SmoothAPLoss(GenericPairLoss):
    def __init__(self, temperature=0.01, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)

    """
    Adapted from https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/Smooth_AP_loss.py
	With more flexibility:
		1. does not require the minibatch to be in the particular format
		2. embeddings and ref_emb do not need to be the same
		3. supports XBM
    """
    def _compute_loss(self, aff_mat, pos_mask, neg_mask):
        bz, m = aff_mat.size()

        # if dealing with actual distances, use negative distances
        if not self.distance.is_inverted:
            aff_mat = -aff_mat

        if pos_mask.bool().any() and neg_mask.bool().any():
            aff_mat_repeat = aff_mat.unsqueeze(dim=1).repeat(1, m, 1)
            # compute the difference matrix
            aff_diff = aff_mat_repeat - aff_mat_repeat.permute(0, 2, 1)
            # pass through the sigmoid
            aff_sg = sigmoid(aff_diff, temp=self.temperature)

            pos_mask_repeat = pos_mask.unsqueeze(1).repeat(1, m, 1)
            neg_mask_repeat = neg_mask.unsqueeze(1).repeat(1, m, 1)
            aff_pos = aff_sg * pos_mask_repeat     # positive only (considers indices -- handles self-comparisons)
            aff_all = aff_pos + aff_sg * neg_mask_repeat   # positive + negative

            # rank by inner summation (\sum_{j\in S_p} G(D_{ij}))
            aff_pos_rk = torch.sum(aff_pos, dim=-1) + 1
            aff_all_rk = torch.sum(aff_all, dim=-1) + 1

            # ap by outer summation (\sum_{i\in S_p} R_(i, S_p)/R_(i, S_\Omega))
            ap_i = (aff_pos_rk / aff_all_rk) * pos_mask
            ap = torch.sum(ap_i, dim=-1) / torch.sum(pos_mask, dim=-1)

            return {
                "loss": {
                    "losses": 1. - ap,
                    "indices": c_f.torch_arange_from_size(aff_mat),
                    "reduction_type": "element",
                }
            }
        return self.zero_losses()

    def get_default_distance(self):
        return CosineSimilarity()

    def get_default_reducer(self):
        return MeanReducer()
