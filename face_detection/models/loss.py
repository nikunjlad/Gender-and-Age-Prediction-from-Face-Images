"""RetinaFace training loss: box, face/background and landmark terms combined.

The loss has three parts:
    Box loss: how far the predicted box offsets are from the target offsets.
    Classification loss: how well each anchor is labelled face or background.
    Landmark loss: how far the predicted 5 landmarks are from the labelled ones.

An image has about 16,800 anchors but usually only a few dozen face anchors. If every
background anchor counted, the model would learn to say "background" everywhere. Online
hard example mining (OHEM) fixes this: only the background anchors the model gets most
wrong are kept, up to neg_pos_ratio times the number of face anchors. Example: with 10
face anchors and neg_pos_ratio 7, the 70 most confusing background anchors are used and
the rest are ignored.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    """Combined RetinaFace loss with hard negative mining.

    The total is loc_weight * box_loss + class_loss + landm_weight * landmark_loss.

    Args:
        neg_pos_ratio: How many background anchors to keep per face anchor.
        loc_weight: Weight of the box loss in the total.
        landm_weight: Weight of the landmark loss in the total.
        min_negatives_per_image: Minimum background anchors kept per image. This lets
            images with no faces still teach the model what background looks like.
    """

    def __init__(self, neg_pos_ratio=7, loc_weight=2.0, landm_weight=1.0, min_negatives_per_image=0):
        super().__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.loc_weight = loc_weight
        self.landm_weight = landm_weight
        self.min_neg = min_negatives_per_image

    def forward(self, preds, targets):
        """Computes the loss for one batch.

        Args:
            preds: Model outputs (loc, conf, landm) with shapes (B, P, 4), (B, P, 2)
                and (B, P, 10). B is the batch size and P the number of anchors.
            targets: Dict from the DataLoader with "loc" (B, P, 4), "conf" (B, P),
                "landm" (B, P, 10) and "landm_mask" (B, P).

        Returns:
            A tuple (total, parts). total is the scalar loss to backpropagate. parts is a
            dict with the detached "loc", "cls" and "landm" losses, for logging.
        """
        loc_p, conf_p, landm_p = (p.float() for p in preds)  # float32 for stable loss maths under AMP
        loc_t, conf_t = targets["loc"], targets["conf"]
        landm_t, landm_mask = targets["landm"], targets["landm_mask"]
        b, p = conf_t.shape  # batch size, anchors per image

        pos = conf_t > 0  # face anchors
        num_pos = pos.sum(1, keepdim=True)  # face anchors per image
        n = pos.sum().clamp(min=1).float()  # total face anchors, at least 1 to avoid divide-by-zero
        n_lm = landm_mask.sum().clamp(min=1).float()  # total anchors with landmark labels

        # Box and landmark losses only count face anchors.
        loss_l = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction="sum") / n
        loss_lm = F.smooth_l1_loss(landm_p[landm_mask], landm_t[landm_mask], reduction="sum") / n_lm

        ce = F.cross_entropy(conf_p.reshape(-1, 2), conf_t.reshape(-1), reduction="none").view(b, p)  # per-anchor
        with torch.no_grad():  # picking hard negatives needs no gradient
            mining = ce.masked_fill(pos, -1.0)  # push face anchors to the bottom of the ranking
            _, order = mining.sort(1, descending=True)  # anchors sorted by loss, highest first
            _, rank = order.sort(1)  # each anchor's position in that ranking
            num_neg = (self.neg_pos_ratio * num_pos).clamp(min=self.min_neg, max=p - 1)  # negatives to keep
            neg = rank < num_neg  # the hardest background anchors
        loss_c = ce[pos | neg].sum() / n  # class loss over face anchors + hard negatives

        total = self.loc_weight * loss_l + loss_c + self.landm_weight * loss_lm
        return total, {"loc": loss_l.detach(), "cls": loss_c.detach(), "landm": loss_lm.detach()}
