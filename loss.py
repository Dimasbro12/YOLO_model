import torch
import torch.nn as nn
import torch.nn.functional as F

class DistributionFocalLoss(nn.Module):
    def __init__(self, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        self.bin_vals = torch.arange(0, num_bins).float()

    def forward(self, pred_logits, target):
        device = pred_logits.device
        self.bin_vals = self.bin_vals.to(device)
        pred_probs = F.softmax(pred_logits, dim=-1)  # (N, 4, num_bins)
        target_left = target.floor().long()
        target_right = target_left + 1
        weight_right = target - target_left.float()
        weight_left = 1.0 - weight_right
        target_left = target_left.clamp(0, self.num_bins - 1)
        target_right = target_right.clamp(0, self.num_bins - 1)
        loss = 0.0
        for i in range(4):
            prob = pred_probs[:, i]  # (N, num_bins)
            left_idx = target_left[:, i]
            right_idx = target_right[:, i]
            wl = weight_left[:, i]
            wr = weight_right[:, i]
            left_prob = prob[torch.arange(len(prob)), left_idx]
            right_prob = prob[torch.arange(len(prob)), right_idx]
            loss_i = -torch.log(left_prob + 1e-6) * wl - torch.log(right_prob + 1e-6) * wr
            loss += loss_i.mean()
        return loss / 4

class YOLOv8Loss(nn.Module):
    def __init__(self, num_classes=20, num_bins=16, lambda_box=7.5, lambda_obj=1.0, lambda_cls=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

        self.dfl = DistributionFocalLoss(num_bins=num_bins)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        B, S, _, _ = targets.shape
        preds = predictions.view(B, S, S, 4 * self.num_bins + 1 + self.num_classes)

        bbox_logits = preds[..., :4 * self.num_bins].view(B, S, S, 4, self.num_bins)
        conf_logits = preds[..., 4 * self.num_bins]
        cls_logits = preds[..., 4 * self.num_bins + 1:]

        target_box = targets[..., 1:5] * self.num_bins
        target_conf = targets[..., 5]
        target_cls = targets[..., 0].long()

        obj_mask = target_conf > 0

        if obj_mask.sum() > 0:
            pred_boxes_pos = bbox_logits[obj_mask]
            target_boxes_pos = target_box[obj_mask]
            box_loss = self.dfl(pred_boxes_pos, target_boxes_pos)
        else:
            box_loss = torch.tensor(0.0, device=preds.device)

        obj_loss = self.bce(conf_logits, target_conf.float())

        if obj_mask.sum() > 0:
            cls_loss = self.bce(
                cls_logits[obj_mask],
                F.one_hot(target_cls[obj_mask].long(), self.num_classes).float()
            )
        else:
            cls_loss = torch.tensor(0.0, device=preds.device)

        total_loss = (
            self.lambda_box * box_loss +
            self.lambda_obj * obj_loss +
            self.lambda_cls * cls_loss
        )

        return total_loss, box_loss.item(), cls_loss.item()
