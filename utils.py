import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from typing import List, Tuple

def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format="midpoint") -> torch.Tensor:
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners":
        box1_x1, box1_y1, box1_x2, box1_y2 = boxes_preds[..., 0:1], boxes_preds[..., 1:2], boxes_preds[..., 2:3], boxes_preds[..., 3:4]
        box2_x1, box2_y1, box2_x2, box2_y2 = boxes_labels[..., 0:1], boxes_labels[..., 1:2], boxes_labels[..., 2:3], boxes_labels[..., 3:4]
    else:
        raise ValueError(f"Invalid box_format {box_format}")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    box1_area = (box1_x2 - box1_x1).clamp(min=0) * (box1_y2 - box1_y1).clamp(min=0)
    box2_area = (box2_x2 - box2_x1).clamp(min=0) * (box2_y2 - box2_y1).clamp(min=0)

    return intersection / (box1_area + box2_area - intersection + 1e-6)

def non_max_suppression(bboxes: List[List[float]], iou_threshold: float, threshold: float, box_format="corners") -> List[List[float]]:
    assert isinstance(bboxes, list)
    bboxes = [box for box in bboxes if box[1] > threshold]  # filter by confidence threshold
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0] or intersection_over_union(
                torch.tensor(chosen_box[2:], dtype=torch.float32),
                torch.tensor(box[2:], dtype=torch.float32),
                box_format=box_format,
            ).item() < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

import torch
from collections import Counter
from utils import intersection_over_union

def mean_average_precision(
    pred_boxes, 
    true_boxes, 
    iou_threshold=0.5, 
    box_format="midpoint", 
    num_classes=20
):
    average_precisions = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1] == c]
        ground_truths = [box for box in true_boxes if box[1] == c]

        amount_bboxes = Counter([gt[0] for gt in ground_truths])  # image_idx: count
        for key in amount_bboxes:
            amount_bboxes[key] = torch.zeros(amount_bboxes[key])

        detections.sort(key=lambda x: x[2], reverse=True)  # sort by confidence
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truths)

        for det_idx, detection in enumerate(detections):
            image_idx = detection[0]
            best_iou = 0
            best_gt_idx = -1

            gts_in_img = [gt for gt in ground_truths if gt[0] == image_idx]

            for gt_idx, gt in enumerate(gts_in_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > iou_threshold:
                if amount_bboxes[image_idx][best_gt_idx] == 0:
                    TP[det_idx] = 1
                    amount_bboxes[image_idx][best_gt_idx] = 1  # mark as used
                else:
                    FP[det_idx] = 1
            else:
                FP[det_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))

        # 11-point interpolation (optional)
        ap = torch.trapz(precisions, recalls)
        average_precisions.append(ap)

    if len(average_precisions) == 0:
        return torch.tensor(0.0)

    return sum(average_precisions) / len(average_precisions)

def plot_image(loader, model, threshold, iou_threshold, device, class_names=None):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)

    with torch.no_grad():
        out = model(x)
        bboxes = non_max_suppression(out, iou_threshold=iou_threshold, threshold=threshold)

    img = x[0].permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    if bboxes[0] is not None:
        for box in bboxes[0]:
            x_center, y_center, width, height, conf, cls = box[:6]
            x1 = x_center - width / 2
            y1 = y_center - height / 2

            rect = patches.Rectangle(
                (x1 * 448, y1 * 448), width * 448, height * 448,
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            if class_names:
                ax.text(x1 * 448, y1 * 448, class_names[int(cls)], color='white', backgroundcolor='red')

    plt.show()

def get_bboxes(loader, model, iou_threshold, threshold, box_format="midpoint", device="cpu") -> Tuple[List, List]:
    all_pred_boxes, all_target_boxes = [], []
    model.eval()
    train_idx = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_boxes = cellboxes_to_boxes(predictions, S=7)
        true_boxes = cellboxes_to_boxes(y, S=7)

        for idx in range(x.shape[0]):
            nms_boxes = non_max_suppression(batch_boxes[idx], iou_threshold, threshold, box_format)
            all_pred_boxes.extend([[train_idx] + box for box in nms_boxes])
            all_target_boxes.extend([[train_idx] + box for box in true_boxes[idx] if box[1] > threshold])
            train_idx += 1

    model.train()
    return all_pred_boxes, all_target_boxes

def convert_cellboxes(predictions: torch.Tensor, S=7) -> torch.Tensor:
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, -1)  # (B, S, S, 25)

    # Ambil bbox tunggal: [x, y, w, h, confidence] mulai dari channel ke-20 sampai 24
    bboxes = predictions[..., 20:25]  # (B, S, S, 5)

    device = predictions.device
    cell_indices_x = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1).to(device)  # (B, S, S, 1)
    cell_indices_y = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1).permute(0, 2, 1, 3).to(device)  # (B, S, S, 1)

    x = (bboxes[..., 0:1] + cell_indices_x) / S
    y = (bboxes[..., 1:2] + cell_indices_y) / S
    w_h = bboxes[..., 2:4] / S

    converted_bboxes = torch.cat((x, y, w_h), dim=-1)  # (B, S, S, 4)

    predicted_class = predictions[..., :20].argmax(dim=-1).unsqueeze(-1).float()  # (B, S, S, 1)

    best_confidence = bboxes[..., 4].unsqueeze(-1)  # (B, S, S, 1)

    # Gabungkan: [class, confidence, x, y, w, h]
    return torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

def cellboxes_to_boxes(out: torch.Tensor, S=7) -> List[List[List[float]]]:
    converted_pred = convert_cellboxes(out, S).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = [converted_pred[ex_idx, bbox_idx, :].tolist() for bbox_idx in range(S * S)]
        all_bboxes.append(bboxes)
    return all_bboxes

def save_checkpoint(state: dict, filename: str = "checkpoint.pt") -> None:
    print("📂 Saving checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint: dict, model: torch.nn.Module, optimizer=None) -> None:
    print("🔁 Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    print("✅ Checkpoint loaded.")
