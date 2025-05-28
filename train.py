import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F

from yolo_model import YoloTiny
from dataset import VOCDataset
from loss import YOLOv8Loss
from utils import (
    mean_average_precision,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)

# ======= Hyperparameters =======
seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 50
NUM_WORKERS = 2
PIN_MEMORY = False
LOAD_MODEL = False
LOAD_MODEL_FILE = "iki.pt"
IMG_DIR = "DATASET/train/images"
LABEL_DIR = "DATASET/train/labels"
NUM_CLASSES = 20
NUM_BINS = 16
GRID_SIZE = 7

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img = t(img)
        return img, bboxes

transform = Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


def dfl_decode(pred_logits):
    num_bins = pred_logits.shape[-1]
    probs = F.softmax(pred_logits, dim=-1)
    bins = torch.arange(num_bins, device=pred_logits.device).float()
    decoded = torch.sum(probs * bins, dim=-1)
    return decoded

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    model.train()
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        preds = model(x)
        loss, box_loss, cls_loss = loss_fn(preds, y)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item(), box_loss=box_loss, cls_loss=cls_loss)

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss):.4f}")

def main():
    model = YoloTiny(num_classes=NUM_CLASSES, num_bins=NUM_BINS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YOLOv8Loss(num_classes=NUM_CLASSES, num_bins=NUM_BINS)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset(
        csv_file="examples.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=GRID_SIZE,
        B=1,
        C=NUM_CLASSES,
    )

    test_dataset = VOCDataset(
        csv_file="test.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=GRID_SIZE,
        B=1,
        C=NUM_CLASSES,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=True,
    )

    best_map = 0.6
    for epoch in range(EPOCHS):
        model.eval()
        with torch.no_grad():
            pred_boxes, true_boxes = get_bboxes(
                train_loader,
                model,
                iou_threshold=0.5,
                threshold=0.25,
                device=DEVICE,
            )

            mean_avg_prec = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=0.5,
                box_format="midpoint",
                num_classes=NUM_CLASSES,
            ).item()

        print(f"Epoch {epoch+1}/{EPOCHS}")

        # ======= TRAINING =======
        model.train()
        train_fn(train_loader, model, optimizer, loss_fn)

        if mean_avg_prec > best_map:
            print(f"Saving checkpoint with mAP {mean_avg_prec:.4f}")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
            best_map = mean_avg_prec

if __name__ == "__main__":
    main()
