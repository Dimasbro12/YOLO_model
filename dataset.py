import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

class VOCDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, transform=None, S=7, B=1, C=20):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path).convert("RGB")
        boxes = []

        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, w, h = map(float, label.strip().split())
                boxes.append([class_label, x, y, w, h])

        if self.transform:
            image, _ = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, 5 + self.C))  # (class, x, y, w, h, confidence)
        for box in boxes:
            class_label, x, y, w, h = box
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            w_cell, h_cell = w * self.S, h * self.S

            if label_matrix[i, j, 0] == 0:
                label_matrix[i, j, 0] = 1  # objectness
                label_matrix[i, j, 1:5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, 5 + class_label] = 1

        return image, label_matrix
