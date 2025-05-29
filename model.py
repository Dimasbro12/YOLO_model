# === model.py ===
import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------------------------------------------
# Utility Blocks (Conv, C2f, Bottleneck, SPPF)
#--------------------------------------------------
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return x + out if self.use_shortcut else out

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.convs = nn.ModuleList([Bottleneck(hidden_channels, hidden_channels, shortcut) for _ in range(n)])
        self.conv2 = Conv((n + 2) * hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(2, 1)
        outputs = [x1, x2]
        for conv in self.convs:
            x2 = conv(x2)
            outputs.append(x2)
        return self.conv2(torch.cat(outputs, dim=1))

class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=5):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = Conv(out_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))

#--------------------------------------------------
# Detection Head (YOLOv8-style)
#--------------------------------------------------
class Detect(nn.Module):
    def __init__(self, num_classes=3, num_bins=16, channels=(256, 512, 1024)):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.out_dim = 4 * num_bins + 1 + num_classes

        self.detect_layers = nn.ModuleList([
            nn.Sequential(
                Conv(c, c, 3),
                nn.Conv2d(c, self.out_dim, 1)
            ) for c in channels
        ])

    def forward(self, x):
        outs = []
        for xi, detect in zip(x, self.detect_layers):
            out = detect(xi)  
            out = out.permute(0, 2, 3, 1).contiguous()  
            outs.append(out.view(out.size(0), -1, self.out_dim)) 
        return torch.cat(outs, dim=1)  

#--------------------------------------------------
# YOLOv8 Architecture
#--------------------------------------------------
class YOLOv8(nn.Module):
    def __init__(self, num_classes=3, width=1.0, depth=1.0, num_bins=16):
        super().__init__()

        def c(ch): return max(int(ch * width), 1)
        def d(n): return max(round(n * depth), 1)

        # Stem
        self.stem = nn.Sequential(
            Conv(3, c(64), 3, 2),
            Conv(c(64), c(128), 3, 2),
            C2f(c(128), c(128), n=d(3)),
            Conv(c(128), c(256), 3, 2),
            C2f(c(256), c(256), n=d(6)),
            Conv(c(256), c(512), 3, 2),
            C2f(c(512), c(512), n=d(6)),
            Conv(c(512), c(1024), 3, 2),
            C2f(c(1024), c(1024), n=d(3)),
            SPPF(c(1024), c(1024))
        )

        # Neck
        self.neck = nn.Sequential(
            Conv(c(1024), c(512), 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.c2f_neck_1 = C2f(c(1024), c(512), n=d(3), shortcut=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_neck_2 = C2f(c(768), c(256), n=d(3), shortcut=False)

        # Bottom-up
        self.down1 = Conv(c(256), c(256), 3, 2)
        self.c2f_down_1 = C2f(c(768), c(512), n=d(3), shortcut=False)

        self.down2 = Conv(c(512), c(512), 3, 2)
        self.c2f_down_2 = C2f(c(1536), c(1024), n=d(3), shortcut=False)

        # Head
        self.detect = Detect(num_classes=num_classes, num_bins=num_bins, channels=(c(256), c(512), c(1024)))

    def forward(self, x):
        outputs = []
        for layer in self.stem:
            x = layer(x)
            outputs.append(x)

        P5 = outputs[-1]
        x = self.neck[0](P5)
        x = self.neck[1](x)
        x = torch.cat((x, outputs[6]), dim=1)
        x = self.c2f_neck_1(x)

        x = self.upsample2(x)
        x = torch.cat((x, outputs[4]), dim=1)
        P3 = self.c2f_neck_2(x)

        x = self.down1(P3)
        x = torch.cat((x, outputs[6]), dim=1)
        P4 = self.c2f_down_1(x)

        x = self.down2(P4)
        x = torch.cat((x, P5), dim=1)
        P5_out = self.c2f_down_2(x)

        return self.detect([P3, P4, P5_out])

#--------------------------------------------------
# Test Run
#--------------------------------------------------
if __name__ == "__main__":
    model = YOLOv8(num_classes=3, num_bins=16)
    dummy_input = torch.randn(1, 3, 640, 640)
    outputs = model(dummy_input)
    print("Output shape:", outputs.shape)  # [B, N, 4*num_bins+1+num_classes]
