import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConditionalUNet(nn.Module):
    def __init__(self, num_colors, base_c=64, img_channels=1, out_channels=3, embed_dim=32):
        super().__init__()
        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, base_c),
            nn.ReLU(inplace=True)
        )
        self.inc = DoubleConv(img_channels + 1, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.up1 = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up2 = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up3 = Up(base_c * 2 + base_c, base_c)
        self.outc = nn.Conv2d(base_c, out_channels, kernel_size=1)

    def forward(self, img, color_idx):
        emb = self.color_embed(color_idx)  # [B, E]
        cond = self.embed_proj(emb)       # [B, base_c]
        B, _, H, W = img.shape
        cond_map = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, base_c, H, W]
        cond_channel = torch.mean(cond_map, dim=1, keepdim=True)  # [B,1,H,W]
        x1 = self.inc(torch.cat([img, cond_channel], dim=1))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = self.outc(x)
        out = torch.sigmoid(out)
        return out
