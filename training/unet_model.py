import torch
import torch.nn as nn


class UNet(nn.Module):
    """U-Net with bilinear upsampling (no ConvTranspose => no checkerboard)."""

    def __init__(self, in_channels=1, out_channels=1, load_from=None):
        super().__init__()

        # --- Encoder ---
        self.encode1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
        )
        self.encode2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
        )
        self.encode3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
        )
        self.encode4 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
        )
        self.encode5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1), nn.LeakyReLU(0.1), nn.MaxPool2d(2)
        )

        # --- Bottleneck + first upsample ---
        self.encode6 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )

        # --- Decoder ---
        self.decode1 = nn.Sequential(
            nn.Conv2d(96, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.decode3 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.decode4 = nn.Sequential(
            nn.Conv2d(144, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        )
        self.decode5 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1),
        )

        # --- Output ---
        self.output_layer = nn.Conv2d(32, out_channels, 3, padding=1)

        if load_from is None:
            self._init_weights()
        else:
            # Always materialize on CPU, then let the caller .to(device):
            # the tracked checkpoints carry MPS-tagged storages, which fail
            # to load on machines without the MPS backend (Linux/CUDA box,
            # RunPod). Weights are identical either way.
            self.load_state_dict(
                torch.load(load_from, map_location="cpu", weights_only=True))

    def forward(self, x):
        p1 = self.encode1(x)
        p2 = self.encode2(p1)
        p3 = self.encode3(p2)
        p4 = self.encode4(p3)
        p5 = self.encode5(p4)

        u5 = self.encode6(p5)
        u4 = self.decode1(torch.cat([u5, p4], 1))
        u3 = self.decode2(torch.cat([u4, p3], 1))
        u2 = self.decode3(torch.cat([u3, p2], 1))
        u1 = self.decode4(torch.cat([u2, p1], 1))
        u0 = self.decode5(torch.cat([u1, x], 1))
        return self.output_layer(u0)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
