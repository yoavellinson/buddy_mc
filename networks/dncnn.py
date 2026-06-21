import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=None,
        features=64,
        depth=17,
        kernel_size=3,
        use_batchnorm=True,
        use_noise_level=True,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("DnCNN depth must be at least 2")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels or in_channels)
        self.use_noise_level = bool(use_noise_level)

        first_in_channels = self.in_channels + (1 if self.use_noise_level else 0)
        padding = kernel_size // 2

        layers = [
            nn.Conv2d(first_in_channels, features, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]

        for _ in range(depth - 2):
            layers.append(nn.Conv2d(features, features, kernel_size, padding=padding, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(features, self.out_channels, kernel_size, padding=padding))
        self.net = nn.Sequential(*layers)

    def forward(self, x, sigma=None):
        if self.use_noise_level:
            if sigma is None:
                raise ValueError("DnCNN was configured with use_noise_level=True, but sigma was not provided")
            if not torch.is_tensor(sigma):
                sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
            sigma = sigma.to(device=x.device, dtype=x.dtype)
            while sigma.ndim < x.ndim:
                sigma = sigma.unsqueeze(-1)
            sigma_map = sigma.expand(x.shape[0], 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, sigma_map], dim=1)

        return self.net(x)

    def denoise(self, x, sigma=None):
        pred_noise = self.forward(x, sigma=sigma)
        if sigma is None:
            return x - pred_noise
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sigma = sigma.to(device=x.device, dtype=x.dtype)
        while sigma.ndim < x.ndim:
            sigma = sigma.unsqueeze(-1)
        return x - sigma * pred_noise
