import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

# Frequency Spatial Residual Block (FSRB)
class FrequencySpatialResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.freq_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Frequency path
        fft_features = fft.fft2(x, dim=(-2, -1))
        amp, phase = torch.abs(fft_features), torch.angle(fft_features)
        amp = F.relu(self.freq_conv(amp))
        phase = F.relu(self.freq_conv(phase))
        freq_out = fft.ifft2(amp * torch.exp(1j * phase), dim=(-2, -1)).real

        # Spatial path
        spatial_out = F.relu(self.spatial_conv(x))
        
        return freq_out + spatial_out


# Frequency Spatial Network (FS-Net)
class FSNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            FrequencySpatialResidualBlock(in_channels),
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Dual Color Encoder (DCE)
class DualColorEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.color_encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.color_encoder2 = nn.Transformer(d_model=in_channels, nhead=4, num_encoder_layers=3)

    def forward(self, x):
        features = self.color_encoder1(x)
        features_flat = features.flatten(2).permute(2, 0, 1)  # Prepare for Transformer
        color_query = self.color_encoder2(features_flat)
        return color_query.mean(dim=0).view(features.shape)


# Fusion Network
class FusionNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fusion = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, coarse, color_features):
        combined = torch.cat([coarse, color_features], dim=1)
        fused = self.fusion(combined)
        return fused


# FDCE-Net
class FDCENet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.fs_net = FSNet(in_channels)
        self.dce = DualColorEncoder(in_channels)
        self.fusion_net = FusionNet(in_channels)

    def forward(self, x):
        coarse_enhanced = self.fs_net(x)
        color_features = self.dce(x)
        final_output = self.fusion_net(coarse_enhanced, color_features)
        return final_output


# Testing the network
if __name__ == "__main__":
    model = FDCENet(in_channels=3)
    input_image = torch.randn(1, 3, 256, 256)  # Batch size, Channels, Height, Width
    output_image = model(input_image)
    print("Output image shape:", output_image.shape)
