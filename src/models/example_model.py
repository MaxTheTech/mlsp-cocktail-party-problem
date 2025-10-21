"""Simple baseline models for audio source separation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSourceSeparationModel(nn.Module):
    """
    Simple encoder-decoder model for separating audio sources
    Uses 1D convolutions with masking approach
    """

    def __init__(self, num_sources=2, encoder_channels=256,
                 hidden_channels=512, num_layers=4,
                 kernel_size=16, stride=8):
        super().__init__()

        self.num_sources = num_sources
        self.encoder_channels = encoder_channels
        self.stride = stride
        self.kernel_size = kernel_size

        # Encoder: waveform -> latent representation
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2
        )

        # Bottleneck: process temporal features
        self.bottleneck = nn.ModuleList()
        for i in range(num_layers):
            in_ch = encoder_channels if i == 0 else hidden_channels
            self.bottleneck.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_channels),
                    nn.PReLU(),
                    nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_channels),
                    nn.PReLU()
                )
            )

        # Mask generator: create masks for each source
        self.mask_estimator = nn.Sequential(
            nn.Conv1d(hidden_channels, encoder_channels * num_sources, kernel_size=1),
            nn.Sigmoid()
        )

        # Decoder: latent -> waveform
        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1
        )

    def forward(self, mixture):
        """Forward pass: mixture -> separated sources"""
        # Ensure input is [B, 1, T]
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)

        batch_size, _, time_steps = mixture.shape

        # Encode
        encoded = self.encoder(mixture)

        # Process through bottleneck with residual connections
        hidden = encoded
        for layer in self.bottleneck:
            residual = hidden
            hidden = layer(hidden)
            if hidden.shape[1] == residual.shape[1]:
                hidden = hidden + residual

        # Estimate masks
        masks = self.mask_estimator(hidden)
        masks = masks.view(batch_size, self.num_sources, self.encoder_channels, -1)

        # Apply masks to encoded representation
        encoded_expanded = encoded.unsqueeze(1)
        masked_encoded = masks * encoded_expanded

        # Decode each source
        sources = []
        for i in range(self.num_sources):
            decoded = self.decoder(masked_encoded[:, i, :, :])
            sources.append(decoded.squeeze(1))

        separated = torch.stack(sources, dim=1)

        # Match input length
        if separated.shape[-1] > time_steps:
            separated = separated[..., :time_steps]
        elif separated.shape[-1] < time_steps:
            padding = time_steps - separated.shape[-1]
            separated = F.pad(separated, (0, padding))

        return separated

    def separate(self, mixture):
        """Convenience method for inference"""
        single_input = mixture.ndim == 1
        if single_input:
            mixture = mixture.unsqueeze(0)

        separated = self.forward(mixture)

        # Split into individual sources
        sources = tuple(separated[:, i, :] for i in range(self.num_sources))

        if single_input:
            sources = tuple(s.squeeze(0) for s in sources)

        return sources


class SimpleRNN(nn.Module):
    """RNN-based separation model using LSTM for temporal modeling"""

    def __init__(self, num_sources=2, input_size=512,
                 hidden_size=512, num_layers=2, bidirectional=True):
        super().__init__()

        self.num_sources = num_sources
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, input_size, kernel_size=512, stride=256, padding=256),
            nn.ReLU(),
            nn.BatchNorm1d(input_size)
        )

        # Temporal modeling with LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )

        # Mask estimation
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.mask_estimator = nn.Sequential(
            nn.Linear(lstm_output_size, input_size * num_sources),
            nn.Sigmoid()
        )

        # Reconstruction
        self.decoder = nn.ConvTranspose1d(
            input_size, 1, kernel_size=512, stride=256, padding=256
        )

    def forward(self, mixture):
        """Forward pass: mixture -> separated sources"""
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)

        batch_size, _, time_steps = mixture.shape

        # Extract features
        features = self.feature_extractor(mixture)

        # Process with LSTM
        features_transposed = features.transpose(1, 2)
        lstm_out, _ = self.lstm(features_transposed)

        # Estimate masks
        masks = self.mask_estimator(lstm_out)
        masks = masks.view(batch_size, -1, self.num_sources, self.input_size)

        # Apply masks
        features_expanded = features_transposed.unsqueeze(2)
        masked_features = masks * features_expanded

        # Decode each source
        sources = []
        for i in range(self.num_sources):
            source_features = masked_features[:, :, i, :].transpose(1, 2)
            decoded = self.decoder(source_features).squeeze(1)
            sources.append(decoded)

        separated = torch.stack(sources, dim=1)

        # Match input length
        if separated.shape[-1] > time_steps:
            separated = separated[..., :time_steps]
        elif separated.shape[-1] < time_steps:
            padding = time_steps - separated.shape[-1]
            separated = F.pad(separated, (0, padding))

        return separated


def si_snr_loss(estimated, target, eps=1e-8):
    """
    Scale-Invariant SNR loss for source separation
    Higher SI-SNR is better, so we return negative for minimization
    """
    # Normalize
    target = target - target.mean(dim=-1, keepdim=True)
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)

    # Compute SI-SNR
    s_target = (torch.sum(estimated * target, dim=-1, keepdim=True) /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target

    e_noise = estimated - s_target

    si_snr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )

    # Return negative for minimization
    return -si_snr.mean()


if __name__ == "__main__":
    # Quick test
    print("Testing model...")

    model = SimpleSourceSeparationModel(num_sources=2)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(4, 32000)  # 4 samples, 2 seconds at 16kHz
    output = model(dummy_input)
    print(f"Input: {dummy_input.shape}")
    print(f"Output: {output.shape}")
    assert output.shape == (4, 2, 32000)

    # Test loss
    dummy_target = torch.randn(4, 2, 32000)
    loss = si_snr_loss(output, dummy_target)
    print(f"Loss: {loss.item():.4f}")

    print("All tests passed!")
