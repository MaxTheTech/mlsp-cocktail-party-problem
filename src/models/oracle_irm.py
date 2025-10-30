import torch
import torch.nn as nn


class OracleIRM(nn.Module):
    """
    Oracle Ideal Ratio Mask
    Uses ground truth sources to compute optimal time-frequency masks.
    Provides theoretical upper bound for mask-based separation methods.
    """

    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, mixture, clean_sources):
        """
        Args:
            mixture: [batch, time] - mixture waveform
            clean_sources: [batch, n_src, time] - ground truth sources
        Returns:
            separated: [batch, n_src, time] - separated sources
        """
        batch_size, n_src, _ = clean_sources.shape
        device = mixture.device

        # check window device
        if self.window.device != device:
            self.window = self.window.to(device)

        # compute mixture STFT
        mix_stft = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True
        )

        #cCompute source STFTs and magnitudes
        source_stfts = []
        source_mags = []
        for i in range(n_src):
            src_stft = torch.stft(
                clean_sources[:, i, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                return_complex=True
            )
            source_stfts.append(src_stft)
            source_mags.append(torch.abs(src_stft))

        # compute total magnitude for IRM denominator
        total_mag = sum(source_mags)

        # separate each source using IRM
        separated = []
        for i in range(n_src):
            # Ideal Ratio Mask: source_i / (sum of all sources)
            mask = source_mags[i] / (total_mag + 1e-8)

            # apply mask to mixture STFT
            masked_stft = mask * mix_stft

            # inverse STFT
            audio = torch.istft(
                masked_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window,
                length=mixture.shape[-1]
            )
            separated.append(audio)

        return torch.stack(separated, dim=1)

    def forward_dict(self, batch):
        """
        Forward pass accepting dictionary from dataloader

        Args:
            batch: Dictionary with keys 'mixture', 'sources', 'lengths' (optional)
        Returns:
            separated: [batch, n_src, time] - separated sources
        """
        mixture = batch['mixture']
        sources = batch['sources']
        lengths = batch.get('lengths', None)

        return self.forward(mixture, sources, lengths)

