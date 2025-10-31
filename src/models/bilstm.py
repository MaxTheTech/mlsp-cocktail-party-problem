import torch
import torch.nn as nn


class BiLSTMSeparator(nn.Module):
    """
    BiLSTM-based voice separation model that predicts masks in time-frequency domain.

    Uses Bidirectional LSTM to learn time-frequency masks from the mixture STFT magnitude.
    Applies masks to the mixture STFT to separate sources.
    """

    def __init__(self, num_sources=2, num_layers=3, hidden_size=512, dropout=0.5,n_fft=1024, hop_length=256):
        """
        Args:
            num_sources: Number of sources to separate
            num_layers: Number of BiLSTM layers
            hidden_size: Hidden dimension of LSTM
            dropout: Dropout rate
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
        """
        super().__init__()

        self.num_sources = num_sources
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_fft = n_fft
        self.hop_length = hop_length

        # register window as buffer for STFT
        self.register_buffer('window', torch.hann_window(n_fft))

        # STFT produces n_fft // 2 + 1 frequency bins
        self.num_freq_bins = n_fft // 2 + 1

        # BiLSTM layers, input is magnitude spectrogram
        self.bilstm = nn.LSTM(
            input_size=self.num_freq_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # layer normalization for stability
        self.layer_norm = nn.LayerNorm(2 * hidden_size)

        # mask estimation head: output num_sources masks
        # bidirectional LSTM outputs 2*hidden_size
        self.mask_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_sources * self.num_freq_bins)
        )

    def forward(self, mixture):
        """
        Forward pass: separate sources from mixture waveform

        Args:
            mixture: [batch, time] - mixture waveform
        Returns:
            separated: [batch, num_sources, time] - separated sources
        """
        batch_size = mixture.shape[0]
        device = mixture.device
        original_length = mixture.shape[-1]

        # check if we need CPU fallback for STFT and ISTFT (doesn't support MPS)
        use_cpu_for_istft = device.type == 'mps'

        # if MPS: move mixture and window to CPU
        if use_cpu_for_istft:
            mixture = mixture.cpu()
            window_device = self.window.cpu()
        else:
            window_device = self.window

        # compute mixture STFT
        mix_stft = torch.stft(
            mixture,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window_device,
            return_complex=True
        )

        # if MPS: move mixture STFT back to MPS
        if use_cpu_for_istft:
            mix_stft = mix_stft.to(device)

        # Get magnitude spectrogram [batch, freq, time]
        mix_mag = torch.abs(mix_stft)

        # transpose to [batch, time, freq] for LSTM
        # mix_mag_t = mix_mag.transpose(1, 2)
        mix_mag_t = torch.log1p(mix_mag).transpose(1, 2) # log compression

        # pass through BiLSTM
        lstm_out, _ = self.bilstm(mix_mag_t)

        # apply layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # estimate masks [batch, time, num_sources*freq]
        mask_logits = self.mask_head(lstm_out)

        # reshape to [batch, time, num_sources, freq]
        mask_logits = mask_logits.view(
            batch_size, -1, self.num_sources, self.num_freq_bins
        )

        # apply sigmoid to get masks, allows source overlap
        masks = torch.sigmoid(mask_logits)

        # transpose masks to [batch, num_sources, freq, time]
        masks = masks.permute(0, 2, 3, 1)

        # apply masks to mixture STFT
        # expand mix_stft for broadcasting: [batch, 1, freq, time]
        mix_stft_expanded = mix_stft.unsqueeze(1)
        
        # apply masks: [batch, num_sources, freq, time]
        masked_stft = masks * mix_stft_expanded

        # if MPS: move mask STFT and window to CPU
        if use_cpu_for_istft:
            masked_stft = masked_stft.cpu()
            window_device = self.window.cpu()
        else:
            window_device = self.window

        # inverse STFT for each source
        separated = []
        for src_idx in range(self.num_sources):
            # get masked STFT for this source: [batch, freq, time]
            src_stft = masked_stft[:, src_idx, :, :]
            
            # inverse STFT to get waveform
            audio = torch.istft(
                src_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=window_device,
                length=original_length
            )
            
            # if MPS: move separated audio back to MPS
            if use_cpu_for_istft:
                audio = audio.to(device)

            separated.append(audio)
        
        return torch.stack(separated, dim=1)

    def separate(self, mixture_single):
        """
        Convenience method for separating a single (unsqueezed) waveform

        Args:
            mixture_single: [time] - single mixture waveform
        Returns:
            tuple of source tensors, one per source
        """
        # add batch dimension
        mixture_batch = mixture_single.unsqueeze(0)
        separated_batch = self.forward(mixture_batch)

        # remove batch dimension and return as tuple
        separated = tuple(separated_batch[0, i] for i in range(self.num_sources))
        return separated
