import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        
        self.conv1x1_1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.prelu1 = nn.PReLU()
        
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise_conv = nn.Conv1d(
            out_channels, out_channels, 
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=out_channels  # groups=channels for depthwise
        )

        self.norm2 = nn.GroupNorm(1, out_channels)
        self.prelu2 = nn.PReLU()
        
        self.skip_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.residual_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        # x: [B, in_channels, L]
        out = self.conv1x1_1(x)
        out = self.prelu1(self.norm1(out))
        out = self.depthwise_conv(out)
        out = self.prelu2(self.norm2(out))
        
        skip = self.skip_conv(out) # [B, in_channels, L]
        residual = self.residual_conv(out)
        output = x + residual # [B, in_channels, L]
        
        return output, skip


class TemporalConvNet(nn.Module):
    def __init__(self, bottleneck_channels, hidden_channels, kernel_size=3, num_blocks=8, num_repeats=3):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_repeats = num_repeats
        
        self.tcn_blocks = nn.ModuleList()
        
        for r in range(num_repeats):
            for x in range(num_blocks):
                dilation = 2 ** x
                self.tcn_blocks.append(
                    DepthwiseSeparableConv1d(
                        in_channels=bottleneck_channels,
                        out_channels=hidden_channels,
                        kernel_size=kernel_size,
                        dilation=dilation
                    )
                )
    
    def forward(self, x):
        # x: [B, bottleneck_channels, L]
        skip_connections = []
        
        for block in self.tcn_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        
        output = torch.stack(skip_connections, dim=0).sum(dim=0) # [B, bottleneck_channels, L]
        
        return output


class MultiScaleEncoder(nn.Module):
    """
    Multiple encoder/decoder pairs with different filter lengths
    Captures both fine-grained and coarse temporal features
    """
    def __init__(self, encoder_channels=256, kernel_sizes=[8, 16, 32, 64], strides=None):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        
        # Default strides (half of kernel size)
        if strides is None:
            self.strides = [k // 2 for k in kernel_sizes]
        else:
            self.strides = strides
        
        # Create multiple encoders with different kernel sizes
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        
        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            self.encoders.append(
                nn.Conv1d(
                    in_channels=1,
                    out_channels=encoder_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias=False
                )
            )
            self.encoder_norms.append(nn.GroupNorm(1, encoder_channels))
        
        # Fusion layer to combine multi-scale features
        self.fusion = nn.Conv1d(
            encoder_channels * self.num_scales,
            encoder_channels,
            kernel_size=1
        )
        
    def forward(self, mixture):
        # mixture: [B, 1, T]
        encoded_features = []
        
        for encoder, norm in zip(self.encoders, self.encoder_norms):
            encoded = encoder(mixture)  # [B, N, L_i]
            encoded = norm(encoded)
            encoded_features.append(encoded)
        
        # Interpolate all features to the same length (use finest scale as reference)
        target_length = encoded_features[0].shape[-1]
        
        aligned_features = []
        for i, feat in enumerate(encoded_features):
            if feat.shape[-1] != target_length:
                feat = F.interpolate(feat, size=target_length, mode='linear', align_corners=False)
            aligned_features.append(feat)
        
        # Concatenate and fuse
        concatenated = torch.cat(aligned_features, dim=1)  # [B, N*num_scales, L]
        fused = self.fusion(concatenated)  # [B, N, L]
        
        return fused, encoded_features


class MultiScaleDecoder(nn.Module):
    """
    Multiple decoders corresponding to multi-scale encoders
    Combines features from different scales for reconstruction
    """
    def __init__(self, encoder_channels=256, kernel_sizes=[8, 16, 32, 64], strides=None):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        
        if strides is None:
            self.strides = [k // 2 for k in kernel_sizes]
        else:
            self.strides = strides
        
        # Create multiple decoders
        self.decoders = nn.ModuleList()
        
        for kernel_size, stride in zip(self.kernel_sizes, self.strides):
            self.decoders.append(
                nn.ConvTranspose1d(
                    in_channels=encoder_channels,
                    out_channels=1,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    output_padding=stride - 1,
                    bias=False
                )
            )
        
    def forward(self, masked_features, target_length):
        # masked_features: list of [B, N, L] tensors at different scales
        # Decode each scale
        decoded_sources = []
        
        for i, (decoder, feat) in enumerate(zip(self.decoders, masked_features)):
            decoded = decoder(feat)  # [B, 1, T_i]
            
            # Match target length
            if decoded.shape[-1] > target_length:
                decoded = decoded[..., :target_length]
            elif decoded.shape[-1] < target_length:
                padding = target_length - decoded.shape[-1]
                decoded = F.pad(decoded, (0, padding))
            
            decoded_sources.append(decoded)
        
        # Average all scales
        output = torch.stack(decoded_sources, dim=0).mean(dim=0)  # [B, 1, T]
        
        return output


class FrequencyAwareConv(nn.Module):
    """
    Frequency-aware convolution that adapts kernel sizes based on frequency content
    Uses multiple parallel convolutions with different kernel sizes and learns to weight them
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9], dilation=1):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.num_kernels = len(kernel_sizes)
        
        # Multiple parallel convolutions with different kernel sizes
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            padding = (k - 1) * dilation // 2
            self.convs.append(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=k,
                    padding=padding,
                    dilation=dilation,
                    groups=in_channels  # Depthwise
                )
            )
        
        # Frequency content analyzer
        # Uses 1D convolutions to extract frequency-like features
        self.freq_analyzer = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(in_channels // 4, self.num_kernels),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x: [B, C, L]
        batch_size = x.shape[0]
        
        # Compute frequency-aware weights
        weights = self.freq_analyzer(x)  # [B, num_kernels]
        
        # Apply each convolution
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))  # Each is [B, C, L]
        
        # Stack and weight
        stacked = torch.stack(conv_outputs, dim=1)  # [B, num_kernels, C, L]
        
        # Reshape weights for broadcasting: [B, num_kernels, 1, 1]
        weights = weights.view(batch_size, self.num_kernels, 1, 1)
        
        # Weighted sum
        output = (stacked * weights).sum(dim=1)  # [B, C, L]
        
        return output


class HierarchicalTCN(nn.Module):
    """
    Hierarchical TCN with multiple branches at different dilation scales
    Each branch focuses on different temporal resolutions
    """
    def __init__(self, bottleneck_channels, hidden_channels, kernel_size=3, 
                 num_blocks=8, num_repeats=3, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.bottleneck_channels = bottleneck_channels
        
        # Create multiple TCN branches with different dilation progressions
        self.tcn_branches = nn.ModuleList()
        
        for scale_idx in range(num_scales):
            branch = nn.ModuleList()
            # Different starting dilation for each scale
            base_dilation = 2 ** scale_idx
            
            for r in range(num_repeats):
                for x in range(num_blocks):
                    dilation = base_dilation * (2 ** x)
                    branch.append(
                        DepthwiseSeparableConv1d(
                            in_channels=bottleneck_channels,
                            out_channels=hidden_channels,
                            kernel_size=kernel_size,
                            dilation=dilation
                        )
                    )
            self.tcn_branches.append(branch)
        
        # Cross-scale fusion
        self.scale_fusion = nn.Conv1d(
            bottleneck_channels * num_scales,
            bottleneck_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        # x: [B, bottleneck_channels, L]
        
        # Process each scale branch
        scale_outputs = []
        
        for branch in self.tcn_branches:
            branch_input = x
            skip_connections = []
            
            for block in branch:
                branch_input, skip = block(branch_input)
                skip_connections.append(skip)
            
            # Sum skip connections for this scale
            scale_out = torch.stack(skip_connections, dim=0).sum(dim=0)
            scale_outputs.append(scale_out)
        
        # Concatenate and fuse across scales
        concatenated = torch.cat(scale_outputs, dim=1)  # [B, bottleneck_channels*num_scales, L]
        fused = self.scale_fusion(concatenated)  # [B, bottleneck_channels, L]
        
        return fused


class ConvTasNet(nn.Module):
    """
    Original Conv-TasNet
    """
    def __init__(self, 
                 num_sources=2,
                 encoder_channels=256,      # N 
                 bottleneck_channels=256,   # B 
                 hidden_channels=512,       # H 
                 kernel_size=16,            # L  (encoder kernel)
                 tcn_kernel_size=3,         # P  (TCN kernel)
                 num_blocks=8,              # X 
                 num_repeats=3,             # R 
                 stride=8):                 # Encoder stride (L/2)
        super().__init__()
        
        self.num_sources = num_sources
        self.encoder_channels = encoder_channels
        self.stride = stride
        self.kernel_size = kernel_size
        
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=encoder_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False
        )

        self.encoder_norm = nn.GroupNorm(1, encoder_channels)
        
        self.bottleneck_conv = nn.Conv1d(
            encoder_channels, 
            bottleneck_channels, 
            kernel_size=1
        )
        
        self.separation = TemporalConvNet(
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            kernel_size=tcn_kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats
        )
        
        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck_channels, encoder_channels * num_sources, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.decoder = nn.ConvTranspose1d(
            in_channels=encoder_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            output_padding=stride - 1,
            bias=False
        )
    
    def forward(self, mixture):
        # mixture: [B, T] or [B, 1, T] - input mixture waveform
        # Ensure input is [B, 1, T]
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        
        batch_size, _, time_steps = mixture.shape
        
        # Encoder: waveform -> representation
        encoded = self.encoder(mixture)  # [B, N, L]
        
        # Layer normalization
        encoded_norm = self.encoder_norm(encoded)  # [B, N, L]
        
        # Bottleneck 1x1 conv
        bottleneck = self.bottleneck_conv(encoded_norm)  # [B, B, L]
        
        # Separation module - TCN
        separated_features = self.separation(bottleneck)  # [B, B, L]
        
        # Mask generation
        masks = self.mask_generator(separated_features)  # [B, N*num_sources, L]
        masks = masks.view(batch_size, self.num_sources, self.encoder_channels, -1)  # [B, num_sources, N, L]
        
        # Apply masks to encoder output
        encoded_expanded = encoded.unsqueeze(1)  # [B, 1, N, L]
        masked_encoded = masks * encoded_expanded  # [B, num_sources, N, L]
        
        # Decoder: reconstruct waveforms for each source
        sources = []
        for i in range(self.num_sources):
            decoded = self.decoder(masked_encoded[:, i, :, :])  # [B, 1, T']
            sources.append(decoded.squeeze(1))  # [B, T']
        
        separated = torch.stack(sources, dim=1)  # [B, num_sources, T']
        
        # Match input length (handle padding/truncation)
        if separated.shape[-1] > time_steps:
            separated = separated[..., :time_steps]
        elif separated.shape[-1] < time_steps:
            padding = time_steps - separated.shape[-1]
            separated = F.pad(separated, (0, padding))
        
        return separated # [B, num_sources, T]
    
    def separate(self, mixture):
        # mixture: [T] or [B, T]
        single_input = mixture.ndim == 1
        if single_input:
            mixture = mixture.unsqueeze(0)
        
        separated = self.forward(mixture)
        
        # Split into individual sources
        sources = tuple(separated[:, i, :] for i in range(self.num_sources))
        
        if single_input:
            sources = tuple(s.squeeze(0) for s in sources)
        
        return sources # tuple of [T] or [B, T] tensors - separated sources

def si_snr_loss(estimated, target, eps=1e-8):
    """
    Scale-Invariant SNR loss for source separation with Permutation Invariant Training (PIT)
    Higher SI-SNR is better, so we return negative for minimization

    Args:
        estimated: [B, num_sources, T] - estimated sources
        target: [B, num_sources, T] - ground truth sources
        eps: Small constant for numerical stability

    Returns:
        Negative SI-SNR loss (lower is better)
    """
    batch_size, num_sources, _ = estimated.shape

    # Compute SI-SNR for all permutations
    # For 2 sources, we have 2 permutations: [0,1] and [1,0]

    def compute_si_snr(est, tgt):
        """Compute SI-SNR between estimated and target signals"""
        # Normalize
        tgt = tgt - tgt.mean(dim=-1, keepdim=True)
        est = est - est.mean(dim=-1, keepdim=True)

        # Compute SI-SNR
        s_target = (torch.sum(est * tgt, dim=-1, keepdim=True) /
                    (torch.sum(tgt ** 2, dim=-1, keepdim=True) + eps)) * tgt

        e_noise = est - s_target

        si_snr = 10 * torch.log10(
            (torch.sum(s_target ** 2, dim=-1) + eps) /
            (torch.sum(e_noise ** 2, dim=-1) + eps)
        )

        return si_snr

    # Generate all permutations (for num_sources=2, this is [[0,1], [1,0]])
    import itertools
    perms = list(itertools.permutations(range(num_sources)))

    # Compute loss for each permutation
    losses = []
    for perm in perms:
        # Apply permutation to estimated sources
        perm_estimated = estimated[:, perm, :]  # [B, num_sources, T]

        # Compute SI-SNR for each source and sum
        si_snr_sum = 0
        for src_idx in range(num_sources):
            si_snr_val = compute_si_snr(perm_estimated[:, src_idx, :], target[:, src_idx, :])
            si_snr_sum = si_snr_sum + si_snr_val

        # Average over sources
        avg_si_snr = si_snr_sum / num_sources  # [B]
        losses.append(avg_si_snr)

    # Stack losses: [num_perms, B]
    losses = torch.stack(losses, dim=0)

    # Find best permutation (highest SI-SNR) for each sample
    best_si_snr, _ = torch.max(losses, dim=0)  # [B]

    # Return negative mean for minimization
    return -best_si_snr.mean()


class ConvTasNetMultiScale(nn.Module):
    """
    Multi-Scale Conv-TasNet with:
    1. Multiple encoder/decoder pairs with different filter lengths
    2. Hierarchical TCN with different dilation rates at different scales
    3. Frequency-aware convolutions that adapt based on frequency content
    """
    def __init__(self,
                 num_sources=2,
                 encoder_channels=256,
                 bottleneck_channels=256,
                 hidden_channels=512,
                 encoder_kernel_sizes=[8, 16, 32, 64],  # Multi-scale encoders
                 tcn_kernel_size=3,
                 num_blocks=8,
                 num_repeats=3,
                 num_tcn_scales=3,  # Number of hierarchical TCN branches
                 use_frequency_aware=True):
        super().__init__()
        
        self.num_sources = num_sources
        self.encoder_channels = encoder_channels
        self.encoder_kernel_sizes = encoder_kernel_sizes
        self.num_scales = len(encoder_kernel_sizes)
        self.use_frequency_aware = use_frequency_aware
        
        # 1. Multi-scale encoder (captures different temporal resolutions)
        self.multi_scale_encoder = MultiScaleEncoder(
            encoder_channels=encoder_channels,
            kernel_sizes=encoder_kernel_sizes
        )
        
        # Bottleneck conv
        self.bottleneck_conv = nn.Conv1d(
            encoder_channels,
            bottleneck_channels,
            kernel_size=1
        )
        
        # 2. Hierarchical TCN (processes at different dilation scales)
        self.separation = HierarchicalTCN(
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            kernel_size=tcn_kernel_size,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
            num_scales=num_tcn_scales
        )
        
        # 3. Frequency-aware mask generation (optional)
        if use_frequency_aware:
            # Frequency-aware convolution before mask generation
            self.freq_aware_conv = FrequencyAwareConv(
                in_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                kernel_sizes=[3, 5, 7, 9],
                dilation=1
            )
        
        # Mask generation
        self.mask_generator = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bottleneck_channels, encoder_channels * self.num_scales * num_sources, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Multi-scale decoder
        self.multi_scale_decoder = MultiScaleDecoder(
            encoder_channels=encoder_channels,
            kernel_sizes=encoder_kernel_sizes
        )
    
    def forward(self, mixture):
        # mixture: [B, T] or [B, 1, T]
        if mixture.ndim == 2:
            mixture = mixture.unsqueeze(1)
        
        batch_size, _, time_steps = mixture.shape
        
        # 1. Multi-scale encoding
        fused_features, scale_features = self.multi_scale_encoder(mixture)
        # fused_features: [B, N, L]
        # scale_features: list of [B, N, L_i] at different scales
        
        # Bottleneck
        bottleneck = self.bottleneck_conv(fused_features)  # [B, B, L]
        
        # 2. Hierarchical TCN separation
        separated_features = self.separation(bottleneck)  # [B, B, L]
        
        # 3. Frequency-aware processing (optional)
        if self.use_frequency_aware:
            separated_features = self.freq_aware_conv(separated_features)  # [B, B, L]
        
        # Generate masks for all scales and sources
        masks = self.mask_generator(separated_features)  # [B, N*num_scales*num_sources, L]
        masks = masks.view(batch_size, self.num_sources, self.num_scales, 
                          self.encoder_channels, -1)  # [B, num_sources, num_scales, N, L]
        
        # Apply masks to each scale's encoded features
        all_separated_sources = []
        
        for src_idx in range(self.num_sources):
            # Get masked features for each scale
            masked_scale_features = []
            
            for scale_idx in range(self.num_scales):
                # Get the scale feature and mask
                scale_feat = scale_features[scale_idx]  # [B, N, L_i]
                scale_mask = masks[:, src_idx, scale_idx, :, :]  # [B, N, L]
                
                # Align mask to scale feature length
                if scale_mask.shape[-1] != scale_feat.shape[-1]:
                    scale_mask = F.interpolate(scale_mask, size=scale_feat.shape[-1], 
                                              mode='linear', align_corners=False)
                
                # Apply mask
                masked_feat = scale_feat * scale_mask  # [B, N, L_i]
                masked_scale_features.append(masked_feat)
            
            # Decode using multi-scale decoder
            decoded_source = self.multi_scale_decoder(masked_scale_features, time_steps)  # [B, 1, T]
            all_separated_sources.append(decoded_source.squeeze(1))  # [B, T]
        
        # Stack all sources
        separated = torch.stack(all_separated_sources, dim=1)  # [B, num_sources, T]
        
        # Ensure output matches input length
        if separated.shape[-1] > time_steps:
            separated = separated[..., :time_steps]
        elif separated.shape[-1] < time_steps:
            padding = time_steps - separated.shape[-1]
            separated = F.pad(separated, (0, padding))
        
        return separated  # [B, num_sources, T]
    
    def separate(self, mixture):
        """Convenience method for separating sources"""
        single_input = mixture.ndim == 1
        if single_input:
            mixture = mixture.unsqueeze(0)
        
        separated = self.forward(mixture)
        
        # Split into individual sources
        sources = tuple(separated[:, i, :] for i in range(self.num_sources))
        
        if single_input:
            sources = tuple(s.squeeze(0) for s in sources)
        
        return sources  # tuple of [T] or [B, T] tensors