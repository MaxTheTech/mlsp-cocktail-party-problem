import torch

def si_snr_loss(estimated, target, eps=1e-8):
    """
    Calculate Scale-Invariant SNR per sample and source

    Args:
        estimated: [batch, n_src, time]
        target: [batch, n_src, time]

    Returns:
        si_snr: [batch, n_src] - SI-SNR in dB for each sample and source
    """
    # normalize
    target = target - target.mean(dim=-1, keepdim=True)
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)

    # compute SI-SNR
    s_target = (torch.sum(estimated * target, dim=-1, keepdim=True) /
                (torch.sum(target ** 2, dim=-1, keepdim=True) + eps)) * target

    e_noise = estimated - s_target

    si_snr = 10 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )

    return si_snr