import argparse
import gradio as gr
import torch
import torchaudio
import numpy as np
from pathlib import Path
import tempfile
from src.models.conv_tasnet import ConvTasNet, ConvTasNetMultiScale
from src.models.dprnn import DPRNNSeparator
from src.utils.logger import setup_logger


def load_model(checkpoint_path, device, logger):
    """
    Load trained model from checkpoint (supports Conv-TasNet standard/multi-scale and DPRNN)
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        logger: logger instance
        
    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        # New format with full config
        config = checkpoint['config']
        model_config = config['model']
        dataset_config = config['dataset']
        run_config = config.get('run', {})
        
        # Determine model type
        model_type = run_config.get('model_type', None)
        
        # Infer model type if not specified (backward compatibility)
        if model_type is None:
            # Check for DPRNN-specific parameters
            if 'enc_dim' in model_config or 'segment_size' in model_config:
                model_type = 'dprnn'
            elif 'encoder_kernel_sizes' in model_config:
                model_type = 'multi_scale'
            else:
                model_type = 'standard'
        
        logger.info(f"Detected model type: {model_type}")
        
        if model_type == 'dprnn':
            model = DPRNNSeparator(
                num_sources=model_config.get('num_sources', dataset_config['n_src']),
                enc_dim=model_config['enc_dim'],
                feature_dim=model_config['feature_dim'],
                hidden_dim=model_config['hidden_dim'],
                layers=model_config['layers'],
                segment_size=model_config['segment_size'],
                win_len=model_config['win_len'],
                rnn_type=model_config.get('rnn_type', 'LSTM')
            )
        elif model_type == 'multi_scale':
            model = ConvTasNetMultiScale(
                num_sources=dataset_config['n_src'],
                encoder_channels=model_config['encoder_channels'],
                bottleneck_channels=model_config['bottleneck_channels'],
                hidden_channels=model_config['hidden_channels'],
                encoder_kernel_sizes=model_config['encoder_kernel_sizes'],
                tcn_kernel_size=model_config['tcn_kernel_size'],
                num_blocks=model_config['num_blocks'],
                num_repeats=model_config['num_repeats'],
                num_tcn_scales=model_config['num_tcn_scales'],
                use_frequency_aware=model_config['use_frequency_aware']
            )
        else:  # standard
            model = ConvTasNet(
                num_sources=dataset_config['n_src'],
                encoder_channels=model_config['encoder_channels'],
                bottleneck_channels=model_config['bottleneck_channels'],
                hidden_channels=model_config['hidden_channels'],
                kernel_size=model_config['kernel_size'],
                tcn_kernel_size=model_config['tcn_kernel_size'],
                num_blocks=model_config['num_blocks'],
                num_repeats=model_config['num_repeats'],
                stride=model_config['stride']
            )
    else:
        # Old format - try to infer from state dict
        logger.warning("Checkpoint doesn't contain config. Using default standard Conv-TasNet config...")
        
        # Default 2-source model (assume standard for old checkpoints)
        num_sources = 2
        
        # Create model with default standard config
        model = ConvTasNet(
            num_sources=num_sources,
            encoder_channels=256,
            bottleneck_channels=256,
            hidden_channels=512,
            kernel_size=16,
            tcn_kernel_size=3,
            num_blocks=8,
            num_repeats=3,
            stride=8
        )
    
    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    logger.info("✓ Model loaded successfully")
    return model


def preprocess_audio(audio_input, target_sr=16000, device='cpu'):
    """
    Preprocess audio for model input (matches LibriMix dataset format)
    
    Args:
        audio_input: tuple of (sample_rate, audio_data) from Gradio
        target_sr: target sample rate for the model
        device: torch device
        
    Returns:
        Preprocessed audio tensor [B, T] where B=1 (batch size)
    """
    sr, audio_data = audio_input
    
    # Convert to torch tensor
    if isinstance(audio_data, np.ndarray):
        # Handle stereo to mono conversion if needed
        if len(audio_data.shape) == 2:
            audio_data = audio_data.mean(axis=1)
        
        audio_tensor = torch.from_numpy(audio_data).float()
    else:
        audio_tensor = torch.tensor(audio_data).float()
    
    # Normalize to [-1, 1] if needed
    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)
    
    # Ensure 1D tensor [T]
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # Resample if needed (operates on 1D tensor)
    if sr != target_sr:
        # Add channel dim for resampler [1, T]
        audio_tensor = audio_tensor.unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio_tensor = resampler(audio_tensor)
        # Remove channel dim back to [T]
        audio_tensor = audio_tensor.squeeze(0)
    
    # Add batch dimension [1, T] (matches LibriMix format)
    audio_tensor = audio_tensor.unsqueeze(0)
    
    return audio_tensor.to(device), target_sr


def separate_audio(audio_input, model, device, logger, target_sr=16000):
    """
    Separate audio into sources using trained model
    
    Args:
        audio_input: Audio input from Gradio (tuple of sample_rate, audio_data)
        model: Loaded model (Conv-TasNet/Multi-Scale/DPRNN)
        device: torch device
        logger: logger instance
        target_sr: target sample rate (default: 16000 Hz)
        
    Returns:
        Tuple of (sr, source1_audio, source2_audio) for Gradio outputs
        
    Note:
        Model expects input: [B, T] and returns: [B, num_sources, T]
        (matching LibriMix dataset format)
    """
    if audio_input is None:
        return None, None
    
    try:
        # Preprocess audio: [B, T] where B=1
        audio_tensor, sr = preprocess_audio(audio_input, target_sr=target_sr, device=device)
        
        logger.info(f"Processing audio: shape={audio_tensor.shape}, sr={sr}")
        
        # Perform separation: [B, T] -> [B, num_sources, T]
        with torch.no_grad():
            separated = model(audio_tensor)
        
        logger.info(f"Separation complete: shape={separated.shape}")
        
        # Convert to numpy for Gradio
        separated_np = separated.cpu().numpy()
        
        # Extract sources: [B, num_sources, T] -> [T] for each source
        source1 = separated_np[0, 0, :]  # [T]
        source2 = separated_np[0, 1, :]  # [T]
        
        # Normalize to [-1, 1] range for better playback
        source1 = np.clip(source1 / (np.abs(source1).max() + 1e-8), -1, 1)
        source2 = np.clip(source2 / (np.abs(source2).max() + 1e-8), -1, 1)
        
        # Convert to int16 for audio output
        source1_int = (source1 * 32767).astype(np.int16)
        source2_int = (source2 * 32767).astype(np.int16)
        
        return (sr, source1_int), (sr, source2_int)
        
    except Exception as e:
        logger.error(f"Error during separation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


def create_gradio_interface(model_path, device='cpu'):
    """
    Create Gradio interface for audio source separation
    
    Args:
        model_path: Path to trained model checkpoint
        device: Device to use for inference
        
    Returns:
        Gradio interface
    """
    logger = setup_logger(__name__, level='INFO')
    
    # Load model
    logger.info("Loading model for Gradio app...")
    model = load_model(model_path, device, logger)
    logger.info("Model loaded successfully!")
    
    # Define separation function for Gradio
    def separate_fn(audio_input):
        return separate_audio(audio_input, model, device, logger)
    
    # Create Gradio interface
    with gr.Blocks(title="Audio Source Separation - Conv-TasNet") as demo:
        gr.Markdown(
            """
            # 🎵 Audio Source Separation
            
            This app separates mixed audio into 2 individual sources using deep learning models.
            
            **Supported Models** (automatically detected):
            - Conv-TasNet (standard)
            - Conv-TasNet Multi-Scale
            - DPRNN (Dual-Path RNN)
            
            **Instructions:**
            1. Record audio using your microphone OR upload an audio file
            2. Click "Separate Audio" to process
            3. Listen to the 2 separated sources below
            
            """
        )
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Input Mixed Audio",
                    format="wav"
                )
                separate_btn = gr.Button("🎯 Separate Audio", variant="primary", size="lg")
            
        with gr.Row():
            with gr.Column():
                output1 = gr.Audio(
                    label="🔊 Separated Source 1",
                    type="numpy"
                )
            with gr.Column():
                output2 = gr.Audio(
                    label="🔊 Separated Source 2",
                    type="numpy"
                )
        
        gr.Markdown(
            """
            ---
            ### About
            This application uses deep learning models for blind source separation on audio mixtures:
            - **Conv-TasNet**: Convolutional Time-domain Audio Separation Network
            - **DPRNN**: Dual-Path Recurrent Neural Network
            
            The model type is automatically detected from the checkpoint file.
            """
        )
        
        # Connect button to function
        separate_btn.click(
            fn=separate_fn,
            inputs=audio_input,
            outputs=[output1, output2]
        )
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio app for Conv-TasNet audio source separation")
    
    parser.add_argument('--model-path', default='models/conv_tasnet_medium_mixboth/conv_tasnet_medium_mixboth_20251104_ba440e/best_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'mps', 'cpu'], 
                       help='Device to use (auto-detect if not specified)')
    parser.add_argument('--share', action='store_true', help='Create a public share link')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the app on (default: 7860)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create and launch interface
    demo = create_gradio_interface(args.model_path, device=device)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()

