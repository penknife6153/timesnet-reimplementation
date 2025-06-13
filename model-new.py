import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TimesNetConfig:
    k: int  # Number of top periods to select
    in_channels: int  # Number of input channels/variates
    d_model: int  # Hidden dimension of the model
    num_layers: int = 1  # Number of TimesNetBlocks


class InceptionBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        # Branches of the Inception module
        # Output channels for branches are chosen to be somewhat standard,
        # then projected back to d_model.
        # The paper mentions "parameter-efficient inception block".
        # These channel sizes are illustrative.
        
        # Branch 1: 1x1 Convolution
        self.branch1 = nn.Conv2d(d_model, 64, kernel_size=1)

        # Branch 2: 1x1 Convolution followed by 5x5 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv2d(d_model, 96, kernel_size=1),
            nn.Conv2d(96, 128, kernel_size=5, padding=2)  # padding=2 for kernel_size=5 to preserve H, W
        )
        
        # Branch 3: 1x1 Convolution followed by 3x3 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv2d(d_model, 16, kernel_size=1), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1)  # padding=1 for kernel_size=3 to preserve H, W
        )

        # Branch 4: 3x3 Max Pooling followed by 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1), # padding=1, stride=1 to preserve H, W
            nn.Conv2d(d_model, 32, kernel_size=1)
        )
        
        # Projection layer to bring concatenated channels (64+128+32+32=256) back to d_model
        self.proj = nn.Conv2d(256, d_model, kernel_size=1)

    def forward(self, x_2d:torch.Tensor): # x_2d shape: (Batch, d_model, Period, Freq)
        # Apply branches
        b1 = self.branch1(x_2d)
        b2 = self.branch2(x_2d)
        b3 = self.branch3(x_2d)
        b4 = self.branch4(x_2d)
        
        # Concatenate along the channel dimension (dim=1)
        x_concat = torch.cat([b1, b2, b3, b4], dim=1) # (Batch, 256, Period, Freq)
        
        # Project back to d_model channels
        x_projected = self.proj(x_concat) # (Batch, d_model, Period, Freq)
        
        return x_projected


def fft_period(x_1d: torch.Tensor, k: int = 5):
    # x_1d shape: (Time, Channels) or (Batch, Time, Channels)
    # Assuming (Time, Channels) for now as per original code structure
    if x_1d.ndim == 3: # (Batch, Time, Channels)
        T = x_1d.shape[1]
        fft_dim = 1 # FFT along time dimension for each batch item
        mean_dim = 2 # Mean across channel dim for each batch item
    elif x_1d.ndim == 2: # (Time, Channels)
        T = x_1d.shape[0]
        fft_dim = 0
        mean_dim = 1
    else:
        raise ValueError(f"Input x_1d has unsupported ndim: {x_1d.ndim}")

    # Apply FFT and get amplitudes
    fft_result = torch.fft.rfft(x_1d, dim=fft_dim)
    amplitudes_complex = fft_result
    amplitudes_mag = torch.abs(amplitudes_complex)
    
    # Average amplitudes across channel dimension
    A_mean_over_channels = torch.mean(amplitudes_mag, dim=mean_dim)  # Shape: (T//2 + 1,) or (Batch, T//2+1)

    # Exclude DC component (frequency 0)
    # rfft output length is T//2 + 1. Index 0 is DC, 1 to T//2 are positive frequencies.
    valid_freq_amplitudes = A_mean_over_channels[..., 1:] # Shape: (T//2,) or (Batch, T//2)
    
    if valid_freq_amplitudes.shape[-1] == 0: # Handle cases with very short T
        # Fallback: return dummy values or handle as error
        # For now, returning empty/zero tensors of expected shapes
        num_to_select = 0
        if x_1d.ndim == 3: # Batch
            return A_mean_over_channels, torch.empty((x_1d.shape[0],0), device=x_1d.device, dtype=torch.long), \
                   torch.empty((x_1d.shape[0],0), device=x_1d.device, dtype=torch.long), \
                   torch.empty((x_1d.shape[0],0), device=x_1d.device, dtype=x_1d.dtype)
        else: # Single
             return A_mean_over_channels, torch.empty((0,), device=x_1d.device, dtype=torch.long), \
                   torch.empty((0,), device=x_1d.device, dtype=torch.long), \
                   torch.empty((0,), device=x_1d.device, dtype=x_1d.dtype)


    num_to_select = min(k, valid_freq_amplitudes.shape[-1])
    
    # Get top-k amplitude values and their indices
    top_k_amp_values, top_k_indices_in_valid = torch.topk(valid_freq_amplitudes, num_to_select, dim=-1)
    
    # Adjust indices to correspond to original FFT bin indices (add 1 because DC was excluded)
    top_k_fft_bins = top_k_indices_in_valid + 1
    
    # Calculate corresponding periods: period = T / frequency_bin_index
    # (Frequency = fft_bin_index / T, so Period = T / fft_bin_index)
    periods = torch.ceil(T / top_k_fft_bins.float()).long()
    
    return A_mean_over_channels, top_k_fft_bins, periods, top_k_amp_values


def transform_to_2d(x_1d: torch.Tensor, k: int = 5):
    # x_1d shape: (Time, Channels) - d_model is the channel dim here
    T, C = x_1d.shape
    
    # Extract periods and corresponding amplitudes
    A_full_spectrum, freqs_bins, periods, top_k_amplitudes = fft_period(x_1d, k)
    
    x_2d_list = []
    
    # Iterate over the selected frequencies/periods
    # len(freqs_bins) might be less than k
    for i in range(len(freqs_bins)): 
        freq_bin = freqs_bins[i].item()
        period = periods[i].item()

        # Length of sequence after padding for reshaping
        required_length = freq_bin * period # This is T' in paper, freq_bin is f_i, period is p_i
        
        # Pad if necessary
        if T < required_length:
            pad_length = required_length - T
            # Zero padding at the beginning (as in original code)
            # Ensure zeros tensor is on the same device and dtype
            padding_tensor = torch.zeros(pad_length, C, device=x_1d.device, dtype=x_1d.dtype)
            x_1d_padded = torch.cat([padding_tensor, x_1d], dim=0)
        else:
            x_1d_padded = x_1d[:required_length] # Truncate if T > required_length (unlikely if freq_bin*period is based on T/freq_bin)


        # Reshape to 2D: (Period, Freq_bin, Channels)
        # The paper uses (pi, fi, C) which is (period, freq_bin, C)
        x_reshaped_2d = x_1d_padded.reshape(period, freq_bin, C)

        # Permute for Conv2d: (Batch=1, Channels=C, Height=Period, Width=Freq_bin)
        x_conv2d_format = x_reshaped_2d.permute(2, 0, 1).unsqueeze(0)

        x_2d_list.append(x_conv2d_format)
    
    return A_full_spectrum, x_2d_list, freqs_bins, periods, top_k_amplitudes


def transform_to_1d(x_2d_list: list[torch.Tensor], original_length: int):
    # x_2d_list contains tensors of shape (1, Channels, Period, Freq_bin)
    x_1d_list = []

    for x_2d in x_2d_list:
        # Squeeze batch, permute back to (Period, Freq_bin, Channels)
        x_permuted = x_2d.squeeze(0).permute(1, 2, 0)
        period, freq_bin, C = x_permuted.shape
        
        # Reshape to 1D: (Period*Freq_bin, Channels)
        x_1d_flat = x_permuted.reshape(period * freq_bin, C)
        
        # Truncate to original length T
        x_1d_truncated = x_1d_flat[:original_length]
        x_1d_list.append(x_1d_truncated)
    
    return x_1d_list


class TimesNetBlock(nn.Module):
    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.config = config
        self.inception = InceptionBlock(config.d_model) # Inception block operates on d_model channels

    def forward(self, x: torch.Tensor): # x shape: (Time, d_model)
        T, C = x.shape # C here is d_model

        # Transform to 2D representations for different periods
        # A_full_spectrum contains mean amplitudes over channels for all frequencies
        # top_k_amplitudes contains the specific amplitude values for the selected k frequencies
        _, x_2d_list, _, _, top_k_amplitudes = transform_to_2d(x, k=self.config.k)
        
        if not x_2d_list:
            # If no periods were found (e.g., k=0 or very short T),
            # return zeros to maintain structure for residual connection.
            return torch.zeros_like(x)

        # Process each 2D representation with the Inception block
        # Output of inception block is (1, d_model, Period, Freq_bin)
        processed_x_2d_list = [self.inception(x_2d) for x_2d in x_2d_list]
        
        # Transform processed 2D representations back to 1D
        processed_x_1d_list = transform_to_1d(processed_x_2d_list, original_length=T)

        # Adaptive aggregation using softmax of amplitudes
        # Ensure top_k_amplitudes is on the same device and dtype for softmax
        weights = torch.softmax(top_k_amplitudes.to(device=x.device, dtype=x.dtype), dim=-1)
        
        # Initialize sum_output with the correct shape and device
        sum_output = torch.zeros_like(processed_x_1d_list[0]) # (T, d_model)
        
        for i in range(len(processed_x_1d_list)):
            sum_output = sum_output + weights[i] * processed_x_1d_list[i]
            
        return sum_output


class TimesNet(nn.Module):
    def __init__(self, config: TimesNetConfig):
        super().__init__()
        self.config = config

        # Embedding layer to project in_channels to d_model
        if config.in_channels == config.d_model:
            self.embedding = nn.Identity()
        else:
            # Using Conv1d for embedding: (Batch, in_channels, Time) -> (Batch, d_model, Time)
            self.embedding = nn.Conv1d(config.in_channels, config.d_model, kernel_size=1)
            # Alternative: nn.Linear applied on the last dimension if input is (Batch, Time, in_channels)
            # self.embedding = nn.Linear(config.in_channels, config.d_model)


        self.blocks = nn.ModuleList()
        for _ in range(config.num_layers):
            self.blocks.append(TimesNetBlock(config)) # TimesNetBlock expects d_model features
        
        # Depending on the task, a final projection/head might be needed
        # e.g., self.projection = nn.Linear(config.d_model, num_output_features_or_steps)

    def forward(self, x_input: torch.Tensor):
        # x_input expected shape: (Batch, Time, in_channels)
        
        if x_input.ndim != 3 or x_input.shape[2] != self.config.in_channels:
            raise ValueError(f"Input tensor shape mismatch. Expected (Batch, Time, {self.config.in_channels}), got {x_input.shape}")

        # Apply embedding
        if isinstance(self.embedding, nn.Conv1d):
            # Permute for Conv1d: (Batch, Time, in_channels) -> (Batch, in_channels, Time)
            x_permuted = x_input.permute(0, 2, 1)
            x_embedded = self.embedding(x_permuted) # (Batch, d_model, Time)
            # Permute back: (Batch, d_model, Time) -> (Batch, Time, d_model)
            x_processed = x_embedded.permute(0, 2, 1)
        else: # nn.Identity or nn.Linear
            x_processed = self.embedding(x_input) # (Batch, Time, d_model)

        # Apply TimesNetBlocks layer by layer with residual connections
        # The current TimesNetBlock expects (Time, d_model) input.
        # We will iterate over the batch dimension.
        # For more efficiency, TimesNetBlock and sub-modules could be modified
        # to handle the batch dimension directly in tensor operations.
        
        batch_outputs = []
        for i in range(x_processed.shape[0]): # Iterate over batch
            item_features = x_processed[i] # (Time, d_model)
            
            for block in self.blocks:
                # Output of block is (Time, d_model)
                block_output = block(item_features)
                # Residual connection (Eq 4 in paper: Xl_1D = TimesBlock(Xl-1_1D) + Xl-1_1D)
                item_features = item_features + block_output 
            batch_outputs.append(item_features)
        
        # Stack batch results
        final_output = torch.stack(batch_outputs, dim=0) # (Batch, Time, d_model)

        # Apply final projection if defined for a specific task
        # if hasattr(self, 'projection') and self.projection is not None:
        #     final_output = self.projection(final_output)
            
        return final_output
