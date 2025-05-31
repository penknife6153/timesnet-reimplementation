import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TimesNetConfig:
    k: int
    in_channels: int # no. of channels
    d_model: int
    num_layers:int = 1 # no. of TimesNetBlocks


# TODO: find out a way to preserve the number of channels (C)
class InceptionBlock(nn.Module):
    def __init__(self, C:int):
        super().__init__()
        self.branch1 = nn.Conv2d(C, 64, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(C, 96, kernel_size=1),
            nn.Conv2d(96, 128, kernel_size=5, padding=2) # padding to preserve size - wouldn't this depend on the 2d representation? different 2d rep would need different padding right?
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=C, out_channels=16), 
            nn.Conv2d(kernel_size=3, in_channels=16, out_channels=32)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(kernel_size=1, in_channels=C, out_channels=32)
        )
        self.proj = nn.Linear(256, C)

    def forward(self, x_2d:torch.Tensor):
        x_2d = torch.cat([
            self.branch1(x_2d),
            self.branch2(x_2d),
            self.branch3(x_2d),
            self.branch4(x_2d)],
            dim=1 # TODO: verify if channel is dim#1
        )

        return self.proj(x_2d)


def fft_period(x_1d, k=5):
    T, C = x_1d.shape # time, channel
    
    # Apply FFT and get amplitudes
    fft_result = torch.fft.rfft(x_1d, dim=0)  # FFT along time dimension
    amplitudes = torch.abs(fft_result)  # TODO: check if it's actually just the magnitude (abs value) that we want.
    
    A = torch.mean(amplitudes, dim=1)  # mean across channel dim turns shape into (T//2 + 1,)

    valid_freqs = A[1:T//2+1]  # TODO: should we exclude DC component (corresponding to frequency of zero)

    # TODO: is indices really all we want?
    top_k_values, top_k_indices = torch.topk(valid_freqs, min(k, valid_freqs)) # just in case there's less than k points
    
    # CORRECTED: Adjust indices to account for skipping DC component
    top_k_freqs = top_k_indices + 1
    
    # Calculate corresponding periods: period = T / frequency
    periods = torch.ceil(T / top_k_freqs.float()).int()
    
    return A, top_k_freqs, periods


def transform_to_2d(x_1d:torch.Tensor, k:int=5):
    T, C = x_1d.shape
    
    # Extract periods
    A, freqs, periods = fft_period(x_1d, k)
    
    x_2d_list = []
    
    for i in range(k):
        freq = freqs[i].item()
        period = periods[i].item()

        required_length = freq * period
        pad_length = required_length - T

        # zero padding at the beginning
        x_1d = torch.cat([torch.zeros(pad_length, C), x_1d], dim=0) # concatenate along time dimension
        
        # reshape to 2d
        x_2d = torch.reshape(x_1d, (period, freq, C)) # TODO: verify if this keeps the data as expected

        # Conv2d format: (batch=1, channels=C, height=period, width=freq)
        x_2d = x_2d.permute(2, 0, 1).unsqueeze(0)

        x_2d_list.append(x_2d)
    
    return A, x_2d_list, freqs, periods


def transform_to_1d(x_2d_list:torch.Tensor, original_length=None):
    x_1d_list = []

    for x_2d in x_2d_list:
        x_2d = x_2d.squeeze(0).permute(1, 2, 0)
        period, freq, C = x_2d.shape
        x_1d = x_2d.reshape(period*freq, C)
        x_1d = x_1d[:original_length] if original_length else x_1d
        x_1d_list.append(x_1d)
    
    return x_1d_list


class TimesNetBlock(nn.Module):
    def __init__(self, config:TimesNetConfig):
        super().__init__()
        self.config = config
        self.inception = InceptionBlock(config.d_model)
        # TODO: implement embedding

    def forward(self, x):
        T, C = x.shape
        A, x_2d_list, freqs, periods = transform_to_2d(x)
        x_1d_list = transform_to_1d([self.inception(x_2d) for x_2d in x_2d_list], original_length=T)
        topk_A = torch.softmax(torch.tensor([A[freq] for freq in freqs]))
        for i in range(self.config.k):
            sum += topk_A[i] @ x_1d_list[i]
        return sum