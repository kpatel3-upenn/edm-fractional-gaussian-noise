import torch
import numpy as np


fractional_gaussian_correction_factors = {128: 4.3022, 240: 4.8834, 256: 4.9422}

def rand_fractional_gaussian(shape, generator=None, alpha=1, device=None):
    if shape[-1] not in fractional_gaussian_correction_factors:
        raise ValueError(f'Correction factor for fractional Gaussian noise not available for shape {shape}')

    device = device or 'cpu'

    noise = torch.randn(shape, generator=generator, device=device)

    f_noise = torch.fft.fftshift(torch.fft.fft2(noise), dim=[-2, -1])
    # Efficient distance calculation using broadcasting
    rows = torch.arange(shape[-2], device=device) - shape[-2] // 2
    cols = torch.arange(shape[-1], device=device) - shape[-1] // 2
    distance = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2)
    distance = distance / torch.mean(distance)
    # Avoid division by zero by setting the center value to a large number before division
    distance[distance == 0] = 1e+35
    f_noise = f_noise / (distance ** alpha)

    noise = torch.fft.ifft2(torch.fft.ifftshift(f_noise, dim=[-2, -1])).real
    correction = np.log(shape[-1] / 2) / fractional_gaussian_correction_factors[shape[-1]]
    noise = noise / np.sqrt(np.log(shape[-1] / 2)) * np.sqrt(correction)

    return noise


def rand_fractional_gaussian_like(input_tensor, generator=None, alpha=1):
    shape = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype  # Extract dtype of the input tensor

    if shape[-1] not in fractional_gaussian_correction_factors:
        raise ValueError(f'Correction factor for fractional Gaussian noise not available for shape {shape}')

    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)  # Generate noise with the same dtype

    f_noise = torch.fft.fftshift(torch.fft.fft2(noise), dim=[-2, -1])
    rows = torch.arange(shape[-2], device=device, dtype=dtype) - shape[-2] // 2
    cols = torch.arange(shape[-1], device=device, dtype=dtype) - shape[-1] // 2
    distance = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2)
    distance = distance / torch.mean(distance)
    distance[distance == 0] = torch.tensor(1e+35, device=device, dtype=dtype)  # Use dtype for the tensor
    f_noise = f_noise / (distance ** alpha)

    noise = torch.fft.ifft2(torch.fft.ifftshift(f_noise, dim=[-2, -1])).real
    correction = np.log(shape[-1] / 2) / fractional_gaussian_correction_factors[shape[-1]]
    correction_factor = torch.tensor(np.sqrt(correction), device=device, dtype=dtype)
    noise = noise / torch.sqrt(torch.log(torch.tensor(shape[-1] / 2, device=device, dtype=dtype))) * correction_factor

    return noise