# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
@persistence.persistent_class
class EDMLoss_fractional_gaussian:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, alpha=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.alpha = alpha

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = rand_fractional_gaussian([images.shape[0], 1, 1, 1], alpha=alpha, device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = rand_fractional_gaussian_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

# Correction factors empirically determined for fractional Gaussian noise
fractional_gaussian_correction_factors = { 240: 4.8834 }
def rand_fractional_gaussian(shape, alpha=1, device=None):
    if shape[-1] not in fractional_gaussian_correction_factors:
        raise ValueError(f'Correction factor for fractional Gaussian noise not available for shape {shape}')

    device = device or 'cpu'

    noise = torch.randn(shape, device=device)

    f_noise = torch.fft.fftshift(torch.fft.fft2(noise), dim=[-2, -1])
    # Efficient distance calculation using broadcasting
    rows = torch.arange(shape[-2], device=device) - shape[-2] // 2
    cols = torch.arange(shape[-1], device=device) - shape[-1] // 2
    distance = torch.sqrt(rows[:, None] ** 2 + cols[None, :] ** 2)
    distance = distance / torch.mean(distance)
    # Avoid division by zero by setting the center value to a large number before division
    distance[distance == 0] = 1e+35
    f_noise = f_noise / (distance**alpha)

    noise = torch.fft.ifft2(torch.fft.ifftshift(f_noise, dim=[-2, -1])).real
    correction = np.log(shape[-1] / 2) / fractional_gaussian_correction_factors[shape[-1]]
    noise = noise / np.sqrt(np.log(shape[-1]/2)) * np.sqrt(correction)

    return noise


def randn_fractional_gaussian_like(input_tensor, alpha=1):
    shape = input_tensor.shape
    device = input_tensor.device
    dtype = input_tensor.dtype  # Extract dtype of the input tensor

    if shape[-1] not in fractional_gaussian_correction_factors:
        raise ValueError(f'Correction factor for fractional Gaussian noise not available for shape {shape}')

    noise = torch.randn(shape, device=device, dtype=dtype)  # Generate noise with the same dtype

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