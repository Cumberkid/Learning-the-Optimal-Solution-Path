import torch
import numpy as np

def wigner_semicircle_sample(radius, num_samples):
    # Generate uniform samples within the interval [-R, R]
    x = torch.linspace(-radius, radius, num_samples)
    
    # Compute the probability density of the Wigner semicircle
    density = (2 / (np.pi * radius ** 2)) * torch.sqrt(radius ** 2 - x ** 2)
    
    # Normalize the density
    density /= density.sum()
    
    # Sample from the distribution using the computed densities
    samples = torch.multinomial(density, num_samples, replacement=True)
    
    # Return the sampled x values
    return x[samples]
