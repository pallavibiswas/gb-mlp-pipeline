import matplotlib.pyplot as plt

# Data from your table
sigma = [0.01, 0.02, 0.03, 0.05]

# L subset
noisy_L = [0.03049, 0.05949, 0.08683, 0.13757]
denoised_L = [0.05519, 0.05506, 0.05588, 0.05859]

# kite subset
noisy_kite = [0.01597, 0.03193, 0.04787, 0.07969]
denoised_kite = [0.02579, 0.02596, 0.02596, 0.02591]

plt.figure(figsize=(8,6))

# Plot L
plt.plot(sigma, noisy_L, marker='o', label='Noisy (L)')
plt.plot(sigma, denoised_L, marker='o', linestyle='--', label='Denoised (L)')

# Plot kite
plt.plot(sigma, noisy_kite, marker='s', label='Noisy (Kite)')
plt.plot(sigma, denoised_kite, marker='s', linestyle='--', label='Denoised (Kite)')

plt.xlabel("Noise Level (σ)")
plt.ylabel("Average Displacement (Å)")
plt.title("Denoising Performance vs Noise Level")

plt.legend()
plt.grid(True)

plt.savefig("denoising_plot.png", dpi=300, bbox_inches='tight')
