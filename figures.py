import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Constants
AREA_SIZE = 50
STEP_SIZE = 5
STEPS = 30
CENTER = np.array([AREA_SIZE / 2, AREA_SIZE / 2])

# Gauss-Markov Model
def gauss_markov(steps, area_size, alpha=0.75):
    pos = np.zeros((steps, 2))
    pos[0] = CENTER
    direction = np.random.rand(2) - 0.5
    direction /= np.linalg.norm(direction)
    for i in range(1, steps):
        random_factor = np.random.rand(2) - 0.5
        direction = alpha * direction + (1 - alpha) * random_factor
        direction /= np.linalg.norm(direction)
        pos[i] = pos[i-1] + direction * STEP_SIZE
        pos[i] = np.clip(pos[i], 0, area_size)
    return pos

# Simplified SLAW Model
def slaw(steps, area_size):
    pos = np.zeros((steps, 2))
    pos[0] = CENTER
    cluster_centers = np.random.rand(3, 2) * area_size
    for i in range(1, steps):
        target = cluster_centers[np.random.choice(len(cluster_centers))]
        direction = target - pos[i-1]
        if np.linalg.norm(direction) != 0:
            direction /= np.linalg.norm(direction)
        pos[i] = pos[i-1] + direction * STEP_SIZE
        pos[i] = np.clip(pos[i], 0, area_size)
    return pos

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Gauss-Markov Plot
for _ in range(5):
    pos = gauss_markov(STEPS, AREA_SIZE)
    x, y = pos[:, 0], pos[:, 1]
    dx, dy = np.diff(x), np.diff(y)
    colors = cm.viridis(np.linspace(0, 1, len(dx)))
    axs[0].quiver(x[:-1], y[:-1], dx, dy, color=colors, scale_units='xy', angles='xy', scale=1, width=0.003)
axs[0].set_title("Gauss-Markov (From Center)")
axs[0].set_xlim(0, AREA_SIZE)
axs[0].set_ylim(0, AREA_SIZE)
axs[0].set_aspect('equal')
axs[0].grid(True)

# SLAW Plot
for _ in range(5):
    pos = slaw(STEPS, AREA_SIZE)
    x, y = pos[:, 0], pos[:, 1]
    dx, dy = np.diff(x), np.diff(y)
    colors = cm.viridis(np.linspace(0, 1, len(dx)))
    axs[1].quiver(x[:-1], y[:-1], dx, dy, color=colors, scale_units='xy', angles='xy', scale=1, width=0.003)
axs[1].set_title("SLAW (From Center)")
axs[1].set_xlim(0, AREA_SIZE)
axs[1].set_ylim(0, AREA_SIZE)
axs[1].set_aspect('equal')
axs[1].grid(True)

plt.tight_layout()
plt.show()
