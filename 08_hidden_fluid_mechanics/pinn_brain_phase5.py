import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import os
os.makedirs('data', exist_ok=True)
# -------------------------------------------------------
# Phase 5a: Generate Pulsatile (time‑dependent) Data
# -------------------------------------------------------

def vessel_height(x):
    """Aneurysm vessel height function."""
    return 1.0 + 0.8 * np.exp(-4.0 * (x - 2.0) ** 2)

def generate_pulsatile_data(N_points, noise_level=0.05):
    # Sample in space (x, y) and time t ∈ [0, 1] s
    x = np.random.uniform(0, 4, N_points * 3)
    y = np.random.uniform(0, 2, N_points * 3)
    t = np.random.uniform(0, 1, N_points * 3)
    inside = y <= vessel_height(x)
    x = x[inside][:N_points]
    y = y[inside][:N_points]
    t = t[inside][:N_points]
    # Heartbeat factor (sinusoidal pulse)
    heartbeat = 1.0 + 0.5 * np.sin(2 * np.pi * t)
    H = vessel_height(x)
    eta = y / H
    u_true = 4 * heartbeat * (eta - eta ** 2)
    u_mri = u_true + noise_level * np.std(u_true) * np.random.randn(N_points)
    v_mri = 0.05 * np.random.randn(N_points)  # small transversal component
    return x, y, t, u_mri, v_mri

print("Phase 5a: Generating pulsatile dataset (30,000 points)…")
x_p, y_p, t_p, u_p, v_p = generate_pulsatile_data(30000)
# Save as CSV for downstream use
pulsatile_df = pd.DataFrame({"x": x_p, "y": y_p, "t": t_p, "u": u_p, "v": v_p})
pulsatile_df.to_csv('data/pulsatile_data.csv', index=False)
print("Pulsatile CSV saved to data/pulsatile_data.csv")

# -------------------------------------------------------
# Phase 5b: Time‑Dependent 3‑D PINN (x, y, t) → (u, v, p)
# -------------------------------------------------------

class PulsatileBrainPINN(nn.Module):
    """3‑D PINN that learns the unsteady Navier‑Stokes dynamics."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3)  # u, v, p
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, y, t):
        out = self.net(torch.cat([x, y, t], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3]

def physics_residual_time(model, x, y, t, NU=0.003):
    # Enable gradients
    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)
    u, v, p = model(x, y, t)
    # First‑order derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    # Second‑order derivatives
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]
    # Navier‑Stokes residuals
    e1 = u_x + v_y
    e2 = u_t + (u * u_x + v * u_y) + p_x - NU * (u_xx + u_yy)
    e3 = v_t + (u * v_x + v * v_y) + p_y - NU * (v_xx + v_yy)
    return e1, e2, e3

# -------------------------------------------------------
# Phase 5b: Training the Unsteady PINN
# -------------------------------------------------------
print("Phase 5b: Loading pulsatile data for training…")
train_df = pd.read_csv('data/pulsatile_data.csv')
X = torch.tensor(train_df['x'].values, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(train_df['y'].values, dtype=torch.float32).unsqueeze(1)
T = torch.tensor(train_df['t'].values, dtype=torch.float32).unsqueeze(1)
U_obs = torch.tensor(train_df['u'].values, dtype=torch.float32).unsqueeze(1)
V_obs = torch.tensor(train_df['v'].values, dtype=torch.float32).unsqueeze(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
X, Y, T, U_obs, V_obs = X.to(device), Y.to(device), T.to(device), U_obs.to(device), V_obs.to(device)

model = PulsatileBrainPINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
EPOCHS = 20000
print(f"Starting unsteady training for {EPOCHS} epochs (≈15‑20 min on a T4 GPU)…")

for epoch in range(EPOCHS + 1):
    optimizer.zero_grad()
    u_pred, v_pred, _ = model(X, Y, T)
    data_loss = torch.mean((u_pred - U_obs) ** 2) + torch.mean((v_pred - V_obs) ** 2)
    e1, e2, e3 = physics_residual_time(model, X, Y, T)
    pde_loss = torch.mean(e1 ** 2) + torch.mean(e2 ** 2) + torch.mean(e3 ** 2)
    loss = 10.0 * data_loss + pde_loss
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"[Unsteady] Epoch {epoch:5d} | Total Loss {loss.item():.5f} | PDE {pde_loss.item():.5f}")

print("Phase 5 Training Complete! Unsteady PINN ready for inference.")

# Optional sanity‑check snapshot (t=0.5) saved to CSV
with torch.no_grad():
    t_snapshot = torch.full_like(X, 0.5)
    u_snap, v_snap, _ = model(X, Y, t_snapshot)
    snap_df = pd.DataFrame({
        "x": X.cpu().numpy().flatten()[:100],
        "y": Y.cpu().numpy().flatten()[:100],
        "t": t_snapshot.cpu().numpy().flatten()[:100],
        "u_pred": u_snap.cpu().numpy().flatten()[:100],
        "v_pred": v_snap.cpu().numpy().flatten()[:100]
    })
    snap_df.to_csv('data/pulsatile_snapshot.csv', index=False)
    print("Snapshot saved to data/pulsatile_snapshot.csv")

print("All phases (5a‑5b) executed successfully.")

print("Phase 5c: Generating Heartbeat Animation (Cine-Loop)...")

model.eval()

# 1. CREATE THE SPATIAL GRID
X_res, Y_res = 300, 150
x_lin = np.linspace(0, 4, X_res)
y_lin = np.linspace(0, 1.8, Y_res)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
y_flat = torch.tensor(Y_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

# Masking logic
x_wall_plot = np.linspace(0, 4, 100)
y_wall_plot = vessel_height(x_wall_plot)
mask = Y_grid > vessel_height(X_grid)

# 2. SET UP THE FIGURE
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("4D Flow MRI Digital Twin: Pulsatile Aneurysm", fontsize=16, fontweight='bold')

# 3. ANIMATION UPDATE FUNCTION
num_frames = 30
time_steps = np.linspace(0, 1, num_frames)

def update(frame):
    ax1.clear()
    ax2.clear()
    
    t_val = time_steps[frame]
    t_flat = torch.full_like(x_flat, t_val)
    
    with torch.no_grad():
        u_pred, v_pred, p_pred = model(x_flat, y_flat, t_flat)
        
    u_plot = u_pred.cpu().numpy().reshape(Y_res, X_res)
    p_plot = p_pred.cpu().numpy().reshape(Y_res, X_res)
    
    u_plot[mask] = np.nan
    p_plot[mask] = np.nan
    
    # Panel 1: Velocity
    # Fixed vmin/vmax so the colors don't flicker as the pulse hits
    ax1.contourf(X_grid, Y_grid, u_plot, levels=50, cmap='jet', vmin=0, vmax=1.5)
    ax1.plot(x_wall_plot, y_wall_plot, 'k-', lw=3)
    ax1.plot(x_wall_plot, np.zeros_like(x_wall_plot), 'k-', lw=3)
    ax1.set_title(f"1. Blood Velocity | t = {t_val:.2f} s")
    ax1.set_xlim(0, 4); ax1.set_ylim(0, 1.85)
    ax1.axis('off')
    
    # Panel 2: Pressure
    ax2.contourf(X_grid, Y_grid, p_plot, levels=50, cmap='inferno')
    ax2.plot(x_wall_plot, y_wall_plot, 'k-', lw=3)
    ax2.plot(x_wall_plot, np.zeros_like(x_wall_plot), 'k-', lw=3)
    ax2.set_title(f"2. Hidden Pressure Field | t = {t_val:.2f} s")
    ax2.set_xlim(0, 4); ax2.set_ylim(0, 1.85)
    ax2.axis('off')

print("Rendering frames (this may take 1-2 minutes)...")
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)

# Save as GIF
gif_path = 'data/aneurysm_heartbeat.gif'
ani.save(gif_path, writer='pillow', fps=10)
plt.close()

print(f"Saved to: {gif_path}")
