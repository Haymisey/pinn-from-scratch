import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import matplotlib.gridspec as gridspec

# Let's set the kinematic viscosity of human blood (approximate)
NU = 0.003  # cm^2/s

class BrainSurgeonPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: x, y, t (3 dimensions)
        # Output: u, v, p (3 variables: x-velocity, y-velocity, pressure)
        self.net = nn.Sequential(
            nn.Linear(3, 128),  nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3) # Outputs u, v, p
        )
        
        # Initialize weights
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, y, t):
        out = self.net(torch.cat([x, y, t], dim=1))
        u = out[:, 0:1] # First column is u
        v = out[:, 1:2] # Second column is v
        p = out[:, 2:3] # Third column is pressure
        return u, v, p

def navier_stokes_residual(model, x, y, t):
    # Enable gradients for spatial and temporal derivatives
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    # Predict u, v, p
    u, v, p = model(x, y, t)

    # --- Compute First Derivatives ---
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # --- Compute Second Derivatives ---
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # --- Rule 1: Mass Conservation (Continuity) ---
    e1 = u_x + v_y

    # --- Rule 2: X-Momentum ---
    e2 = u_t + (u * u_x + v * u_y) + p_x - NU * (u_xx + u_yy)

    # --- Rule 3: Y-Momentum ---
    e3 = v_t + (u * v_x + v * v_y) + p_y - NU * (v_xx + v_yy)

    return e1, e2, e3

print("Phase 1: Physics Engine Initialized!")


print("Phase 2: Generating Virtual Patient Data...")

# 1. DEFINE THE BLOOD VESSEL GEOMETRY
# Vessel length from x=0 to x=4
# Normal height is 1.0. Aneurysm bulges up to 1.8 at x=2.0
def vessel_height(x):
    # Base height of 1.0 + a Gaussian bump for the aneurysm
    return 1.0 + 0.8 * np.exp(-4.0 * (x - 2.0)**2)

# 2. GENERATE THE FLUID FLOW (Using a Stream Function)
# This guarantees Mass Conservation (u_x + v_y = 0)
def generate_mri_data(N_points, noise_level=0.05):
    # Randomly scatter points in the bounding box
    x = np.random.uniform(0, 4, N_points * 2)
    y = np.random.uniform(0, 2, N_points * 2)
    
    # Keep only points that are INSIDE the blood vessel
    inside_mask = y <= vessel_height(x)
    x = x[inside_mask][:N_points]
    y = y[inside_mask][:N_points]
    
    # Mathematical Stream Function for flow expanding into a bulge
    H = vessel_height(x)
    eta = y / H
    U_max = 1.0
    
    # psi = Stream function. Fluid naturally follows contours of psi.
    psi = U_max * H * (2 * eta**2 - (4/3) * eta**3)
    
    # Calculate Velocity (u = d(psi)/dy, v = -d(psi)/dx)
    # Exact analytical derivatives:
    u_true = 4 * U_max * (eta - eta**2)
    
    dH_dx = -6.4 * (x - 2.0) * np.exp(-4.0 * (x - 2.0)**2)
    v_true = U_max * dH_dx * (2 * eta**2 - (4/3) * eta**3) - \
             U_max * H * (4 * eta - 4 * eta**2) * (y * dH_dx / H**2)

    # ADD MRI NOISE
    u_mri = u_true + noise_level * np.std(u_true) * np.random.randn(N_points)
    v_mri = v_true + noise_level * np.std(v_true) * np.random.randn(N_points)
    
    return x, y, u_true, v_true, u_mri, v_mri

# Generate 5,000 "MRI" sensor points
x_data, y_data, u_true, v_true, u_mri, v_mri = generate_mri_data(5000, noise_level=0.05)

# Convert to PyTorch Tensors for Phase 3
x_t = torch.tensor(x_data, dtype=torch.float32).unsqueeze(1)
y_t = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1)
u_t = torch.tensor(u_mri, dtype=torch.float32).unsqueeze(1)
v_t = torch.tensor(v_mri, dtype=torch.float32).unsqueeze(1)

# 3. VISUALIZE THE PATIENT'S "MRI SCAN"
plt.figure(figsize=(12, 4))
plt.scatter(x_data, y_data, c=u_mri, cmap='jet', s=10, alpha=0.8)
plt.colorbar(label='Horizontal Velocity (u)')

# Draw the vessel wall
x_wall = np.linspace(0, 4, 100)
plt.plot(x_wall, vessel_height(x_wall), 'k-', lw=3, label='Vessel Wall')
plt.plot(x_wall, np.zeros_like(x_wall), 'k-', lw=3)

plt.title("Patient 'MRI' Data: Noisy Blood Velocity entering an Aneurysm")
plt.xlabel("Vessel Length (x)")
plt.ylabel("Vessel Height (y)")
plt.legend()
plt.tight_layout()
plt.savefig('virtual_patient.png', dpi=150)
plt.show()

print(f"Generated {len(x_data)} data points.")
print("Phase 2 Complete!")


print("Phase 3: Initializing Brain Surgeon PINN...")

# --- 1. GPU SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move Patient Data to GPU and enable gradients for physics calculations
x_train = x_t.to(device).requires_grad_(True)
y_train = y_t.to(device).requires_grad_(True)
u_train = u_t.to(device)
v_train = v_t.to(device)

# --- 2. BOUNDARY CONDITIONS SETUP ---
# Wall Boundary (No-Slip: u=0, v=0)
N_wall = 1000
x_wall_pts = np.random.uniform(0, 4, N_wall)
y_wall_top = vessel_height(x_wall_pts)
y_wall_bot = np.zeros(N_wall)

x_wall = torch.tensor(np.concatenate([x_wall_pts, x_wall_pts]), dtype=torch.float32).unsqueeze(1).to(device)
y_wall = torch.tensor(np.concatenate([y_wall_top, y_wall_bot]), dtype=torch.float32).unsqueeze(1).to(device)

# Outlet Boundary (Pressure Anchor: p=0 at x=4)
N_out = 200
x_out = torch.ones(N_out, 1, dtype=torch.float32).to(device) * 4.0
y_out = torch.tensor(np.random.uniform(0, 1, N_out), dtype=torch.float32).unsqueeze(1).to(device)


# --- 3. THE 2D STEADY-STATE NETWORK ---
class BrainSurgeonPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: x, y (2D steady state)
        # Output: u, v, p (Velocity and Pressure)
        self.net = nn.Sequential(
            nn.Linear(2, 128),  nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 3) 
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x, y):
        out = self.net(torch.cat([x, y], dim=1))
        return out[:, 0:1], out[:, 1:2], out[:, 2:3] # u, v, p

model = BrainSurgeonPINN().to(device)
NU = 0.003 # Blood kinematic viscosity

# --- 4. THE PHYSICS RESIDUAL ---
def physics_residual(model, x, y):
    u, v, p = model(x, y)

    # First Derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # Second Derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # Navier-Stokes Equations (Steady State)
    e1 = u_x + v_y                                       # Mass Conservation
    e2 = (u * u_x + v * u_y) + p_x - NU * (u_xx + u_yy)  # X-Momentum
    e3 = (u * v_x + v * v_y) + p_y - NU * (v_xx + v_yy)  # Y-Momentum

    return e1, e2, e3

# --- 5. THE LOSS FUNCTION ---
def compute_loss():
    # 1. Data Loss (Does the PINN match the MRI scans?)
    u_pred, v_pred, _ = model(x_train, y_train)
    loss_data = torch.mean((u_pred - u_train)**2) + torch.mean((v_pred - v_train)**2)

    # 2. Physics Loss (Does the PINN obey Navier-Stokes?)
    e1, e2, e3 = physics_residual(model, x_train, y_train)
    loss_pde = torch.mean(e1**2) + torch.mean(e2**2) + torch.mean(e3**2)

    # 3. Boundary Loss (No-Slip on Walls)
    u_w, v_w, _ = model(x_wall, y_wall)
    loss_wall = torch.mean(u_w**2) + torch.mean(v_w**2)

    # 4. Pressure Anchor (p=0 at outlet)
    _, _, p_out = model(x_out, y_out)
    loss_p = torch.mean(p_out**2)

    # Total loss weighting
    total_loss = 10.0 * loss_data + loss_pde + 2.0 * loss_wall + loss_p
    return total_loss, loss_data, loss_pde


# --- 6. TRAINING LOOP ---
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

EPOCHS = 10000
print(f"Beginning Surgery (Training for {EPOCHS} epochs)...")

for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss, l_data, l_pde = compute_loss()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Total: {loss.item():.5f} | MRI Error: {l_data.item():.5f} | Physics Error: {l_pde.item():.5f}")

print("Phase 3 Complete: AI Surgery Finished!")

print("Phase 4: Extracting Clinical Metrics and Plotting...")

# 1. CREATE A HIGH-RESOLUTION MEDICAL GRID
X_res, Y_res = 400, 200
x_lin = np.linspace(0, 4, X_res)
y_lin = np.linspace(0, 1.8, Y_res)
X_grid, Y_grid = np.meshgrid(x_lin, y_lin)

x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)
y_flat = torch.tensor(Y_grid.flatten(), dtype=torch.float32).unsqueeze(1).to(device)

# 2. PREDICT THE FLUID DYNAMICS
model.eval()
with torch.no_grad():
    u_pred, v_pred, p_pred = model(x_flat, y_flat)
    
u_plot = u_pred.cpu().numpy().reshape(Y_res, X_res)
p_plot = p_pred.cpu().numpy().reshape(Y_res, X_res)

# Mask out the areas outside the blood vessel
mask = Y_grid > vessel_height(X_grid)
u_plot[mask] = np.nan
p_plot[mask] = np.nan

# 3. CALCULATE WALL SHEAR STRESS (WSS) AT THE BOTTOM WALL
x_bot = torch.linspace(0, 4, 200).unsqueeze(1).to(device).requires_grad_(True)
y_bot = torch.zeros_like(x_bot).requires_grad_(True)

u_bot, _, _ = model(x_bot, y_bot)
du_dy = torch.autograd.grad(u_bot, y_bot, grad_outputs=torch.ones_like(u_bot))[0]
wss = NU * torch.abs(du_dy).cpu().detach().numpy() 

# 4. PUBLICATION-QUALITY VISUALIZATIONS
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.8])
fig.suptitle("PINN Aneurysm Analysis: From Noisy MRI to Hidden Physics", fontsize=16, fontweight='bold')

# --- FIX: Create a clean CPU array for drawing the black walls ---
x_wall_plot = np.linspace(0, 4, 100)

# Panel 1: The Cleaned Velocity
ax1 = fig.add_subplot(gs[0])
im1 = ax1.contourf(X_grid, Y_grid, u_plot, levels=50, cmap='jet')
ax1.plot(x_wall_plot, vessel_height(x_wall_plot), 'k-', lw=3)
ax1.plot(x_wall_plot, np.zeros_like(x_wall_plot), 'k-', lw=3)
ax1.set_title("1. PINN Filtered Blood Velocity (Noise Removed)", fontsize=12)
plt.colorbar(im1, ax=ax1, label='Velocity (u)')
ax1.set_xlim(0, 4); ax1.set_ylim(0, 1.85)

# Panel 2: The Discovered Pressure Field
ax2 = fig.add_subplot(gs[1])
im2 = ax2.contourf(X_grid, Y_grid, p_plot, levels=50, cmap='inferno')
ax2.plot(x_wall_plot, vessel_height(x_wall_plot), 'k-', lw=3)
ax2.plot(x_wall_plot, np.zeros_like(x_wall_plot), 'k-', lw=3)
ax2.set_title("2. Secret Pressure Field Discovered by PINN (Never seen in training data!)", fontsize=12)
plt.colorbar(im2, ax=ax2, label='Relative Pressure (p)')
ax2.set_xlim(0, 4);

# Panel 3: Wall Shear Stress (Clinical Risk Metric)
ax3 = fig.add_subplot(gs[2])
ax3.plot(x_bot.cpu().detach().numpy(), wss, 'r-', lw=3)
ax3.fill_between(x_bot.cpu().detach().numpy().flatten(), 0, wss.flatten(), color='red', alpha=0.3)
ax3.set_title("3. Bottom Wall Shear Stress (Tearing force on the artery)", fontsize=12)
ax3.set_xlabel("Vessel Length (x)")
ax3.set_ylabel("Shear Stress")
ax3.set_xlim(0, 4)
ax3.grid(True)

plt.tight_layout()
plt.savefig('aneurysm_discovery.png', dpi=200)
plt.show()

print("Saved!")