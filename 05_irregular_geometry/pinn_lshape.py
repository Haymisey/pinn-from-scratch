# PINN — 2D Steady-State Heat Equation on L-Shaped Domain
# PDE:  u_xx + u_yy = 0  (Laplace equation)
# Domain: L-shape = unit square MINUS top-right corner
#
# Boundary conditions:
#   Bottom edge (y=0):        u = 1.0  (hot)
#   Top edge (y=1, x<0.5):   u = 0.0  (cold)
#   Inner corner edges:       u = 0.0  (cold)
#   Left edge (x=0):          du/dn = 0 (insulated)
#   Right edge (x=0.5, y<0.5): du/dn = 0 (insulated)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize

# Tell PyTorch to use the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# Config
N_INTERIOR  = 10000   # collocation points inside domain
N_BOUNDARY  = 2000    # points on each boundary segment
EPOCHS_ADAM = 15000
EPOCHS_LBFGS= 20
LR          = 1e-3

# DOMAIN UTILITIES

def is_inside_L(x, y):
    """Returns boolean mask: True if point is inside L-domain"""
    # L = unit square minus top-right quarter
    in_unit_square  = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    in_void         = (x >= 0.5) & (y >= 0.5)  # top-right corner
    return in_unit_square & ~in_void

def sample_interior(n):
    """Sample n random points strictly inside the L-domain"""
    pts = []
    while len(pts) < n:
        x = torch.rand(n * 3)
        y = torch.rand(n * 3)
        mask = is_inside_L(x, y)
        # Exclude points too close to boundary
        mask &= (x > 0.01) & (x < 0.99) & (y > 0.01)
        mask &= ~((x > 0.49) & (y > 0.49))  # away from inner corner
        valid = torch.stack([x[mask], y[mask]], dim=1)
        pts.append(valid)
    pts = torch.cat(pts, dim=0)[:n]
    return pts[:, 0:1], pts[:, 1:2]


# BOUNDARY POINTS

def sample_boundaries(n):
    # --- DIRICHLET BOUNDARIES (Fixed Temperature) ---
    b_x, b_y, b_u = [], [], []

    # Bottom edge (HOT)
    b_x.append(torch.rand(n)); b_y.append(torch.zeros(n)); b_u.append(torch.ones(n))
    # Top edge (COLD)
    b_x.append(torch.rand(n)*0.5); b_y.append(torch.ones(n)); b_u.append(torch.zeros(n))
    # Inner horizontal edge (COLD)
    b_x.append(0.5 + torch.rand(n)*0.5); b_y.append(torch.ones(n)*0.5); b_u.append(torch.zeros(n))
    # Inner vertical edge (COLD)
    b_x.append(torch.ones(n)*0.5); b_y.append(0.5 + torch.rand(n)*0.5); b_u.append(torch.zeros(n))

    x_dir = torch.cat(b_x).unsqueeze(1)
    y_dir = torch.cat(b_y).unsqueeze(1)
    u_dir = torch.cat(b_u).unsqueeze(1)

    # --- NEUMANN BOUNDARIES (Insulated: du/dx = 0) ---
    # Left edge (x=0)
    x_neu_L = torch.zeros(n, 1)
    y_neu_L = torch.rand(n, 1)

    # Right edge (x=1, y in [0, 0.5])
    x_neu_R = torch.ones(n, 1)
    y_neu_R = torch.rand(n, 1) * 0.5

    return x_dir, y_dir, u_dir, x_neu_L, y_neu_L, x_neu_R, y_neu_R

# NETWORK
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),  nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


# PHYSICS RESIDUAL
#    Laplace: u_xx + u_yy = 0

def physics_residual(model, x, y):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    u    = model(x, y)
    u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    u_y  = torch.autograd.grad(u,   y, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y),
                                create_graph=True)[0]

    return u_xx + u_yy   # = 0 everywhere inside domain

# SAMPLE TRAINING POINTS
print("Sampling training points...")
x_int, y_int         = sample_interior(N_INTERIOR)
x_dir, y_dir, u_dir, x_neu_L, y_neu_L, x_neu_R, y_neu_R = sample_boundaries(N_BOUNDARY)
x_int, y_int = x_int.to(device), y_int.to(device)
x_dir, y_dir, u_dir = x_dir.to(device), y_dir.to(device), u_dir.to(device)
x_neu_L, y_neu_L = x_neu_L.to(device), y_neu_L.to(device)
x_neu_R, y_neu_R = x_neu_R.to(device), y_neu_R.to(device)
print(f"  Interior: {x_int.shape[0]} points")
print(f"  Dirichlet Boundary: {x_dir.shape[0]} points")
print(f"  Neumann Boundary: {x_neu_L.shape[0] + x_neu_R.shape[0]} points")

# LOSS FUNCTION
def compute_loss(model):
    # 1. PDE Loss (Interior)
    res = physics_residual(model, x_int, y_int)
    loss_pde = torch.mean(res**2)

    # 2. Dirichlet BC Loss (Fixed Temperatures)
    u_pred_dir = model(x_dir, y_dir)
    loss_dir = torch.mean((u_pred_dir - u_dir)**2)

    # 3. Neumann BC Loss (Left Edge: du/dx = 0)
    x_nL = x_neu_L.clone().requires_grad_(True) # Require grad to take derivative
    u_nL = model(x_nL, y_neu_L)
    u_x_L = torch.autograd.grad(u_nL, x_nL, grad_outputs=torch.ones_like(u_nL), create_graph=True)[0]
    loss_neu_L = torch.mean(u_x_L**2) # Minimize derivative to 0

    # 4. Neumann BC Loss (Right Edge: du/dx = 0)
    x_nR = x_neu_R.clone().requires_grad_(True)
    u_nR = model(x_nR, y_neu_R)
    u_x_R = torch.autograd.grad(u_nR, x_nR, grad_outputs=torch.ones_like(u_nR), create_graph=True)[0]
    loss_neu_R = torch.mean(u_x_R**2) # Minimize derivative to 0

    # Combine Boundary Losses
    loss_bc = loss_dir + loss_neu_L + loss_neu_R

    total_loss = loss_pde + 10.0 * loss_bc
    return total_loss, loss_pde, loss_bc

# PHASE 1 — ADAM
model     = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=3000, gamma=0.5)
loss_history = []

print(f"\nPHASE 1: Adam — {EPOCHS_ADAM} epochs")
for epoch in range(EPOCHS_ADAM):
    optimizer.zero_grad()
    loss, loss_pde, loss_bc = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
              f"PDE: {loss_pde.item():.6f} | "
              f"BC: {loss_bc.item():.6f}")

print("Adam done.")


# PHASE 2 — L-BFGS
print("\nPHASE 2: L-BFGS fine-tuning...")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=0.1, max_iter=500,
    history_size=50, tolerance_grad=1e-9,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss, _, _ = compute_loss(model)
    loss.backward()
    return loss

for step in range(EPOCHS_LBFGS):
    loss_val = optimizer_lbfgs.step(closure)
    if step % 5 == 0:
        print(f"  L-BFGS step {step+1:2d} | Loss: {loss_val.item():.8f}")

print("L-BFGS done.")


# VISUALIZATION
print("\nGenerating visualization...")
model.eval()

# Create fine grid over unit square
resolution = 300
x_lin = torch.linspace(0, 1, resolution)
y_lin = torch.linspace(0, 1, resolution)
X, Y  = torch.meshgrid(x_lin, y_lin, indexing='xy')

x_flat = X.reshape(-1, 1).to(device)
y_flat = Y.reshape(-1, 1).to(device)

# Predict temperature everywhere
with torch.no_grad():
    u_flat = model(x_flat, y_flat).squeeze().cpu()

u_grid = u_flat.reshape(resolution, resolution).numpy()
X_np   = X.numpy()
Y_np   = Y.numpy()

# Mask out the void region (top-right corner)
mask_void = (X_np >= 0.5) & (Y_np >= 0.5)
u_grid_masked = np.where(mask_void, np.nan, u_grid)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PINN on L-Shaped Domain — 2D Heat Equation", fontsize=14)

# Panel 1: Temperature heatmap
im1 = axes[0].contourf(X_np, Y_np, u_grid_masked,
                        levels=60, cmap='hot', extend='both')
plt.colorbar(im1, ax=axes[0], label='Temperature u(x,y)')
# Draw domain outline
domain_outline = plt.Polygon(
    [(0,0),(1,0),(1,0.5),(0.5,0.5),(0.5,1),(0,1),(0,0)],
    fill=False, edgecolor='white', linewidth=2
)
axes[0].add_patch(domain_outline)
axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
axes[0].set_title('Temperature Field')
axes[0].set_aspect('equal')

# Gray out the void
void_patch = Rectangle((0.5, 0.5), 0.5, 0.5,
                        facecolor='#404040', edgecolor='white', lw=2)
axes[0].add_patch(void_patch)
axes[0].text(0.75, 0.75, 'void', color='white',
             ha='center', va='center', fontsize=12)

# Panel 2: Contour lines (isotherms)
im2 = axes[1].contour(X_np, Y_np, u_grid_masked,
                       levels=15, cmap='coolwarm')
axes[1].clabel(im2, inline=True, fontsize=7, fmt='%.2f')
void_patch2 = Rectangle((0.5, 0.5), 0.5, 0.5,
                         facecolor='#404040', edgecolor='black', lw=2)
axes[1].add_patch(void_patch2)
axes[1].text(0.75, 0.75, 'void', color='white',
             ha='center', va='center', fontsize=12)
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
axes[1].set_title('Isotherms (Temperature Contours)')
axes[1].set_aspect('equal')

# Panel 3: Training loss
axes[2].semilogy(loss_history, color='darkorange', lw=1.2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss (log scale)')
axes[2].set_title('Training Loss')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('lshape_result.png', dpi=150)
plt.show()
print("Saved!")