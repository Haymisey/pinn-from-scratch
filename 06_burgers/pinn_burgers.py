# PINN — Burgers' Equation (Shock Wave Formation)
# PDE:  u_t + u*u_x = nu * u_xx
# Domain: x in [-1, 1], t in [0, 1]
# IC:   u(x, 0) = -sin(pi*x)
# BC:   u(-1, t) = u(1, t) = 0
# Viscosity: nu = 0.01/pi  (thin shock — hardest case)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Config ───────────────────────────────────────────────────
NU          = 0.01 / np.pi   # small viscosity → thin shock
N_COLLOC    = 20000          # need lots of points for nonlinear PDE
N_BC        = 500
N_IC        = 500
EPOCHS_ADAM = 15000
LR          = 1e-3


# 1. EXACT SOLUTION via Cole-Hopf transformation
#    Used only for verification — PINN never sees this
def exact_burgers(x, t, nu=NU, n_terms=100):
    """
    Exact solution via Fourier series / Cole-Hopf.
    Reference: Basdevant et al. (1986)
    """
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    if isinstance(t, torch.Tensor):
        t = t.numpy()

    u = np.zeros_like(x, dtype=float)

    # Cole-Hopf: u = -2*nu * phi_x / phi
    # phi(x,t) = sum_n a_n * exp(-n^2*pi^2*nu*t) * cos(n*pi*x)  [even terms]
    #          + sum_n b_n * exp(-n^2*pi^2*nu*t) * sin(n*pi*x)  [odd terms]
    # For IC u(x,0) = -sin(pi*x), the exact form is:

    # Numerically integrate using the heat equation Green's function
    # phi(x,t) = integral exp(-cos(pi*s)/(2*pi*nu)) * G(x-s,t) ds
    # This is the most stable approach for small nu

    n_quad = 1000
    s = np.linspace(-1, 1, n_quad)
    ds = s[1] - s[0]

    # phi0(s) = exp(-integral_0^s u(s',0) ds' / (2*nu))
    # For u(x,0) = -sin(pi*x):  integral = cos(pi*s)/pi
    phi0 = np.exp(-np.cos(np.pi * s) / (2 * np.pi * nu))

    for i in range(len(x)):
        xi, ti = x[i], t[i]
        if ti == 0:
            u[i] = -np.sin(np.pi * xi)
        else:
            # Heat kernel: G(x-s, t) = exp(-(x-s)^2 / (4*nu*t))
            G    = np.exp(-(xi - s)**2 / (4 * nu * ti))
            phi  = np.sum(phi0 * G) * ds
            # phi_x: d/dx of G
            Gx   = -(xi - s) / (2 * nu * ti) * G
            phix = np.sum(phi0 * Gx) * ds
            u[i] = -2 * nu * phix / (phi + 1e-10)

    return u


# 2. NETWORK — deeper for nonlinear problem
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),   nn.Tanh(),
            nn.Linear(64, 64),  nn.Tanh(),
            nn.Linear(64, 64),  nn.Tanh(),
            nn.Linear(64, 64),  nn.Tanh(),
            nn.Linear(64, 64),  nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Xavier initialization — important for deep networks
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# 3. PHYSICS RESIDUAL
#    u_t + u*u_x - nu*u_xx = 0
#    The u*u_x term is the nonlinearity — this is what causes shocks
def physics_residual(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    u    = model(x, t)

    u_t  = torch.autograd.grad(u,   t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]

    # ★ The nonlinear term: u * u_x
    return u_t + u * u_x - NU * u_xx


# 4. TRAINING POINTS
#    Collocation points clustered near t=1 where shock is sharpest

# Uniform collocation
x_col_u = torch.FloatTensor(N_COLLOC // 2, 1).uniform_(-1, 1)
t_col_u = torch.FloatTensor(N_COLLOC // 2, 1).uniform_(0, 1)

# Extra points near shock region (x≈0, t≈0.5-1.0) — where gradient is steep
x_col_s = torch.FloatTensor(N_COLLOC // 2, 1).uniform_(-0.2, 0.2)
t_col_s = torch.FloatTensor(N_COLLOC // 2, 1).uniform_(0.4, 1.0)

x_col = torch.cat([x_col_u, x_col_s]).to(device)
t_col = torch.cat([t_col_u, t_col_s]).to(device)

# Boundary: u=0 at x=-1 and x=1
t_bc   = torch.FloatTensor(N_BC, 1).uniform_(0, 1).to(device)
x_bc_l = -torch.ones(N_BC, 1).to(device)
x_bc_r =  torch.ones(N_BC, 1).to(device)

# Initial condition: u(x,0) = -sin(pi*x)
x_ic   = torch.FloatTensor(N_IC, 1).uniform_(-1, 1).to(device)
t_ic   = torch.zeros(N_IC, 1).to(device)
u_ic   = -torch.sin(np.pi * x_ic).to(device)


# 5. LOSS FUNCTION
def compute_loss(model):
    # PDE residual — nonlinear Burgers
    res      = physics_residual(model, x_col, t_col)
    loss_pde = torch.mean(res**2)

    # Boundary: u=0 at both ends
    loss_bc  = (torch.mean(model(x_bc_l, t_bc)**2) +
                torch.mean(model(x_bc_r, t_bc)**2))

    # Initial condition: u(x,0) = -sin(pi*x)
    loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic)**2)

    return loss_pde + loss_bc + 20.0 * loss_ic, loss_pde, loss_bc, loss_ic


# 6. PHASE 1 — ADAM
model     = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=3000, gamma=0.5)
loss_history = []

print(f"Burgers PINN | nu={NU:.6f} | N_colloc={N_COLLOC}")
print(f"Training Phase 1: Adam ({EPOCHS_ADAM} epochs)...")

for epoch in range(EPOCHS_ADAM):
    optimizer.zero_grad()
    loss, loss_pde, loss_bc, loss_ic = compute_loss(model)
    loss.backward()

    # Gradient clipping — important for nonlinear PDEs
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
              f"PDE: {loss_pde.item():.6f} | "
              f"BC: {loss_bc.item():.6f} | "
              f"IC: {loss_ic.item():.6f}")

print("Adam done.")


# 7. PHASE 2 — L-BFGS
print("\nPhase 2: L-BFGS fine-tuning...")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=0.1,
    max_iter=500, history_size=50,
    tolerance_grad=1e-9,
    line_search_fn='strong_wolfe'
)

def closure():
    optimizer_lbfgs.zero_grad()
    loss, *_ = compute_loss(model)
    loss.backward()
    return loss

for step in range(20):
    loss_val = optimizer_lbfgs.step(closure)
    if step % 5 == 0:
        print(f"  L-BFGS step {step+1:2d} | Loss: {loss_val.item():.8f}")

print("L-BFGS done.")


# 8. VISUALIZATION
print("\nGenerating plots...")
model.eval()

# Move plot points to GPU for prediction
x_plot = torch.linspace(-1, 1, 256).unsqueeze(1).to(device)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"PINN — Burgers' Equation  (ν={NU:.5f})", fontsize=14)

# --- Rows 1: Solution snapshots at 3 time slices ---
for i, t_val in enumerate([0.25, 0.50, 0.75]):
    t_plot = torch.full_like(x_plot, t_val).to(device)

    with torch.no_grad():
        u_pred = model(x_plot, t_plot).cpu().numpy() # Bring back to CPU

    # Exact solution needs CPU numpy arrays
    u_ex = exact_burgers(x_plot.cpu().squeeze().numpy(),
                         np.full(256, t_val))

    axes[0, i].plot(x_plot.cpu().numpy(), u_ex,   'b-',  lw=2.5, label='Exact')
    axes[0, i].plot(x_plot.cpu().numpy(), u_pred, 'r--', lw=2,   label='PINN')
    axes[0, i].set_title(f't = {t_val}')
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('u(x,t)')
    axes[0, i].legend()
    axes[0, i].grid(True)
    axes[0, i].set_ylim(-1.2, 1.2)

    if t_val >= 0.5:
        axes[0, i].axvline(0, color='gray', ls=':', lw=1.5, label='Shock at x≈0')
        axes[0, i].annotate('shock\nfront', xy=(0.05, 0), fontsize=9, color='gray')

# --- Row 2 left: Space-time heatmap ---
x_st = torch.linspace(-1, 1, 200)
t_st = torch.linspace(0,  1, 200)
X_st, T_st = torch.meshgrid(x_st, t_st, indexing='ij')

# Send flat grids to GPU, predict, bring back to CPU, and reshape
with torch.no_grad():
    u_st = model(X_st.reshape(-1,1).to(device),
                 T_st.reshape(-1,1).to(device)).cpu().reshape(200,200).numpy()

im = axes[1,0].contourf(T_st.numpy(), X_st.numpy(), u_st,
                         levels=100, cmap='RdBu_r')
plt.colorbar(im, ax=axes[1,0], label='u(x,t)')
axes[1,0].set_xlabel('t')
axes[1,0].set_ylabel('x')
axes[1,0].set_title('Space-Time Solution Field')
axes[1,0].axhline(0, color='white', ls='--', lw=1, alpha=0.7, label='Shock trajectory')
axes[1,0].legend(fontsize=8)

# --- Row 2 middle: Shock profile close-up at t=0.75 ---
x_zoom = torch.linspace(-0.3, 0.3, 500).unsqueeze(1).to(device)
t_zoom = torch.full_like(x_zoom, 0.75).to(device)

with torch.no_grad():
    u_zoom = model(x_zoom, t_zoom).cpu().numpy()

u_ex_zoom = exact_burgers(x_zoom.cpu().squeeze().numpy(),
                          np.full(500, 0.75))

axes[1,1].plot(x_zoom.cpu().numpy(), u_ex_zoom, 'b-',  lw=3, label='Exact')
axes[1,1].plot(x_zoom.cpu().numpy(), u_zoom,    'r--', lw=2, label='PINN')
axes[1,1].set_title('Shock Front Close-up  (t=0.75, x∈[-0.3,0.3])')
axes[1,1].set_xlabel('x')
axes[1,1].set_ylabel('u(x,t)')
axes[1,1].legend()
axes[1,1].grid(True)
axes[1,1].set_ylim(-1.1, 1.1)

# --- Row 2 right: Loss history ---
axes[1,2].semilogy(loss_history, color='darkorange', lw=1.2)
axes[1,2].set_xlabel('Epoch')
axes[1,2].set_ylabel('Loss (log scale)')
axes[1,2].set_title('Training Loss')
axes[1,2].grid(True)

plt.tight_layout()
plt.savefig('burgers_result.png', dpi=150)
plt.show()
print("Saved!")