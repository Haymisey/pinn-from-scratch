# PINN — Multi-Parameter Inverse Problem

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

ALPHA_TRUE = 0.4
A_TRUE     = 0.5

N_COLLOC    = 8000
N_SENSORS   = 20
NOISE_LEVEL = 0.01
EPOCHS_ADAM = 20000
LR          = 1e-3


# 1. EXACT SOLUTION 
def exact_solution(x, t, alpha=ALPHA_TRUE, A=A_TRUE):
    decay  = torch.exp(-alpha * np.pi**2 * t)
    source = A / (alpha * np.pi**2)
    return torch.sin(np.pi * x) * (decay + source * (1 - decay))


# 2. SENSOR DATA
x_sensors = torch.rand(N_SENSORS, 1)
t_sensors = torch.rand(N_SENSORS, 1)
u_exact_s = exact_solution(x_sensors, t_sensors)
noise_scale = NOISE_LEVEL * torch.abs(u_exact_s)
u_sensors = u_exact_s + noise_scale * torch.randn_like(u_exact_s)

print("=" * 60)
print("MULTI-PARAMETER INVERSE PROBLEM")
print(f"Recovering: alpha (true={ALPHA_TRUE}) and A (true={A_TRUE})")
print(f"From {N_SENSORS} sensors with {NOISE_LEVEL*100:.0f}% noise")
print("=" * 60)

noise_pcts = []
for i in range(N_SENSORS):
    pct = abs((u_sensors[i,0]-u_exact_s[i,0]).item()) / \
          (abs(u_exact_s[i,0].item()) + 1e-8) * 100
    noise_pcts.append(pct)
    print(f"  Sensor {i+1:2d}: u_true={u_exact_s[i,0]:.4f}  "
          f"u_noisy={u_sensors[i,0]:.4f}  noise={pct:.1f}%")


# 3. NETWORK — two learnable parameters now
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.alpha = nn.Parameter(torch.tensor([0.2]))   # true = 0.4
        self.A     = nn.Parameter(torch.tensor([1.0]))   # true = 0.5

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# 4. PHYSICS RESIDUAL

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


    source = model.A * torch.sin(np.pi * x)
    return u_t - model.alpha * u_xx - source

# 5. DYNAMIC TRAINING POINTS GENERATOR
def generate_points():
    x_col = torch.rand(N_COLLOC, 1)
    t_col = torch.rand(N_COLLOC, 1)

    t_bc  = torch.rand(300, 1)
    x_bc0 = torch.zeros(300, 1)
    x_bc1 = torch.ones(300, 1)

    x_ic  = torch.rand(300, 1)
    t_ic  = torch.zeros(300, 1)
    u_ic  = torch.sin(np.pi * x_ic)   # u(x,0) = sin(pi*x)
    
    return x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic

# Generate initial points
x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic = generate_points()

# 6. RAW LOSS FUNCTION (No hardcoded weights here anymore)
def compute_loss_components(model, x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic):
    # PDE residual
    res      = physics_residual(model, x_col, t_col)
    loss_pde = torch.mean(res**2)

    # Boundary conditions
    loss_bc  = (torch.mean(model(x_bc0, t_bc)**2) +
                torch.mean(model(x_bc1, t_bc)**2))

    # Initial condition
    loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic)**2)

    # Data loss (No hardcoded 10.0 weight)
    u_pred    = model(x_sensors, t_sensors)
    residuals = (u_pred - u_sensors)**2
    weights   = torch.abs(u_sensors).detach()
    weights   = weights / weights.sum()
    loss_data = torch.sum(weights * residuals)

    return loss_pde, loss_bc, loss_ic, loss_data


# 7. PHASE 1 — ADAM WITH GRADIENT ANNEALING & DYNAMIC POINTS

model     = PINN()
optimizer = torch.optim.Adam([
    {'params': model.net.parameters(), 'lr': 1e-3},
    {'params': [model.alpha],          'lr': 5e-4}, 
    {'params': [model.A],              'lr': 2e-3}, 
], lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.5)

alpha_history, A_history, loss_history  = [], [], []

# Initialize dynamic loss weights
w_bc, w_ic, w_data = 1.0, 1.0, 10.0  # Starting guesses
alpha_ema = 0.9                      # Exponential Moving Average factor for stability

print(f"\nPHASE 1: Adam — {EPOCHS_ADAM} epochs")

for epoch in range(EPOCHS_ADAM):
    # --- DYNAMIC RESAMPLING ---
    # Resample spatial/temporal domains every 100 epochs
    if epoch % 100 == 0:
        x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic = generate_points()
    
    # --- GRADIENT ANNEALING (Wang et al.) ---
    # Update loss weights dynamically every 100 epochs to balance gradient flow
    if epoch % 100 == 0 and epoch > 0:
        optimizer.zero_grad()
        l_pde, l_bc, l_ic, l_data = compute_loss_components(model, x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic)
        
        # 1. Get max gradient of the PDE loss w.r.t the last layer weights
        l_pde.backward(retain_graph=True)
        grad_pde = model.net[-1].weight.grad.abs().max().detach()
        
        # 2. Get mean gradients of the other losses
        model.zero_grad(); l_data.backward(retain_graph=True)
        grad_data = model.net[-1].weight.grad.abs().mean().detach()
        
        model.zero_grad(); l_bc.backward(retain_graph=True)
        grad_bc = model.net[-1].weight.grad.abs().mean().detach()
        
        model.zero_grad(); l_ic.backward(retain_graph=True)
        grad_ic = model.net[-1].weight.grad.abs().mean().detach()
        
        # 3. Update weights using Exponential Moving Average
        # Formula: weight = target_grad / (current_grad + epsilon)
        w_data = alpha_ema * w_data + (1 - alpha_ema) * (grad_pde / (grad_data + 1e-8)).item()
        w_bc   = alpha_ema * w_bc   + (1 - alpha_ema) * (grad_pde / (grad_bc + 1e-8)).item()
        w_ic   = alpha_ema * w_ic   + (1 - alpha_ema) * (grad_pde / (grad_ic + 1e-8)).item()
        
        model.zero_grad() # Clean up before actual training step

    # --- STANDARD TRAINING STEP ---
    optimizer.zero_grad()
    l_pde, l_bc, l_ic, l_data = compute_loss_components(model, x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic)
    
    # Apply dynamically updated weights
    loss = l_pde + (w_bc * l_bc) + (w_ic * l_ic) + (w_data * l_data)
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Parameter Clamping
    with torch.no_grad():
        model.alpha.clamp_(0.01, 2.0)
        model.A.clamp_(-5.0, 5.0)

    # Logging
    alpha_history.append(model.alpha.item())
    A_history.append(model.A.item())
    loss_history.append(loss.item())

    if epoch % 2000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.5f} | "
              f"α={model.alpha.item():.5f} | A={model.A.item():.5f} | "
              f"w_data={w_data:.1f}")

print("\nAdam done.")

# 8. PHASE 2 — L-BFGS

print(f"\nPHASE 2: L-BFGS precision fine-tuning")

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=0.1,
    max_iter=1000,
    history_size=100,
    tolerance_grad=1e-11,
    tolerance_change=1e-13,
    line_search_fn='strong_wolfe'
)

lbfgs_alpha, lbfgs_A = [], []

def closure():
    optimizer_lbfgs.zero_grad()
    
    # 1. Pass all the dynamically generated points to the function
    l_pde, l_bc, l_ic, l_data = compute_loss_components(
        model, x_col, t_col, t_bc, x_bc0, x_bc1, x_ic, t_ic, u_ic
    )
    
    # 2. Combine the loss using the final weights learned from Adam
    loss = l_pde + (w_bc * l_bc) + (w_ic * l_ic) + (w_data * l_data)
    
    loss.backward()
    
    # 3. Parameter Clamping
    with torch.no_grad():
        model.alpha.clamp_(0.01, 2.0)
        model.A.clamp_(-5.0, 5.0)
        
    lbfgs_alpha.append(model.alpha.item())
    lbfgs_A.append(model.A.item())
    return loss

for step in range(20):
    loss_val = optimizer_lbfgs.step(closure)
    print(f"L-BFGS step {step+1:2d} | Loss: {loss_val.item():.10f} | "
          f"alpha={model.alpha.item():.8f} | A={model.A.item():.8f}")

alpha_history.extend(lbfgs_alpha)
A_history.extend(lbfgs_A)

alpha_f = model.alpha.item()
A_f     = model.A.item()
err_a   = abs(alpha_f - ALPHA_TRUE) / ALPHA_TRUE * 100
err_A   = abs(A_f - A_TRUE) / A_TRUE * 100

print(f"\n{'='*60}")
print(f"FINAL RESULT")
print(f"  alpha learned = {alpha_f:.8f}  |  true = {ALPHA_TRUE}  |  error = {err_a:.4f}%")
print(f"  A     learned = {A_f:.8f}  |  true = {A_TRUE}  |  error = {err_A:.4f}%")
print(f"{'='*60}")


# 9. PLOTTING — 4 panels
model.eval()
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-Parameter Inverse PINN\n"
             f"Recovered: α={alpha_f:.4f} (true=0.4), "
             f"A={A_f:.4f} (true=0.5)", fontsize=13)

# alpha convergence
axes[0,0].plot(alpha_history, color='crimson', lw=1.2, label='α learned')
axes[0,0].axhline(ALPHA_TRUE, color='navy', lw=2, ls='--',
                  label=f'True α={ALPHA_TRUE}')
axes[0,0].axhline(0.2, color='gray', lw=1, ls=':',
                  label='Initial guess=0.2')
axes[0,0].axvline(EPOCHS_ADAM, color='green', lw=1.5, ls='-.',
                  label='Adam → L-BFGS')
axes[0,0].set_title(f'α Convergence  (error: {err_a:.3f}%)')
axes[0,0].set_xlabel('Iteration')
axes[0,0].set_ylabel('α')
axes[0,0].legend(fontsize=8)
axes[0,0].grid(True)

# A convergence
axes[0,1].plot(A_history, color='darkorange', lw=1.2, label='A learned')
axes[0,1].axhline(A_TRUE, color='navy', lw=2, ls='--',
                  label=f'True A={A_TRUE}')
axes[0,1].axhline(1.0, color='gray', lw=1, ls=':',
                  label='Initial guess=1.0')
axes[0,1].axvline(EPOCHS_ADAM, color='green', lw=1.5, ls='-.',
                  label='Adam → L-BFGS')
axes[0,1].set_title(f'A Convergence  (error: {err_A:.3f}%)')
axes[0,1].set_xlabel('Iteration')
axes[0,1].set_ylabel('A (heat source amplitude)')
axes[0,1].legend(fontsize=8)
axes[0,1].grid(True)

# solution comparison at t=0.5
x_plot  = torch.linspace(0, 1, 300).unsqueeze(1)
t_plot  = torch.full_like(x_plot, 0.5)

with torch.no_grad():
    u_pred = model(x_plot, t_plot).numpy()

u_ex = exact_solution(x_plot, t_plot).numpy()

axes[1,0].plot(x_plot.numpy(), u_ex,   'b-',  lw=2.5, label='Exact')
axes[1,0].plot(x_plot.numpy(), u_pred, 'r--', lw=2,   label='PINN')
mask = (t_sensors.squeeze() > 0.3) & (t_sensors.squeeze() < 0.7)
if mask.sum() > 0:
    axes[1,0].scatter(x_sensors[mask].numpy(), u_sensors[mask].numpy(),
                      color='green', s=80, zorder=5,
                      marker='x', lw=2, label='Sensors near t=0.5')
axes[1,0].set_title('Solution at t=0.5')
axes[1,0].set_xlabel('x')
axes[1,0].set_ylabel('u(x, 0.5)')
axes[1,0].legend()
axes[1,0].grid(True)

# 2D heatmap of PINN solution
x_2d = torch.linspace(0, 1, 100)
t_2d = torch.linspace(0, 1, 100)
X, T = torch.meshgrid(x_2d, t_2d, indexing='ij')
x_flat = X.reshape(-1, 1)
t_flat = T.reshape(-1, 1)

with torch.no_grad():
    u_2d = model(x_flat, t_flat).reshape(100, 100).numpy()

im = axes[1,1].contourf(t_2d.numpy(), x_2d.numpy(), u_2d,
                         levels=50, cmap='hot')
plt.colorbar(im, ax=axes[1,1], label='u(x,t)')
axes[1,1].scatter(t_sensors.numpy(), x_sensors.numpy(),
                  color='cyan', s=60, zorder=5,
                  marker='x', lw=2, label='Sensor locations')
axes[1,1].set_xlabel('t')
axes[1,1].set_ylabel('x')
axes[1,1].set_title('Full Solution Field u(x,t)')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('multiparameter_result.png', dpi=150)
plt.show()
print("Saved!")