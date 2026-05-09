#PINN for 1D INVERSE HEAT EQUATION problem
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

ALPHA_TRUE  = 0.4
N_COLLOC    = 5000
N_SENSORS   = 15      # more sensors to average out noise
NOISE_LEVEL = 0.01    # 1% noise — more realistic
EPOCHS_ADAM = 8000
LR          = 1e-3


x_sensors = torch.rand(N_SENSORS, 1)
t_sensors = torch.rand(N_SENSORS, 1)

u_exact_sensors = (torch.sin(np.pi * x_sensors) *
                   torch.exp(-ALPHA_TRUE * np.pi**2 * t_sensors))

noise     = NOISE_LEVEL * torch.randn_like(u_exact_sensors)
u_sensors = u_exact_sensors + noise

print("Sensor noise levels:")
for i in range(N_SENSORS):
    noise_pct = abs(noise[i,0].item()) / (abs(u_exact_sensors[i,0].item()) + 1e-8) * 100
    print(f"  Sensor {i+1:2d}: u_true={u_exact_sensors[i,0]:.4f}  "
          f"u_noisy={u_sensors[i,0]:.4f}  noise={noise_pct:.1f}%")


#NETWORK
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Start guess at 0.2 — far from truth, proving we find it
        self.alpha = nn.Parameter(torch.tensor([0.2]))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


#PHYSICS RESIDUAL
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
    return u_t - model.alpha * u_xx


#TRAINING POINTS
x_col = torch.rand(N_COLLOC, 1)
t_col = torch.rand(N_COLLOC, 1)

t_bc  = torch.rand(200, 1)
x_bc0 = torch.zeros(200, 1)
x_bc1 = torch.ones(200, 1)

x_ic  = torch.rand(300, 1)
t_ic  = torch.zeros(300, 1)
u_ic  = torch.sin(np.pi * x_ic)


#LOSS FUNCTION
def compute_loss(model):
    res      = physics_residual(model, x_col, t_col)
    loss_pde = torch.mean(res**2)

    loss_bc  = (torch.mean(model(x_bc0, t_bc)**2) +
                torch.mean(model(x_bc1, t_bc)**2))

    loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic)**2)

    loss_data = torch.mean((model(x_sensors, t_sensors) - u_sensors)**2)

    return loss_pde + loss_bc + loss_ic + 10.0 * loss_data, \
           loss_pde, loss_bc, loss_ic, loss_data


#PHASE 1 - adam (fast exploration)
model     = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=2000, gamma=0.5)

alpha_history = []
loss_history  = []

print(f"\n{'='*55}")
print(f"PHASE 1: Adam optimizer ({EPOCHS_ADAM} epochs)")
print(f"Starting alpha = {model.alpha.item():.3f}, True = {ALPHA_TRUE}")
print(f"{'='*55}")

for epoch in range(EPOCHS_ADAM):
    optimizer.zero_grad()
    loss, loss_pde, loss_bc, loss_ic, loss_data = compute_loss(model)
    loss.backward()
    optimizer.step()
    scheduler.step()

    alpha_history.append(model.alpha.item())
    loss_history.append(loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
              f"PDE: {loss_pde.item():.6f} | "
              f"Data: {loss_data.item():.7f} | "
              f"α = {model.alpha.item():.6f}")

print(f"\nAdam done. α = {model.alpha.item():.6f}  "
      f"(error: {abs(model.alpha.item()-ALPHA_TRUE)/ALPHA_TRUE*100:.2f}%)")


#PHASE 2 - lbfgs (precision convergence)
print(f"\n{'='*55}")
print("PHASE 2: L-BFGS optimizer (precision fine-tuning)")
print(f"{'='*55}")

optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(),
    lr=0.1,
    max_iter=500,
    history_size=50,
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
    line_search_fn='strong_wolfe'   # robust line search
)

lbfgs_alpha = []

def closure():
    optimizer_lbfgs.zero_grad()
    loss, *_ = compute_loss(model)
    loss.backward()
    lbfgs_alpha.append(model.alpha.item())
    return loss

for step in range(5):   # L-BFGS does many inner iterations per step
    loss_val = optimizer_lbfgs.step(closure)
    print(f"L-BFGS step {step+1} | Loss: {loss_val.item():.8f} | "
          f"α = {model.alpha.item():.8f}")

alpha_history.extend(lbfgs_alpha)

alpha_final = model.alpha.item()
error_pct   = abs(alpha_final - ALPHA_TRUE) / ALPHA_TRUE * 100

print(f"\n{'='*55}")
print(f"FINAL RESULT")
print(f"  α learned = {alpha_final:.8f}")
print(f"  α true    = {ALPHA_TRUE}")
print(f"  error     = {error_pct:.4f}%")
print(f"{'='*55}")


#PLOTTING
model.eval()
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PINN Inverse Problem (Adam + L-BFGS) — Production Quality",
             fontsize=13)

# Panel 1: alpha convergence
axes[0].plot(alpha_history, color='crimson', linewidth=1.2, label='α learned')
axes[0].axhline(ALPHA_TRUE, color='navy', lw=2,
                linestyle='--', label=f'True α = {ALPHA_TRUE}')
axes[0].axvline(EPOCHS_ADAM, color='gray', lw=1.5,
                linestyle=':', label='Adam → L-BFGS')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('α (learned)')
axes[0].set_title(f'α Convergence  (final error: {error_pct:.3f}%)')
axes[0].legend()
axes[0].grid(True)

# Panel 2: solution comparison at t=0.5
x_plot = torch.linspace(0, 1, 200).unsqueeze(1)
t_plot = torch.full_like(x_plot, 0.5)

with torch.no_grad():
    u_pred = model(x_plot, t_plot).numpy()

u_exact = (np.sin(np.pi * x_plot.numpy()) *
           np.exp(-ALPHA_TRUE * np.pi**2 * 0.5))

axes[1].plot(x_plot.numpy(), u_exact, 'b-',  lw=2.5, label='Exact (true α)')
axes[1].plot(x_plot.numpy(), u_pred,  'r--', lw=2,   label=f'PINN (α={alpha_final:.4f})')

# Only show sensors near t=0.5
mask = (t_sensors.squeeze() > 0.3) & (t_sensors.squeeze() < 0.7)
if mask.sum() > 0:
    axes[1].scatter(x_sensors[mask].numpy(),
                    u_sensors[mask].numpy(),
                    color='green', zorder=5, s=100,
                    label='Sensors near t=0.5', marker='x', linewidths=2)

axes[1].set_xlabel('x')
axes[1].set_ylabel('u(x, 0.5)')
axes[1].set_title('Solution Quality at t=0.5')
axes[1].legend()
axes[1].grid(True)

# Panel 3: loss history
axes[2].semilogy(loss_history, color='darkorange', lw=1.2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Loss (log scale)')
axes[2].set_title('Training Loss (Adam phase)')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('inverse_result.png', dpi=150)
plt.show()
print("Saved!")