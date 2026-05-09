# PINN — Inverse Problem

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ground truth
ALPHA_TRUE  = 0.4

# training config 
N_COLLOC    = 3000
N_SENSORS   = 8
EPOCHS      = 8000
LR          = 1e-3
NOISE_LEVEL = 0.02 

#generate sensor data
x_sensors = torch.rand(N_SENSORS, 1)
t_sensors = torch.rand(N_SENSORS, 1)

u_exact_sensors = (torch.sin(np.pi * x_sensors) *
                   torch.exp(-ALPHA_TRUE * np.pi**2 * t_sensors))

# Add Gaussian noise
noise = NOISE_LEVEL * torch.randn_like(u_exact_sensors)
u_sensors = u_exact_sensors + noise

print(f"Sensor locations (x, t) and noisy measurements:")
for i in range(N_SENSORS):
    print(f"  Sensor {i+1}: x={x_sensors[i,0]:.3f}, "
          f"t={t_sensors[i,0]:.3f}, "
          f"u_measured={u_sensors[i,0]:.4f} "
          f"(true={u_exact_sensors[i,0]:.4f})")

# PINN model 
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.alpha = nn.Parameter(torch.tensor([0.2])) #initialize it far from true alpha to show PINN can recover

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# physics residual - uses learned alpha
def physics_residual(model, x, t):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)

    u   = model(x, t)
    u_t = torch.autograd.grad(u,   t, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx= torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]

    # model.alpha is updated by the optimizer each step
    return u_t - model.alpha * u_xx


# collocation and boundary points 
x_col = torch.rand(N_COLLOC, 1)
t_col = torch.rand(N_COLLOC, 1)

t_bc  = torch.rand(200, 1)
x_bc0 = torch.zeros(200, 1)
x_bc1 = torch.ones(200, 1)

x_ic  = torch.rand(200, 1)
t_ic  = torch.zeros(200, 1)
u_ic  = torch.sin(np.pi * x_ic)


#training loop 
model     = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=2000, gamma=0.5)

alpha_history = []   #track how alpha evolves
loss_history  = []

print(f"\nStarting with alpha_guess = {model.alpha.item():.4f}")
print(f"True alpha               = {ALPHA_TRUE}")
print(f"Must recover from only {N_SENSORS} noisy sensor readings\n")

for epoch in range(EPOCHS):
    optimizer.zero_grad()

    # PDE residual
    res      = physics_residual(model, x_col, t_col)
    loss_pde = torch.mean(res**2)

    #BC
    loss_bc  = (torch.mean(model(x_bc0, t_bc)**2) +
                torch.mean(model(x_bc1, t_bc)**2))

    #IC
    loss_ic  = torch.mean((model(x_ic, t_ic) - u_ic)**2)

    #data loss from 8 sensors 
    u_pred_sensors = model(x_sensors, t_sensors)
    loss_data      = torch.mean((u_pred_sensors - u_sensors)**2)

    # total loss with heavy weighting on data loss because it is our truth anchor
    loss = loss_pde + loss_bc + loss_ic + 10.0 * loss_data

    loss.backward()
    optimizer.step()
    scheduler.step()

    alpha_history.append(model.alpha.item())
    loss_history.append(loss.item())

    if epoch % 1000 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
              f"PDE: {loss_pde.item():.6f} | "
              f"Data: {loss_data.item():.6f} | "
              f"alpha_learned: {model.alpha.item():.6f} | "
              f"alpha_true: {ALPHA_TRUE}")

alpha_final = model.alpha.item()
error_pct   = abs(alpha_final - ALPHA_TRUE) / ALPHA_TRUE * 100
print(f"\n{'='*55}")
print(f"RESULT: alpha learned = {alpha_final:.6f}")
print(f"        alpha true    = {ALPHA_TRUE}")
print(f"        error         = {error_pct:.3f}%")
print(f"{'='*55}")


#plotting
model.eval()

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("PINN Inverse Problem — Recovering Hidden Physics", fontsize=14)

#Panel 1: alpha convergence
axes[0].plot(alpha_history, color='crimson', linewidth=1.5)
axes[0].axhline(ALPHA_TRUE, color='navy', linewidth=2,
                linestyle='--', label=f'True α = {ALPHA_TRUE}')
axes[0].axhline(0.5, color='gray', linewidth=1,
                linestyle=':', label='Initial guess = 0.5')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('α (learned)')
axes[0].set_title('α Convergence During Training')
axes[0].legend()
axes[0].grid(True)

#Panel 2: solution vs exact at t=0.5
x_plot = torch.linspace(0, 1, 200).unsqueeze(1)
t_plot = torch.full_like(x_plot, 0.5)

with torch.no_grad():
    u_pred = model(x_plot, t_plot).numpy()

u_exact = (np.sin(np.pi * x_plot.numpy()) *
           np.exp(-ALPHA_TRUE * np.pi**2 * 0.5))

axes[1].plot(x_plot.numpy(), u_exact, 'b-',  lw=2, label='Exact (true α)')
axes[1].plot(x_plot.numpy(), u_pred,  'r--', lw=2, label='PINN (learned α)')
axes[1].scatter(x_sensors.numpy(), u_sensors.numpy(),
                color='green', zorder=5, s=80,
                label=f'{N_SENSORS} noisy sensors', marker='x')
axes[1].set_xlabel('x')
axes[1].set_ylabel('u(x, 0.5)')
axes[1].set_title('Solution at t=0.5')
axes[1].legend()
axes[1].grid(True)

#Panel 3:losshistory
axes[2].semilogy(loss_history, color='darkorange', linewidth=1.2)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Total Loss (log scale)')
axes[2].set_title('Training Loss')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('inverse_result.png', dpi=150)
plt.show()
print("Plot saved!")