import torch
import torch.nn as nn
import numpy as np

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