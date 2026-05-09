# PINN for the 1D Heat Equation
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 

# reproducibility 
torch.manual_seed(42) 
# hyperparameters 
ALPHA = 0.01
N_COLLOC = 2000
N_BC = 100
N_IC = 200
EPOCHS = 5000
LR = 1e-3

# neural network 
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self, x, t):
        inputs = torch.cat([x,t], dim=1)
        return self.net(inputs)

# the physics residual
def physics_residual(model, x, t):
    x=x.requires_grad_(True)
    t =t.requires_grad_(True)

    u=model(x,t)

    u_t=torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x=torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

# second derivative
    u_xx=torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]
    residual=u_t-ALPHA*u_xx
    return residual

#training points
x_col=torch.rand(N_COLLOC,1)
t_col=torch.rand(N_COLLOC,1) 

t_bc=torch.rand(N_BC, 1)
x_bc0=torch.zeros(N_BC,1)
x_bc1=torch.ones(N_BC,1)

x_ic=torch.rand(N_IC,1)
t_ic=torch.zeros(N_IC,1)
u_ic=torch.sin(np.pi*x_ic)

#training loop
model=PINN()
optimizer=torch.optim.Adam(model.parameters(), lr=LR)
loss_history=[]

print('Started')
for epoch in range(EPOCHS):

    optimizer.zero_grad()
    #PDE residual
    res=physics_residual(model, x_col, t_col)
    loss_pde = torch.mean(res**2)

    #loss2:BC
    u_pred_bc0=model(x_bc0, t_bc)
    u_pred_bc1=model(x_bc1,t_bc)
    loss_bc=torch.mean(u_pred_bc0**2)+torch.mean(u_pred_bc1**2)

    #loss3:IC
    u_pred_ic=model(x_ic, t_ic)
    loss_ic=torch.mean((u_pred_ic-u_ic)**2)

    loss=loss_pde+loss_bc+loss_ic
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch%500==0:
        print(f'Epoch {epoch:5d} | Loss: {loss.item():.6f} |'
            f'PDE Loss: {loss_pde.item():.6f} | '
            f'BC Loss: {loss_bc.item():.6f} | '
            f'IC Loss: {loss_ic.item():.6f} '
        )

print('complete')        

#plotting
model.eval()
x_plot=torch.linspace(0,1,100).unsqueeze(1)

fig, axes = plt.subplots(1,3,figsize=(15,4))
fig.suptitle('PINN vs Exact Solution -1D HEAT Equation', fontsize=16)

for i, t_val in enumerate([0.0, 0.25, 1.0]):
    t_plot = torch.full_like(x_plot, t_val)

    with torch.no_grad():
        u_pred=model(x_plot, t_plot).numpy()

    u_exact=np.sin(np.pi *x_plot.numpy())*np.exp(-ALPHA*np.pi**2*t_val)
    axes[i].plot(x_plot.numpy(), u_exact, 'b-', linewidth=2, label='Exact')
    axes[i].plot(x_plot.numpy(), u_pred, 'r--', linewidth=2, label='PINN')
    axes[i].set_title(f't={t_val}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('u(x,t)')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.savefig('heat_equation_result.png', dpi=150)
plt.show()
print('saved!')    

