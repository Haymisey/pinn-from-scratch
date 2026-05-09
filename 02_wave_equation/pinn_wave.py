#PINN FOR 1D WAVE EQUATION
import torch
import torch.nn as nn 
import numpy as np 
import matplotlib.pyplot as plt 

torch.manual_seed(42)

#hyperparameters 
C=1.0
N_COLLOC=5000
N_BC=200
N_IC=500
EPOCHS=8000
LR=1e-3

#network- wider and deeper then before...needs more capacity due to oscillation
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(2,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,1)
        )

    def forward(self,x,t):
        return self.net(torch.cat([x,t],dim=1))

# define physics residual
def physics_residual(model,x,t):
    x=x.requires_grad_(True)
    t =t.requires_grad_(True)

    u=model(x,t)

    u_t=torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    #second derivative
    u_tt=torch.autograd.grad(
        u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    u_x=torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx=torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    residual=u_tt-C**2*u_xx
    return residual
#training points
x_col=torch.rand(N_COLLOC,1)
t_col=torch.rand(N_COLLOC,1)
#BC points
t_bc=torch.rand(N_BC,1)
x_bc0=torch.zeros(N_BC,1)
x_bc1=torch.ones(N_BC,1)
#IC1
x_ic=torch.rand(N_IC,1)
t_ic=torch.zeros(N_IC,1)
u_ic=torch.sin(np.pi*x_ic)
#IC2
x_ic2=torch.rand(N_IC,1)
t_ic2=torch.zeros(N_IC,1)

#training loop
model=PINN()
optimizer=torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
loss_history=[]

print('Starting Wave Equation Training')
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    
    #PDE residual
    res=physics_residual(model, x_col, t_col)
    loss_pde = torch.mean(res**2)

    #BC loss
    u_pred_bc0=model(x_bc0,t_bc)
    u_pred_bc1=model(x_bc1,t_bc)
    loss_bc = torch.mean(u_pred_bc0**2)+torch.mean(u_pred_bc1**2)

    #IC1 loss  
    u_pred_ic=model(x_ic, t_ic)
    loss_ic1 = torch.mean((u_pred_ic-u_ic)**2)

    #IC2 loss
    t_ic2_grad=t_ic2.requires_grad_(True)
    u_ic2=model(x_ic2,t_ic2_grad)
    u_t_ic2=torch.autograd.grad(
        u_ic2, t_ic2_grad, grad_outputs=torch.ones_like(u_ic2), create_graph=True)[0]
    loss_ic2 = torch.mean(u_t_ic2**2)

    loss=loss_pde+loss_bc+loss_ic1+2*loss_ic2 # we need to weight ic2 more heavily because it is hard to enforce
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_history.append(loss.item())

    if epoch % 1000 == 0:
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f} | "
              f"PDE: {loss_pde.item():.6f} | "
              f"IC1: {loss_ic1.item():.6f} | "
              f"IC2: {loss_ic2.item():.6f} | LR: {lr_now:.5f}")

print("Training complete!")

#evaluation
model.eval()
x_plot=torch.linspace(0,1,200).unsqueeze(1)

fig, axes=plt.subplots(2,3,figsize=(15,8))
fig.suptitle('PINN vs Exact Solution - 1D Wave Equation', fontsize=14)

for i, t_val in enumerate([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]):
    row, col=divmod(i,3)
    t_plot=torch.full_like(x_plot,t_val)

    with torch.no_grad():
        u_pred=model(x_plot,t_plot).numpy()

    u_exact = np.sin(np.pi * x_plot.numpy()) * np.cos(np.pi * C * t_val)
    axes[row][col].plot(x_plot.numpy(), u_exact, 'b-', linewidth=2, label='Exact')
    axes[row][col].plot(x_plot.numpy(), u_pred, 'r--', linewidth=2, label='PINN')
    axes[row][col].set_title(f't={t_val}')
    axes[row][col].set_xlabel('x')
    axes[row][col].set_ylabel('u(x,t)')
    axes[row][col].legend()
    axes[row][col].set_ylim(-1.2,1.2)
    axes[row][col].grid(True)
#loss curve in last subplot
axes[1][2].semilogy(loss_history,'k-')
axes[1][2].set_title('Training Loss')
axes[1][2].set_xlabel('Epoch')
axes[1][2].set_ylabel('Loss(log scale)')
axes[1][2].grid(True)

plt.tight_layout()
plt.savefig('wave_equation_result.png', dpi=150)
plt.show()
print('saved!')
    