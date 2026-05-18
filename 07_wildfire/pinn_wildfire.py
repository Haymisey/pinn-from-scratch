# PINN — Wildfire Spread on Real NASA FIRMS Data
# Path 2: IC-anchored architecture + Temporal Derivative Supervision
#
# Architecture fix: u = u0(x,y) * (1 + f(x,y,t))
#   u0 = learned initial condition (anchored to day-4 data)
#   f  = learned evolution (starts near zero)
# This prevents sol_net from memorizing data independently of physics
#
# PDE:  ∂u/∂t = ∇·[D(x,y)∇u] + β·u·(1-u)
# Data: 2023 Alberta Wildfire, NASA FIRMS VIIRS_SNPP_SP
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_COLLOC    = 15000
EPOCHS_ADAM = 20000
LR          = 1e-3


#LOAD DATA
train_df = pd.read_csv("data/pinn_train.csv")
val_df   = pd.read_csv("data/pinn_val.csv")
deriv_df = pd.read_csv("data/temporal_derivatives.csv")
ic_df    = pd.read_csv("data/initial_condition.csv")
meta     = pd.read_csv("data/normalization.csv",
                       header=None, index_col=0).squeeze()

print(f"Train sensors:     {len(train_df):,}")
print(f"Val sensors:       {len(val_df):,}")
print(f"Derivative points: {len(deriv_df):,}")
print(f"IC points (day 4): {len(ic_df):,}")

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)

x_tr, y_tr, t_tr, u_tr = [to_tensor(train_df[c].values) for c in ['x','y','t','u']]
x_vl, y_vl, t_vl, u_vl = [to_tensor(val_df[c].values)   for c in ['x','y','t','u']]

#Derivative supervision
x_dt     = to_tensor(deriv_df['x'].values)
y_dt     = to_tensor(deriv_df['y'].values)
t_dt     = to_tensor(deriv_df['t'].values)
dudt_obs = to_tensor(deriv_df['dudt'].values)
dudt_obs = torch.clamp(dudt_obs, -5.0, 5.0)
dudt_std = dudt_obs.std() + 1e-8
dudt_obs = dudt_obs / dudt_std   # normalize to unit variance
print(f"dudt normalized std: {dudt_obs.std().item():.3f}")

#Initial condition (day 4 data)
x_ic = to_tensor(ic_df['x'].values)
y_ic = to_tensor(ic_df['y'].values)
u_ic = to_tensor(ic_df['u'].values)
t_ic = torch.zeros_like(x_ic)


#ARCHITECTURE
class InitialCondNet(nn.Module):
    """Learns u(x,y,0) from day-4 satellite observations"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),  nn.Sigmoid()  # u0 ∈ [0,1]
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))


class EvolutionNet(nn.Module):
    """
    Learns f(x,y,t) — temporal evolution relative to IC
    Initialized near zero so u ≈ u0 at start of training
    Forces the network to explain changes via physics (D, β)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128,128), nn.Tanh(),
            nn.Linear(128, 1),  nn.Tanh()   # f ∈ [-1, 1]
        )
        # Small init — f starts near zero
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                l.weight.data *= 0.01
                nn.init.zeros_(l.bias)

    def forward(self, x, y, t):
        return self.net(torch.cat([x, y, t], dim=1))


class DiffusivityNet(nn.Module):
    """D(x,y) bounded in [D_MIN, D_MAX] by construction"""
    D_MIN, D_MAX = 0.01, 2.0

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64, 1), nn.Tanh()    # output ∈ [-1,1]
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, y):
        raw = self.net(torch.cat([x, y], dim=1))
        # Guaranteed range [D_MIN, D_MAX]
        return self.D_MIN + (self.D_MAX - self.D_MIN) * (raw + 1) / 2


class WildfirePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ic_net   = InitialCondNet()
        self.evo_net  = EvolutionNet()
        self.diff_net = DiffusivityNet()
        # raw_beta: softplus ensures β > 0 always
        self.raw_beta = nn.Parameter(torch.tensor([1.0]))

    @property
    def beta(self):
        return torch.nn.functional.softplus(self.raw_beta)

    def u(self, x, y, t):
        u0 = self.ic_net(x, y)          # initial condition field
        f  = self.evo_net(x, y, t)      # temporal evolution
        # u = u0 * (1 + f), clamped to [0,1]
        # At t=0: f≈0 → u≈u0 (matches IC data)
        # At t>0: f encodes growth/decay driven by physics
        return torch.clamp(u0 * (1.0 + f), 0.0, 1.0)

    def D(self, x, y):
        return self.diff_net(x, y)


#PDE TERMS — returns u_t, rhs, and residual separately
def compute_pde_terms(model, x, y, t):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    u = model.u(x, y, t)
    D = model.D(x, y)

    u_t  = torch.autograd.grad(u,   t, torch.ones_like(u),   create_graph=True)[0]
    u_x  = torch.autograd.grad(u,   x, torch.ones_like(u),   create_graph=True)[0]
    u_y  = torch.autograd.grad(u,   y, torch.ones_like(u),   create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    D_x  = torch.autograd.grad(D,   x, torch.ones_like(D),   create_graph=True)[0]
    D_y  = torch.autograd.grad(D,   y, torch.ones_like(D),   create_graph=True)[0]

    diffusion = D * (u_xx + u_yy) + D_x * u_x + D_y * u_y
    reaction  = model.beta * u * (1 - u)
    rhs       = diffusion + reaction

    return u_t, rhs, u_t - rhs   # (predicted dudt, rhs, residual)


#COLLOCATION POINTS
def sample_collocation(n):
    return [torch.rand(n, 1).to(device) for _ in range(3)]

x_col, y_col, t_col = sample_collocation(N_COLLOC)


#LOSS FUNCTION
def compute_loss(model, x_col, y_col, t_col):

    # --- Loss 1: PDE residual at collocation points ---
    _, _, residual = compute_pde_terms(model, x_col, y_col, t_col)
    loss_pde = torch.mean(residual**2)

    # --- Loss 2: Match observed u values ---
    u_pred  = model.u(x_tr, y_tr, t_tr)
    weights = (u_tr + 0.1).detach()
    weights = weights / weights.sum()
    loss_data = torch.sum(weights * (u_pred - u_tr)**2) * len(u_tr)

    # --- Loss 3: Initial condition (anchor to day-4 data) ---
    u_ic_pred = model.u(x_ic, y_ic, t_ic)
    loss_ic   = torch.mean((u_ic_pred - u_ic)**2)

    # --- Loss 4: Temporal derivative supervision ---
    # Forces u_t to match satellite-observed ∂u/∂t
    u_t_pred, rhs_pred, _ = compute_pde_terms(model, x_dt, y_dt, t_dt)
    loss_dudt = torch.mean((u_t_pred - dudt_obs)**2)

    # --- Loss 5: RHS must equal observed dudt ---
    # Directly constrains D and β via the right-hand side
    loss_rhs  = torch.mean((rhs_pred - dudt_obs)**2)

    # --- Loss 6: D smoothness ---
    x_r = torch.rand(500, 1).to(device).requires_grad_(True)
    y_r = torch.rand(500, 1).to(device).requires_grad_(True)
    D_r = model.D(x_r, y_r)
    Dx  = torch.autograd.grad(D_r, x_r, torch.ones_like(D_r), create_graph=True)[0]
    Dy  = torch.autograd.grad(D_r, y_r, torch.ones_like(D_r), create_graph=True)[0]
    loss_smooth = torch.mean(Dx**2 + Dy**2)

    total = (1.0  * loss_pde
           + 5.0  * loss_data
           + 50.0 * loss_ic      # strong IC anchor — prevents memorization
           + 10.0 * loss_dudt    # supervise u_t directly
           + 10.0 * loss_rhs     # constrain D and β via rhs
           + 0.1  * loss_smooth)

    return total, loss_pde, loss_data, loss_ic, loss_dudt, loss_rhs


#TRAINING
model = WildfirePINN().to(device)
optimizer = torch.optim.Adam([
    {'params': model.ic_net.parameters(),   'lr': 1e-3},
    {'params': model.evo_net.parameters(),  'lr': 1e-3},
    {'params': model.diff_net.parameters(), 'lr': 1e-3},
    {'params': [model.raw_beta],            'lr': 1e-2},  # β moves fast
], lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5)

loss_hist = []
beta_hist = []

print(f"\nTraining Wildfire PINN — Path 2 (IC-anchored + derivative supervision)")
print(f"  β initial = {model.beta.item():.4f}\n")

for epoch in range(EPOCHS_ADAM):
    if epoch % 500 == 0 and epoch > 0:
        x_col, y_col, t_col = sample_collocation(N_COLLOC)

    optimizer.zero_grad()
    loss, l_pde, l_data, l_ic, l_dudt, l_rhs = compute_loss(
        model, x_col, y_col, t_col)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_hist.append(loss.item())
    beta_hist.append(model.beta.item())

    if epoch % 2000 == 0:
        with torch.no_grad():
            xe = torch.rand(1000,1).to(device)
            ye = torch.rand(1000,1).to(device)
            Dm = model.D(xe, ye).mean().item()
        print(f"Epoch {epoch:5d} | Loss:{loss.item():8.2f} | "
              f"PDE:{l_pde.item():.4f} | Data:{l_data.item():.3f} | "
              f"IC:{l_ic.item():.4f} | dudt:{l_dudt.item():.4f} | "
              f"rhs:{l_rhs.item():.4f} | β={model.beta.item():.4f} | D={Dm:.4f}")

print("\nAdam done.")


#L-BFGS FINE-TUNING
print("L-BFGS fine-tuning...")
opt_lb = torch.optim.LBFGS(model.parameters(), lr=0.05,
                             max_iter=200, history_size=50,
                             line_search_fn='strong_wolfe')

def closure():
    opt_lb.zero_grad()
    loss, *_ = compute_loss(model, x_col, y_col, t_col)
    loss.backward()
    return loss

for step in range(15):
    lv = opt_lb.step(closure)
    if step % 5 == 0:
        with torch.no_grad():
            Dm = model.D(torch.rand(500,1).to(device),
                         torch.rand(500,1).to(device)).mean().item()
        print(f"  Step {step+1:2d} | Loss:{lv.item():.4f} | "
              f"β={model.beta.item():.5f} | D_mean={Dm:.4f}")

#VALIDATION
model.eval()
with torch.no_grad():
    u_vp    = model.u(x_vl, y_vl, t_vl)
    val_mse = torch.mean((u_vp - u_vl)**2).item()
    val_mae = torch.mean(torch.abs(u_vp - u_vl)).item()

with torch.enable_grad():
    u_t_vp, rhs_vp, _ = compute_pde_terms(model, x_dt, y_dt, t_dt)
    dudt_mae = torch.mean(torch.abs(u_t_vp - dudt_obs)).item()

with torch.no_grad():
    Dfinal = model.D(torch.rand(2000,1).to(device),
                     torch.rand(2000,1).to(device))

print(f"\n{'='*55}")
print(f"VALIDATION RESULTS")
print(f"  Val MSE (u):  {val_mse:.6f}")
print(f"  Val MAE (u):  {val_mae:.6f}")
print(f"  dudt MAE:     {dudt_mae:.6f}")
print(f"  β learned:    {model.beta.item():.6f}")
print(f"  D mean:       {Dfinal.mean().item():.4f}")
print(f"  D std:        {Dfinal.std().item():.4f}")
print(f"  D range:      [{Dfinal.min().item():.4f}, {Dfinal.max().item():.4f}]")
print(f"{'='*55}")

#VISUALIZATION
model.eval()
res = 150
x_lin = torch.linspace(0,1,res); y_lin = torch.linspace(0,1,res)
X, Y  = torch.meshgrid(x_lin, y_lin, indexing='ij')
xf = X.reshape(-1,1).to(device); yf = Y.reshape(-1,1).to(device)

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "PINN — 2023 Alberta Wildfire (Path 2: IC-anchored + Derivative Supervision)\n"
    f"β={model.beta.item():.4f} | Val MAE={val_mae:.4f} | dudt MAE={dudt_mae:.4f}",
    fontsize=12, fontweight='bold')

# Panel 1: D(x,y)
ax1 = fig.add_subplot(2,3,1)
with torch.no_grad():
    Dg = model.D(xf,yf).reshape(res,res).cpu().numpy()
im1 = ax1.contourf(X.numpy(), Y.numpy(), Dg, levels=50, cmap='viridis')
plt.colorbar(im1, ax=ax1, label='D(x,y)')
ax1.scatter(train_df['x'], train_df['y'], s=1, c='white', alpha=0.2)
ax1.set_title(f'★ Recovered D(x,y)\n'
              f'D∈[{Dg.min():.3f},{Dg.max():.3f}]  std={Dg.std():.4f}',
              fontsize=10)
ax1.set_xlabel('x'); ax1.set_ylabel('y')

# Panels 2-4: u snapshots
for idx, tv in enumerate([0.1, 0.5, 0.9]):
    ax = fig.add_subplot(2,3,idx+2)
    tf = torch.full((res*res,1), tv).to(device)
    with torch.no_grad():
        ug = model.u(xf,yf,tf).reshape(res,res).cpu().numpy()
    im = ax.contourf(X.numpy(), Y.numpy(), ug,
                     levels=40, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='u')
    obs = train_df[abs(train_df['t']-tv)<0.08]
    if len(obs):
        ax.scatter(obs['x'], obs['y'], c=obs['u'], cmap='cool',
                   s=8, alpha=0.8, edgecolors='white', lw=0.3,
                   label=f'{len(obs)} obs')
    day = int(tv*(meta['t_max']-meta['t_min'])+meta['t_min'])
    ax.set_title(f't={tv} (≈Day {day})', fontsize=10)
    ax.legend(fontsize=7)

# Panel 5: β convergence
ax5 = fig.add_subplot(2,3,5)
ax5.plot(beta_hist, color='crimson', lw=1.2)
ax5.axhline(model.beta.item(), color='navy', ls='--', lw=1.5,
            label=f'Final β={model.beta.item():.4f}')
ax5.set_xlabel('Epoch'); ax5.set_ylabel('β')
ax5.set_title('β Convergence')
ax5.legend(); ax5.grid(True, alpha=0.3)

# Panel 6: dudt predicted vs observed (identifiability check)
ax6 = fig.add_subplot(2,3,6)
with torch.enable_grad():
    ut_pred = compute_pde_terms(model, x_dt, y_dt, t_dt)[0].detach().cpu().numpy()
ax6.scatter(dudt_obs.cpu().numpy(), ut_pred,
            s=10, alpha=0.6, color='steelblue')
lim = 2.5
ax6.plot([-lim,lim],[-lim,lim],'r--',lw=2, label='Perfect prediction')
ax6.set_xlabel('∂u/∂t observed (satellite)')
ax6.set_ylabel('∂u/∂t predicted (PINN)')
ax6.set_title('Derivative Supervision Quality\n(diagonal = perfect)')
ax6.legend(); ax6.grid(True, alpha=0.3)
ax6.set_xlim(-lim,lim); ax6.set_ylim(-lim,lim)

plt.tight_layout()
plt.savefig('wildfire_pinn_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wildfire_pinn_result.png")

torch.save(model.state_dict(), 'wildfire_pinn_model.pth')
pd.Series({
    'beta':     model.beta.item(),
    'val_mse':  val_mse,
    'val_mae':  val_mae,
    'dudt_mae': dudt_mae,
    'D_mean':   Dfinal.mean().item(),
    'D_std':    Dfinal.std().item(),
}).to_csv('wildfire_results.csv', header=False)
print("Results saved: wildfire_results.csv")



'''
# PINN — Wildfire Spread v2: Full Domain Grid
#
# KEY DIFFERENCE from v1:
#   Before: 1,227 points, all burned (u>0)
#   Now:    40,000 points, 97.5% unburned (u=0)
#
# This makes D and β identifiable because:
#   - PDE must explain WHY u=0 in unburned cells
#   - PDE must explain WHY u>0 in burned cells
#   - The boundary between them constrains D and β uniquely
#
# PDE:  ∂u/∂t = ∇·[D(x,y)∇u] + β·u·(1-u)
# Data: 2023 Alberta Wildfire, NASA FIRMS VIIRS_SNPP_SP

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_COLLOC    = 10000
EPOCHS_ADAM = 20000
LR          = 1e-3


#LOAD GRID DATA
train_df = pd.read_csv("data/grid_train.csv")
val_df   = pd.read_csv("data/grid_val.csv")
meta     = pd.read_csv("data/normalization.csv",
                       header=None, index_col=0).squeeze()

print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")
print(f"Burned train:   {train_df['is_burned'].sum():,} "
      f"({train_df['is_burned'].mean()*100:.1f}%)")
print(f"Unburned train: {(train_df['is_burned']==0).sum():,} "
      f"({(train_df['is_burned']==0).mean()*100:.1f}%)")

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(1).to(device)

x_tr = to_tensor(train_df['x'].values)
y_tr = to_tensor(train_df['y'].values)
t_tr = to_tensor(train_df['t'].values)
u_tr = to_tensor(train_df['u'].values)

x_vl = to_tensor(val_df['x'].values)
y_vl = to_tensor(val_df['y'].values)
t_vl = to_tensor(val_df['t'].values)
u_vl = to_tensor(val_df['u'].values)

# Separate burned and unburned for weighted loss
burned_mask   = torch.tensor(
    train_df['is_burned'].values, dtype=torch.float32).unsqueeze(1).to(device)
unburned_mask = 1.0 - burned_mask

print(f"\nu range in train: [{u_tr.min().item():.3f}, {u_tr.max().item():.3f}]")


#ARCHITECTURE 
class SolutionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),  nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),  nn.Sigmoid()  # u ∈ [0,1]
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, y, t):
        return self.net(torch.cat([x, y, t], dim=1))


class DiffusivityNet(nn.Module):
    """D(x,y) bounded [D_MIN, D_MAX] by construction"""
    D_MIN, D_MAX = 0.01, 2.0

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64,64), nn.Tanh(),
            nn.Linear(64, 1), nn.Tanh()
        )
        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)
                nn.init.zeros_(l.bias)

    def forward(self, x, y):
        raw = self.net(torch.cat([x, y], dim=1))
        return self.D_MIN + (self.D_MAX - self.D_MIN) * (raw + 1) / 2


class WildfirePINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.sol_net  = SolutionNet()
        self.diff_net = DiffusivityNet()
        self.raw_beta = nn.Parameter(torch.tensor([1.0]))

    @property
    def beta(self):
        return torch.nn.functional.softplus(self.raw_beta)

    def u(self, x, y, t):  return self.sol_net(x, y, t)
    def D(self, x, y):     return self.diff_net(x, y)


#PDE RESIDUAL
def pde_residual(model, x, y, t):
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)
    t = t.requires_grad_(True)

    u    = model.u(x, y, t)
    D    = model.D(x, y)
    u_t  = torch.autograd.grad(u,   t, torch.ones_like(u),   create_graph=True)[0]
    u_x  = torch.autograd.grad(u,   x, torch.ones_like(u),   create_graph=True)[0]
    u_y  = torch.autograd.grad(u,   y, torch.ones_like(u),   create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    D_x  = torch.autograd.grad(D,   x, torch.ones_like(D),   create_graph=True)[0]
    D_y  = torch.autograd.grad(D,   y, torch.ones_like(D),   create_graph=True)[0]

    diffusion = D * (u_xx + u_yy) + D_x * u_x + D_y * u_y
    reaction  = model.beta * u * (1 - u)
    return u_t - diffusion - reaction


#COLLOCATION POINTS
def sample_collocation(n):
    return [torch.rand(n, 1).to(device) for _ in range(3)]

x_col, y_col, t_col = sample_collocation(N_COLLOC)


#LOSS FUNCTION
#    Critical: weighted loss treats burned/unburned differently
#    Burned cells (2.5%): high weight — real fire observations
#    Unburned cells (97.5%): lower weight — absence of fire
def compute_loss(model, x_col, y_col, t_col):

    # PDE residual at collocation points
    res      = pde_residual(model, x_col, y_col, t_col)
    loss_pde = torch.mean(res**2)

    # Data loss — separate burned and unburned
    u_pred = model.u(x_tr, y_tr, t_tr)
    err    = (u_pred - u_tr)**2

    # Burned cells: full weight (1.0) — these are real fire observations
    loss_burned   = torch.sum(err * burned_mask) / (burned_mask.sum() + 1e-8)

    # Unburned cells: lower weight — but still important for identifiability
    loss_unburned = torch.sum(err * unburned_mask) / (unburned_mask.sum() + 1e-8)

    # D smoothness
    x_r = torch.rand(500, 1).to(device).requires_grad_(True)
    y_r = torch.rand(500, 1).to(device).requires_grad_(True)
    D_r = model.D(x_r, y_r)
    Dx  = torch.autograd.grad(D_r, x_r, torch.ones_like(D_r), create_graph=True)[0]
    Dy  = torch.autograd.grad(D_r, y_r, torch.ones_like(D_r), create_graph=True)[0]
    loss_smooth = torch.mean(Dx**2 + Dy**2)

    # Weights: PDE and burned equally important
    # Unburned less weight — it's mostly saying "nothing here"
    total = (5.0  * loss_pde
           + 10.0 * loss_burned    # real fire signal
           + 1.0  * loss_unburned  # absence constraint
           + 0.1  * loss_smooth)

    return total, loss_pde, loss_burned, loss_unburned


#TRAINING
model = WildfirePINN().to(device)
optimizer = torch.optim.Adam([
    {'params': model.sol_net.parameters(),  'lr': 1e-3},
    {'params': model.diff_net.parameters(), 'lr': 1e-3},
    {'params': [model.raw_beta],            'lr': 1e-2},
], lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5000, gamma=0.5)

loss_hist = []
beta_hist = []

print(f"\nTraining Wildfire PINN v2 — Full Domain Grid")
print(f"  Grid:  40,000 pts | PDE: {N_COLLOC:,} | Epochs: {EPOCHS_ADAM}")
print(f"  β initial = {model.beta.item():.4f}\n")

for epoch in range(EPOCHS_ADAM):
    if epoch % 1000 == 0 and epoch > 0:
        x_col, y_col, t_col = sample_collocation(N_COLLOC)

    optimizer.zero_grad()
    loss, l_pde, l_burn, l_unburn = compute_loss(
        model, x_col, y_col, t_col)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    loss_hist.append(loss.item())
    beta_hist.append(model.beta.item())

    if epoch % 2000 == 0:
        with torch.no_grad():
            xe = torch.rand(500,1).to(device)
            ye = torch.rand(500,1).to(device)
            Dm = model.D(xe, ye).mean().item()
            Ds = model.D(xe, ye).std().item()
        print(f"Epoch {epoch:5d} | Loss:{loss.item():8.3f} | "
              f"PDE:{l_pde.item():.4f} | "
              f"Burned:{l_burn.item():.4f} | "
              f"Unburned:{l_unburn.item():.6f} | "
              f"β={model.beta.item():.4f} | "
              f"D={Dm:.4f}±{Ds:.4f}")

print("\nAdam done.")


#L-BFGS
print("L-BFGS fine-tuning...")
opt_lb = torch.optim.LBFGS(model.parameters(), lr=0.1,
                             max_iter=500, history_size=100,
                             tolerance_grad=1e-9,
                             line_search_fn='strong_wolfe')

def closure():
    opt_lb.zero_grad()
    loss, *_ = compute_loss(model, x_col, y_col, t_col)
    loss.backward()
    return loss

for step in range(20):
    lv = opt_lb.step(closure)
    if step % 5 == 0:
        with torch.no_grad():
            Dm = model.D(torch.rand(500,1).to(device),
                         torch.rand(500,1).to(device)).mean().item()
        print(f"  Step {step+1:2d} | Loss:{lv.item():.5f} | "
              f"β={model.beta.item():.5f} | D={Dm:.4f}")


#VALIDATION
model.eval()
with torch.no_grad():
    u_vp      = model.u(x_vl, y_vl, t_vl)
    val_mse   = torch.mean((u_vp - u_vl)**2).item()
    val_mae   = torch.mean(torch.abs(u_vp - u_vl)).item()

    # Separate metrics for burned and unburned validation
    vburn     = torch.tensor(val_df['is_burned'].values,
                             dtype=torch.float32).unsqueeze(1).to(device)
    err_v     = torch.abs(u_vp - u_vl)
    mae_burn  = (torch.sum(err_v * vburn) /
                 (vburn.sum() + 1e-8)).item()
    mae_unb   = (torch.sum(err_v * (1-vburn)) /
                 ((1-vburn).sum() + 1e-8)).item()

    Dfinal    = model.D(torch.rand(2000,1).to(device),
                        torch.rand(2000,1).to(device))

print(f"\n{'='*60}")
print(f"RESEARCH RESULTS")
print(f"{'='*60}")
print(f"  β (fire reaction rate):   {model.beta.item():.6f}")
print(f"  D mean:                   {Dfinal.mean().item():.4f}")
print(f"  D std:                    {Dfinal.std().item():.4f}")
print(f"  D range:                  [{Dfinal.min().item():.4f}, "
      f"{Dfinal.max().item():.4f}]")
print(f"  Val MAE (overall):        {val_mae:.6f}")
print(f"  Val MAE (burned cells):   {mae_burn:.6f}")
print(f"  Val MAE (unburned cells): {mae_unb:.6f}")
print(f"{'='*60}")


#VISUALIZATION
model.eval()
res = 150
x_lin = torch.linspace(0,1,res); y_lin = torch.linspace(0,1,res)
X, Y  = torch.meshgrid(x_lin, y_lin, indexing='ij')
xf    = X.reshape(-1,1).to(device)
yf    = Y.reshape(-1,1).to(device)

fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "PINN v2 — 2023 Alberta Wildfire (Full Domain Grid)\n"
    f"β={model.beta.item():.4f} | "
    f"D∈[{Dfinal.min().item():.3f},{Dfinal.max().item():.3f}] | "
    f"Val MAE={val_mae:.4f}",
    fontsize=12, fontweight='bold')

# Panel 1: Recovered D(x,y)
ax1 = fig.add_subplot(2, 3, 1)
with torch.no_grad():
    Dg = model.D(xf, yf).reshape(res,res).cpu().numpy()
im1 = ax1.contourf(X.numpy(), Y.numpy(), Dg, levels=50, cmap='viridis')
plt.colorbar(im1, ax=ax1, label='D(x,y)')
# Overlay fire locations
fire_pts = train_df[train_df['is_burned'] == 1]
ax1.scatter(fire_pts['x'], fire_pts['y'],
            s=1, c='white', alpha=0.3, label='Fire observed')
ax1.set_title(f'★ Recovered D(x,y)\n'
              f'std={Dg.std():.4f}  range=[{Dg.min():.3f},{Dg.max():.3f}]',
              fontsize=10)
ax1.set_xlabel('x (normalized lon)')
ax1.set_ylabel('y (normalized lat)')
ax1.legend(fontsize=7)

# Panels 2-4: Fire spread snapshots
for idx, tv in enumerate([0.1, 0.5, 0.9]):
    ax = fig.add_subplot(2, 3, idx+2)
    tf = torch.full((res*res,1), tv).to(device)
    with torch.no_grad():
        ug = model.u(xf, yf, tf).reshape(res,res).cpu().numpy()
    im = ax.contourf(X.numpy(), Y.numpy(), ug,
                     levels=40, cmap='hot', vmin=0, vmax=0.7)
    plt.colorbar(im, ax=ax, label='u (fire intensity)')

    # Overlay observed grid cells near this time
    obs = train_df[
        (abs(train_df['t'] - tv) < 0.05) &
        (train_df['is_burned'] == 1)
    ]
    if len(obs):
        ax.scatter(obs['x'], obs['y'],
                   c='cyan', s=15, alpha=0.8,
                   marker='s', label=f'{len(obs)} fire cells')
    day = int(tv * (meta['t_max'] - meta['t_min']) + meta['t_min'])
    ax.set_title(f't={tv} (≈ Day {day})', fontsize=10)
    ax.legend(fontsize=7)

# Panel 5: β convergence
ax5 = fig.add_subplot(2, 3, 5)
ax5.plot(beta_hist, color='crimson', lw=1.2)
ax5.axhline(model.beta.item(), color='navy', ls='--', lw=1.5,
            label=f'Final β={model.beta.item():.4f}')
ax5.set_xlabel('Epoch'); ax5.set_ylabel('β')
ax5.set_title('β Convergence'); ax5.legend()
ax5.grid(True, alpha=0.3)

# Panel 6: D(x,y) overlaid with fire perimeter
ax6 = fig.add_subplot(2, 3, 6)
# Show cumulative burned area
all_fire = train_df[train_df['is_burned'] == 1]
ax6.scatter(all_fire['x'], all_fire['y'],
            c=all_fire['t'], cmap='plasma',
            s=5, alpha=0.6, label='Fire observations')
ax6.set_title('Cumulative Fire Observations\n(color = time)')
ax6.set_xlabel('x'); ax6.set_ylabel('y')
ax6.set_facecolor('#0d0d1a')
plt.colorbar(ax6.collections[0], ax=ax6, label='t (normalized)')

plt.tight_layout()
plt.savefig('wildfire_pinn_v2_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: wildfire_pinn_v2_result.png")

torch.save(model.state_dict(), 'wildfire_pinn_v2_model.pth')
pd.Series({
    'beta':          model.beta.item(),
    'D_mean':        Dfinal.mean().item(),
    'D_std':         Dfinal.std().item(),
    'D_min':         Dfinal.min().item(),
    'D_max':         Dfinal.max().item(),
    'val_mse':       val_mse,
    'val_mae':       val_mae,
    'val_mae_burn':  mae_burn,
    'val_mae_unburn':mae_unb,
}).to_csv('wildfire_results_v2.csv', header=False)
print("Results saved: wildfire_results_v2.csv")
'''