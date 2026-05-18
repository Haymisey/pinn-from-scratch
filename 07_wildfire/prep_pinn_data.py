# ============================================================
# Prepare FIRMS data for PINN training
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
import torch

np.random.seed(42)

# ── Load ─────────────────────────────────────────────────────
df = pd.read_csv("data/firms_normalized.csv")
print(f"Loaded: {len(df):,} detections")

# ── Focus on main fire cluster ────────────────────────────────
main = df[
    (df['day_num'] >= 4)  & (df['day_num'] <= 29) &
    (df['latitude']  >= 53.5) & (df['latitude']  <= 57.0) &
    (df['longitude'] >= -119)  & (df['longitude'] <= -113) &
    (df['confidence'].isin(['h', 'n'])) &
    (df['frp'] > 1.0)
].copy()

print(f"Main cluster: {len(main):,} detections")

# ── Use log(FRP) as u ─────────────────────────────────────────
main['log_frp'] = np.log10(main['frp'] + 1)

lon_min, lon_max = main['longitude'].min(), main['longitude'].max()
lat_min, lat_max = main['latitude'].min(),  main['latitude'].max()
t_min,   t_max   = main['day_num'].min(),   main['day_num'].max()
u_min,   u_max   = main['log_frp'].min(),   main['log_frp'].max()

main['x'] = (main['longitude'] - lon_min) / (lon_max - lon_min)
main['y'] = (main['latitude']  - lat_min) / (lat_max - lat_min)
main['t'] = (main['day_num']   - t_min)   / (t_max   - t_min)
main['u'] = (main['log_frp']   - u_min)   / (u_max   - u_min)

print(f"\nDomain:")
print(f"  Lon: [{lon_min:.3f}, {lon_max:.3f}] → x: [0, 1]")
print(f"  Lat: [{lat_min:.3f}, {lat_max:.3f}] → y: [0, 1]")
print(f"  Day: [{t_min}, {t_max}]             → t: [0, 1]")
print(f"  log(FRP): [{u_min:.3f}, {u_max:.3f}] → u: [0, 1]")

# ── Stratified subsampling ────────────────────────────────────
N_SENSORS = 5000
n_per_day = N_SENSORS // len(main['day_num'].unique())

sampled = []
for day in sorted(main['day_num'].unique()):
    day_df  = main[main['day_num'] == day]
    n       = min(len(day_df), n_per_day)
    weights = (day_df['frp'] + 1e-6)
    weights = weights / weights.sum()
    sample  = day_df.sample(n=n, weights=weights, replace=True, random_state=42)
    sampled.append(sample)

sensors = pd.concat(sampled, ignore_index=True)
print(f"\nSampled sensors: {len(sensors):,}")

# ── Train/validation split ────────────────────────────────────
train = sensors[sensors['day_num'] <= t_max - 2]
val   = sensors[sensors['day_num'] >  t_max - 2]
print(f"Train: {len(train):,}  |  Val: {len(val):,}")

train.to_csv("data/pinn_train.csv", index=False)
val.to_csv("data/pinn_val.csv",     index=False)

# ── Save normalization constants ──────────────────────────────
meta = {
    'lon_min': lon_min, 'lon_max': lon_max,
    'lat_min': lat_min, 'lat_max': lat_max,
    't_min':   t_min,   't_max':   t_max,
    'u_min':   u_min,   'u_max':   u_max,
    'n_days':  t_max - t_min + 1
}
pd.Series(meta).to_csv("data/normalization.csv", header=False)
print("Saved: data/normalization.csv")

# ════════════════════════════════════════════════════════════
# SAVE INITIAL CONDITION DATA (day 4 — fire just starting)
# ════════════════════════════════════════════════════════════
day4 = main[main['day_num'] == 4][['x', 'y', 'u']].copy()
day4.to_csv("data/initial_condition.csv", index=False)
print(f"Saved: data/initial_condition.csv  ({len(day4)} points)")

# ════════════════════════════════════════════════════════════
# TEMPORAL DERIVATIVE SUPERVISION
# Compute ∂u/∂t numerically from consecutive daily observations
# ════════════════════════════════════════════════════════════
print("\nComputing temporal derivatives...")

GRID_RES = 20
days = sorted(main['day_num'].unique())

x_edges = np.linspace(0, 1, GRID_RES + 1)
y_edges = np.linspace(0, 1, GRID_RES + 1)

daily_grids = {}
for day in days:
    day_data = main[main['day_num'] == day]
    if len(day_data) < 10:
        continue
    result = binned_statistic_2d(
        day_data['x'].values, day_data['y'].values,
        day_data['u'].values, statistic='mean',
        bins=[x_edges, y_edges]
    )
    daily_grids[day] = result.statistic

print(f"  Gridded days: {list(daily_grids.keys())}")

t_range    = main['day_num'].max() - main['day_num'].min()
day_list   = sorted(daily_grids.keys())
deriv_records = []

for i in range(len(day_list) - 1):
    d1, d2 = day_list[i], day_list[i+1]
    if d2 - d1 > 3:
        continue

    g1 = daily_grids[d1]
    g2 = daily_grids[d2]
    dt = (d2 - d1) / t_range

    valid = ~np.isnan(g1) & ~np.isnan(g2)
    if valid.sum() < 5:
        continue

    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    XX, YY    = np.meshgrid(x_centers, y_centers, indexing='ij')

    dudt  = (g2 - g1) / dt
    t_mid = ((d1 + d2) / 2 - main['day_num'].min()) / t_range

    xi  = XX[valid].flatten()
    yi  = YY[valid].flatten()
    ui  = g1[valid].flatten()
    dui = dudt[valid].flatten()

    for j in range(len(xi)):
        deriv_records.append({
            'x': xi[j], 'y': yi[j],
            't': t_mid,  'u': ui[j],
            'dudt': dui[j]
        })

deriv_df = pd.DataFrame(deriv_records)
deriv_df.to_csv("data/temporal_derivatives.csv", index=False)
print(f"  Temporal derivative points: {len(deriv_df):,}")
print(f"  dudt range: [{deriv_df['dudt'].min():.4f}, {deriv_df['dudt'].max():.4f}]")

# ── Visualization ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("PINN Training Data — 2023 Alberta Wildfire", fontsize=13)

sc = axes[0].scatter(train['x'], train['y'], c=train['t'],
                     cmap='plasma', s=3, alpha=0.7)
plt.colorbar(sc, ax=axes[0], label='t (normalized time)')
axes[0].set_title(f'Training Sensors ({len(train):,} pts)')
axes[0].set_facecolor('#0d0d1a')

sc2 = axes[1].scatter(train['x'], train['y'], c=train['u'],
                      cmap='hot', s=3, alpha=0.7, vmin=0, vmax=1)
plt.colorbar(sc2, ax=axes[1], label='u = log(FRP) normalized')
axes[1].set_title('Fire Intensity (PINN target u)')
axes[1].set_facecolor('#0d0d1a')

daily_u = train.groupby('day_num')['u'].agg(['mean', 'std'])
axes[2].fill_between(daily_u.index,
                     daily_u['mean'] - daily_u['std'],
                     daily_u['mean'] + daily_u['std'],
                     alpha=0.3, color='orangered')
axes[2].plot(daily_u.index, daily_u['mean'], 'o-', color='orangered', lw=2)
axes[2].set_title('Fire Intensity Over Time')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/pinn_training_data.png", dpi=150)
plt.show()
print("Saved: data/pinn_training_data.png")

print("\n" + "="*55)
print("READY FOR PINN TRAINING")
print(f"  Train: {len(train):,} | Val: {len(val):,}")
print(f"  IC points (day 4): {len(day4)}")
print(f"  Derivative points: {len(deriv_df):,}")
print("="*55)
