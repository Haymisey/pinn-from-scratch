# Build burned area grid from FRP observations
# Strategy: grid the full domain, assign u=0 to unburned cells
# This gives spatial contrast needed for D and β identification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

#Load all processed FRP data
df = pd.read_csv("data/firms_normalized.csv")
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['day_num']  = (df['acq_date'] - df['acq_date'].min()).dt.days

#Focus on main cluster
main = df[
    (df['day_num'] >= 4)  & (df['day_num'] <= 29) &
    (df['latitude']  >= 53.5) & (df['latitude']  <= 57.0) &
    (df['longitude'] >= -119)  & (df['longitude'] <= -113) &
    (df['confidence'].isin(['h', 'n'])) &
    (df['frp'] > 1.0)
].copy()

#normalise
main['log_frp'] = np.log10(main['frp'] + 1)
lon_min, lon_max = main['longitude'].min(), main['longitude'].max()
lat_min, lat_max = main['latitude'].min(),  main['latitude'].max()
t_min,   t_max   = main['day_num'].min(),   main['day_num'].max()
u_min,   u_max   = main['log_frp'].min(),   main['log_frp'].max()

main['x'] = (main['longitude'] - lon_min) / (lon_max - lon_min)
main['y'] = (main['latitude']  - lat_min) / (lat_max - lat_min)
main['t'] = (main['day_num']   - t_min)   / (t_max   - t_min)
main['u'] = (main['log_frp']   - u_min)   / (u_max   - u_min)

# Each cell gets either observed u or u=0 (unburned)
GRID_RES = 40
days     = sorted(main['day_num'].unique())

x_edges  = np.linspace(0, 1, GRID_RES + 1)
y_edges  = np.linspace(0, 1, GRID_RES + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
XX, YY    = np.meshgrid(x_centers, y_centers, indexing='ij')

from scipy.stats import binned_statistic_2d

records = []

for day in days:
    day_data = main[main['day_num'] == day]
    t_norm   = (day - t_min) / (t_max - t_min)

    if len(day_data) < 5:
        # Sparse day — just add zeros for the whole grid
        for i in range(GRID_RES):
            for j in range(GRID_RES):
                records.append({
                    'x': XX[i,j], 'y': YY[i,j], 't': t_norm,
                    'u': 0.0, 'is_burned': 0, 'day_num': day
                })
        continue

    # Grid the FRP observations for this day
    result = binned_statistic_2d(
        day_data['x'].values, day_data['y'].values,
        day_data['u'].values,
        statistic='mean',
        bins=[x_edges, y_edges]
    )
    grid = result.statistic  # shape (40,40), NaN where no fire

    for i in range(GRID_RES):
        for j in range(GRID_RES):
            if not np.isnan(grid[i,j]):
                # Observed fire — use actual FRP value
                records.append({
                    'x': XX[i,j], 'y': YY[i,j], 't': t_norm,
                    'u': grid[i,j], 'is_burned': 1, 'day_num': day
                })
            else:
                # No fire observed — u = 0 (unburned)
                records.append({
                    'x': XX[i,j], 'y': YY[i,j], 't': t_norm,
                    'u': 0.0, 'is_burned': 0, 'day_num': day
                })

grid_df = pd.DataFrame(records)
print(f"Total grid points: {len(grid_df):,}")
print(f"Burned cells:   {grid_df['is_burned'].sum():,} "
      f"({grid_df['is_burned'].mean()*100:.1f}%)")
print(f"Unburned cells: {(grid_df['is_burned']==0).sum():,} "
      f"({(grid_df['is_burned']==0).mean()*100:.1f}%)")
print(f"u range: [{grid_df['u'].min():.3f}, {grid_df['u'].max():.3f}]")

#train/val split
train_g = grid_df[grid_df['day_num'] <= t_max - 2]
val_g   = grid_df[grid_df['day_num'] >  t_max - 2]

train_g.to_csv("data/grid_train.csv", index=False)
val_g.to_csv("data/grid_val.csv",     index=False)
print(f"\nTrain: {len(train_g):,}  |  Val: {len(val_g):,}")
print("Saved: data/grid_train.csv, data/grid_val.csv")

#Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("Spatiotemporal Grid — Burned vs Unburned\n"
             "Full domain coverage enables D and β identification",
             fontsize=13)

for idx, day in enumerate([4, 5, 6, 10, 15, 20]):
    row, col  = divmod(idx, 3)
    day_data  = grid_df[grid_df['day_num'] == day]
    if len(day_data) == 0:
        continue

    # Reshape to grid
    u_grid = day_data.pivot_table(
        index='x', columns='y', values='u', aggfunc='mean'
    ).values

    axes[row,col].imshow(
        u_grid.T, origin='lower', cmap='hot',
        vmin=0, vmax=1, aspect='auto',
        extent=[0,1,0,1]
    )
    n_burned = (day_data['is_burned'] == 1).sum()
    axes[row,col].set_title(f"Day {day} — {n_burned} burned cells")
    axes[row,col].set_xlabel('x'); axes[row,col].set_ylabel('y')

plt.tight_layout()
plt.savefig("data/burned_grid.png", dpi=150)
plt.show()
print("Saved: data/burned_grid.png")

#Key statistics for PINN design
print("\n" + "="*55)
print("GRID STATISTICS FOR PINN")
print("="*55)
print(f"Spatial resolution: {GRID_RES}x{GRID_RES} = {GRID_RES**2} cells/day")
print(f"Temporal span:      {len(days)} days")
print(f"Total observations: {len(grid_df):,}")
print(f"Burned fraction:    {grid_df['is_burned'].mean()*100:.1f}%")
print(f"Unburned fraction:  {(1-grid_df['is_burned'].mean())*100:.1f}%")
print("\nThis gives D and β gradient signal in:")
print("  - Burned cells (u > 0): fire spread dynamics")
print("  - Unburned cells (u = 0): fire absence constraint")
print("="*55)