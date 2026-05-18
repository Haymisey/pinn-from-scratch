# Deep exploration of the Alberta 2023 FIRMS data
# Goal: understand spatial structure for PINN design

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import gaussian_kde

# ── Load ─────────────────────────────────────────────────────
df = pd.read_csv("data/firms_processed.csv")
df['acq_date'] = pd.to_datetime(df['acq_date'])
df['day_num']  = (df['acq_date'] - df['acq_date'].min()).dt.days

print(f"Total detections : {len(df):,}")
print(f"Date range       : {df['acq_date'].min()} → {df['acq_date'].max()}")
print(f"FRP range        : {df['frp'].min():.1f} → {df['frp'].max():.1f} MW")
print(f"Lat range        : {df['latitude'].min():.2f} → {df['latitude'].max():.2f}")
print(f"Lon range        : {df['longitude'].min():.2f} → {df['longitude'].max():.2f}")
print(f"\nConfidence distribution:\n{df['confidence'].value_counts()}")
print(f"\nDay/Night split:\n{df['daynight'].value_counts()}")



# FOCUS: Identify the main fire cluster for PINN
# The big fire started around lat 55-57, lon -119 to -114

# Filter to main fire cluster (Edson/Fox Creek area)
# and high-confidence detections only
main = df[
    (df['latitude']  >= 53.0) & (df['latitude']  <= 57.0) &
    (df['longitude'] >= -119) & (df['longitude'] <= -113) &
    (df['confidence'].isin(['h', 'n']))   # high + nominal confidence
].copy()

print(f"\nMain cluster detections: {len(main):,}")
print(f"Days in cluster: {sorted(main['day_num'].unique())}")


# NORMALIZE coordinates to [0,1] for PINN
# This is critical — neural networks work best in unit domain
lon_min, lon_max = main['longitude'].min(), main['longitude'].max()
lat_min, lat_max = main['latitude'].min(),  main['latitude'].max()
t_min,   t_max   = main['day_num'].min(),   main['day_num'].max()

main['x_norm'] = (main['longitude'] - lon_min) / (lon_max - lon_min)
main['y_norm'] = (main['latitude']  - lat_min) / (lat_max - lat_min)
main['t_norm'] = (main['day_num']   - t_min)   / (t_max   - t_min + 1e-8)
main['u_norm'] = main['frp'] / main['frp'].max()

print(f"\nNormalized ranges:")
print(f"  x: {main['x_norm'].min():.3f} → {main['x_norm'].max():.3f}")
print(f"  y: {main['y_norm'].min():.3f} → {main['y_norm'].max():.3f}")
print(f"  t: {main['t_norm'].min():.3f} → {main['t_norm'].max():.3f}")
print(f"  u: {main['u_norm'].min():.3f} → {main['u_norm'].max():.3f}")

# Save normalized data for PINN
main.to_csv("data/firms_normalized.csv", index=False)
print(f"\nSaved: data/firms_normalized.csv  ({len(main):,} rows)")


# VISUALIZATION — 4 scientific panels
fig = plt.figure(figsize=(16, 12))
fig.suptitle("2023 Alberta Wildfire — Data Analysis for PINN Design",
             fontsize=14, fontweight='bold')

# Panel 1: Hotspot count per day (timeline of fire intensity)
ax1 = fig.add_subplot(2, 2, 1)
daily = df.groupby('acq_date').size().reset_index(name='count')
ax1.bar(daily['acq_date'], daily['count'],
        color='orangered', alpha=0.8, edgecolor='darkred', lw=0.5)
ax1.set_xlabel("Date")
ax1.set_ylabel("Active Fire Detections")
ax1.set_title("Fire Intensity Timeline")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
ax1.grid(True, alpha=0.3)

# Mark the ignition event
ax1.axvline(pd.Timestamp('2023-05-05'), color='yellow',
            lw=2, ls='--', label='Major ignition (May 5)')
ax1.legend()

# Panel 2: FRP distribution (log scale)
ax2 = fig.add_subplot(2, 2, 2)
frp_vals = main['frp'][main['frp'] > 0]
ax2.hist(np.log10(frp_vals), bins=60,
         color='darkorange', alpha=0.8, edgecolor='black', lw=0.3)
ax2.set_xlabel("log₁₀(FRP) [MW]")
ax2.set_ylabel("Count")
ax2.set_title("Fire Radiative Power Distribution\n(main cluster, log scale)")
ax2.grid(True, alpha=0.3)

# Add percentile lines
for pct, label in [(50,'median'), (90,'90th'), (99,'99th')]:
    val = np.percentile(frp_vals, pct)
    ax2.axvline(np.log10(val), color='cyan', lw=1.5, ls=':')
    ax2.text(np.log10(val)+0.05, ax2.get_ylim()[1]*0.8,
             f'{label}\n{val:.0f}MW', fontsize=8, color='cyan')

# Panel 3: Fire spread — normalized coordinates over time
ax3 = fig.add_subplot(2, 2, 3)
days_to_show = sorted(main['day_num'].unique())[:8]
cmap = plt.cm.plasma
for i, day in enumerate(days_to_show):
    d = main[main['day_num'] == day]
    color = cmap(i / len(days_to_show))
    ax3.scatter(d['x_norm'], d['y_norm'],
                s=1, alpha=0.5, color=color,
                label=f"Day {day} ({len(d)} pts)")

ax3.set_xlabel("x (normalized longitude)")
ax3.set_ylabel("y (normalized latitude)")
ax3.set_title("Fire Spread in Normalized Domain\n(PINN training space)")
ax3.legend(fontsize=7, markerscale=5)
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_facecolor('#0d0d1a')
ax3.grid(True, alpha=0.2)

# Panel 4: Centroid trajectory (fire spread direction)
ax4 = fig.add_subplot(2, 2, 4)
centroids = main.groupby('day_num').agg(
    x_c=('longitude', 'mean'),
    y_c=('latitude',  'mean'),
    frp_mean=('frp', 'mean'),
    count=('frp', 'count')
).reset_index()

sc = ax4.scatter(centroids['x_c'], centroids['y_c'],
                 c=centroids['day_num'], cmap='plasma',
                 s=centroids['count']/50, alpha=0.9,
                 zorder=5)
# Draw trajectory arrow
for i in range(len(centroids)-1):
    ax4.annotate("",
        xy=(centroids['x_c'].iloc[i+1], centroids['y_c'].iloc[i+1]),
        xytext=(centroids['x_c'].iloc[i], centroids['y_c'].iloc[i]),
        arrowprops=dict(arrowstyle='->', color='white', lw=1.5)
    )
plt.colorbar(sc, ax=ax4, label='Day number')
ax4.set_xlabel("Longitude")
ax4.set_ylabel("Latitude")
ax4.set_title("Fire Centroid Trajectory\n(size = detection count)")
ax4.set_facecolor('#0d0d1a')
ax4.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("data/firms_analysis.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved: data/firms_analysis.png")

# PINN design
print("\n" + "="*55)
print("PINN DESIGN")
print("="*55)
print(f"Domain:      x∈[{lon_min:.2f}, {lon_max:.2f}]  "
      f"y∈[{lat_min:.2f}, {lat_max:.2f}]")
print(f"Time:        {t_max - t_min + 1} days")
print(f"Sensor pts:  {len(main):,} satellite observations")
print(f"PDE:         ∂u/∂t = ∇·[D(x,y)∇u] + βu(1-u)")
print(f"Goal:        Recover D(x,y) field from FRP observations")
print(f"Novelty:     Spatially-varying D — never done with FIRMS data")