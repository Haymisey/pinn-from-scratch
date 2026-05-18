# Download NASA FIRMS — 2023 Canadian Wildfire
# Uses VIIRS_SNPP_SP (Standard Processing) for historical data

import pandas as pd
import requests
import matplotlib.pyplot as plt
import os

MAP_KEY = ""  # your key

#Check transaction status first
def check_transactions():
    url = (f"https://firms.modaps.eosdis.nasa.gov/mapserver/"
           f"mapkey_status/?MAP_KEY={MAP_KEY}")
    try:
        r = requests.get(url, timeout=15)
        d = r.json()
        print(f"Transactions used: {d['current_transactions']} "
              f"/ {d['transaction_limit']} "
              f"(resets every {d['transaction_interval']})")
        return d['current_transactions']
    except Exception as e:
        print(f"Status check failed: {e}")
        return -1

#verify data availability
def check_availability():
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/"
           f"data_availability/csv/{MAP_KEY}/all")
    try:
        df = pd.read_csv(url)
        print("\nAvailable datasets:")
        print(df.to_string())
        return df
    except Exception as e:
        print(f"Availability check failed: {e}")
        return None

#Download historical fire data 
def download_fire(dataset, bbox, date, days, label):
    """
    dataset: e.g. 'VIIRS_SNPP_SP' for historical
    bbox:    'west,south,east,north'
    date:    'YYYY-MM-DD'
    days:    number of days (max 10 per request)
    """
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{MAP_KEY}/{dataset}/{bbox}/{days}/{date}")

    print(f"\nDownloading: {label}")
    print(f"  URL: {url}")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        if "latitude" not in r.text[:200]:
            print(f"  Unexpected response: {r.text[:300]}")
            return None

        os.makedirs("data", exist_ok=True)
        fname = f"data/{label}.csv"
        with open(fname, "w") as f:
            f.write(r.text)
        print(f"  Saved: {fname}  ({len(r.text)} bytes)")
        return fname

    except requests.exceptions.Timeout:
        print("  Timeout — try a smaller bbox or fewer days")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

#Process and visualze
def process_and_plot(fnames, title):
    frames = []
    for fname in fnames:
        if fname and os.path.exists(fname):
            df = pd.read_csv(fname)
            frames.append(df)

    if not frames:
        print("No data to plot.")
        return None

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal detections: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['acq_date'].min()} → {df['acq_date'].max()}")
    print(f"FRP range: {df['frp'].min():.1f} → {df['frp'].max():.1f} MW")

    # Normalize FRP to [0,1] — this becomes u(x,y,t) in our PINN
    df['frp_norm'] = df['frp'] / df['frp'].max()
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['day_num']  = (df['acq_date'] - df['acq_date'].min()).dt.days

    days     = sorted(df['day_num'].unique())
    n_panels = min(len(days), 6)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(title, fontsize=13)

    for idx in range(n_panels):
        row, col  = divmod(idx, 3)
        day       = days[idx]
        day_df    = df[df['day_num'] == day]
        date_str  = day_df['acq_date'].iloc[0].strftime('%b %d')

        sc = axes[row, col].scatter(
            day_df['longitude'], day_df['latitude'],
            c=day_df['frp_norm'], cmap='hot',
            s=4, vmin=0, vmax=1, alpha=0.8
        )
        plt.colorbar(sc, ax=axes[row, col], label='FRP norm')
        axes[row, col].set_title(
            f"{date_str} — {len(day_df)} hotspots")
        axes[row, col].set_xlabel("Longitude")
        axes[row, col].set_ylabel("Latitude")
        axes[row, col].set_facecolor('#1a1a2e')

    plt.tight_layout()
    out = f"data/{title.replace(' ','_')}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved: {out}")

    df.to_csv("data/firms_processed.csv", index=False)
    print("Saved: data/firms_processed.csv")
    return df

if __name__ == "__main__":

    print("=" * 55)
    print("NASA FIRMS Data Downloader")
    print("=" * 55)

    check_transactions()
    check_availability()

    # ── 2023 Alberta Canada Wildfire ─────────────────────────
    # VIIRS_SNPP_SP covers 2012-01-20 to 2025-02-28
    # Peak fire activity: May 2023
    # Bbox: Alberta province [west, south, east, north]
    BBOX    = "-120,49,-110,58"
    DATASET = "VIIRS_SNPP_SP"   # SP = Standard Processing = historical

    # Download in 5‑day chunks, api hard limit is DAY_RANGE 1‑5
    chunks = [
        ("2023-05-01", "5", "alberta_may01"),
        ("2023-05-06", "5", "alberta_may06"),
        ("2023-05-11", "5", "alberta_may11"),
        ("2023-05-16", "5", "alberta_may16"),
        ("2023-05-21", "5", "alberta_may21"),
        ("2023-05-26", "5", "alberta_may26"),
    ]

    files = []
    for date, days, label in chunks:
        fname = download_fire(
            dataset = DATASET,
            bbox    = BBOX,
            date    = date,
            days    = days,
            label   = label
        )
        files.append(fname)
        import time
        time.sleep(2)

    df = process_and_plot(
        files,
        title="2023 Alberta Wildfire — NASA FIRMS VIIRS"
    )

    if df is not None:
        print(f"\n{'='*55}")
        print("SUCCESS — Data ready for PINN training")
        print(f"  Shape:  {df.shape}")
        print(f"  Hotspot locations saved to data/firms_processed.csv")
        print("  Next step: build PINN on this real data")
