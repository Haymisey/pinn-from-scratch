# Download NASA FIRMS Burned Area Product (BA_VIIRS)
# Monthly product — shows cumulative burned area
# This is what makes D and β identifiable in Fisher-KPP

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

MAP_KEY = ""
BBOX    = "-120,49,-110,58"   # Alberta, Canada

os.makedirs("data", exist_ok=True)


# BA_VIIRS uses different date format — monthly queries
# Available: 2012-03-01 to 2026-02-01
# We want May 2023 — use 2023-05-01 with day_range=5
def download_burned_area(date, label):
    # BA_VIIRS endpoint — same structure as active fire
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
           f"{MAP_KEY}/BA_VIIRS/{BBOX}/5/{date}")

    print(f"\nDownloading burned area: {label}")
    print(f"  URL: {url}")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        if len(r.text) < 100:
            print(f"  Empty response: {r.text}")
            return None

        fname = f"data/ba_{label}.csv"
        with open(fname, "w") as f:
            f.write(r.text)
        print(f"  Saved: {fname}  ({len(r.text):,} bytes)")

        # Preview
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  Sample:\n{df.head(3).to_string()}")
        return fname

    except Exception as e:
        print(f"  Error: {e}")
        return None


# Download burned area for May 2023
# BA_VIIRS is monthly so we query around the fire period
print("="*55)
print("Downloading NASA FIRMS Burned Area Product")
print("BA_VIIRS — Monthly burned area detections")
print("="*55)

#Check transactions first
status_url = f"https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={MAP_KEY}"
r = requests.get(status_url, timeout=15)
d = r.json()
print(f"Transactions: {d['current_transactions']} / {d['transaction_limit']}")

# ownload multiple date windows to capture full burned area
dates = [
    ("2023-05-01", "may_w1"),
    ("2023-05-06", "may_w2"),
    ("2023-05-11", "may_w3"),
    ("2023-05-16", "may_w4"),
    ("2023-05-21", "may_w5"),
    ("2023-05-26", "may_w6"),
]

files = []
for date, label in dates:
    fname = download_burned_area(date, label)
    files.append(fname)
    import time
    time.sleep(1)

#Combine and process
frames = []
for fname in files:
    if fname and os.path.exists(fname):
        try:
            df = pd.read_csv(fname)
            if len(df) > 0:
                frames.append(df)
        except:
            pass

if not frames:
    print("\nNo burned area data downloaded.")
    print("BA_VIIRS may use a different query format.")
    print("Let's try the country endpoint instead...")

    #Try countrybased query
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/"
           f"{MAP_KEY}/BA_VIIRS/CAN/1/2023-05-15")
    print(f"Trying: {url}")
    r = requests.get(url, timeout=60)
    print(f"Status: {r.status_code}")
    print(f"Response preview: {r.text[:500]}")

else:
    ba_df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal burned area detections: {len(ba_df):,}")
    print(f"Columns: {list(ba_df.columns)}")
    print(f"\nFull sample:\n{ba_df.head(10).to_string()}")

    ba_df.to_csv("data/burned_area_raw.csv", index=False)
    print("\nSaved: data/burned_area_raw.csv")