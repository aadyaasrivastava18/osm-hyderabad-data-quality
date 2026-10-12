"""
OpenStreetMap Data Quality Assessment: Hyderabad POIs
Measures completeness, consistency (duplicates), and freshness.

Setup:  pip install requests pandas matplotlib
Run:    python osm_quality_hyderabad.py

Notes:
- Hyderabad is split into a GRID x GRID set of small tiles so each Overpass
  request is light. Each tile is cached in ./cache so reruns resume where
  they stopped.
- IMPORTANT: put your real email in USER_AGENT below (Overpass asks for it).
"""
import json
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import requests

USER_AGENT = "osm-hyderabad-data-quality/1.0 (learning project; contact: your-email@example.com)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# south, west, north, east (Hyderabad)
BBOX = (17.25, 78.30, 17.55, 78.60)
GRID = 4
AMENITIES = "restaurant|cafe|hospital|pharmacy|bank|school"
CACHE_DIR = "cache"
MAX_ATTEMPTS = 6


def tiles():
    s, w, n, e = BBOX
    lat_step = (n - s) / GRID
    lon_step = (e - w) / GRID
    for i in range(GRID):
        for j in range(GRID):
            yield (
                i * GRID + j,
                (s + i * lat_step, w + j * lon_step,
                 s + (i + 1) * lat_step, w + (j + 1) * lon_step),
            )


def fetch_tile(idx, bbox):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"tile_{idx}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    s, w, n, e = bbox
    query = (
        f'[out:json][timeout:90];'
        f'nwr["amenity"~"^({AMENITIES})$"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});'
        f'out center meta;'
    )

    for attempt in range(MAX_ATTEMPTS):
        url = MIRRORS[attempt % len(MIRRORS)]
        try:
            resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=120)
            if resp.status_code == 200:
                elements = resp.json()["elements"]
                with open(cache_path, "w") as f:
                    json.dump(elements, f)
                return elements
            print(f"  tile {idx}: HTTP {resp.status_code} from {url.split('/')[2]}")
        except (requests.RequestException, ValueError) as err:
            print(f"  tile {idx}: {type(err).__name__} from {url.split('/')[2]}")
        wait = 10 * (attempt + 1)
        print(f"  retrying in {wait}s (attempt {attempt + 1}/{MAX_ATTEMPTS})")
        time.sleep(wait)
    return None


def fetch():
    all_elements, failed = {}, []
    for idx, bbox in tiles():
        print(f"Tile {idx + 1}/{GRID * GRID}...")
        elements = fetch_tile(idx, bbox)
        if elements is None:
            failed.append(idx)
            continue
        for el in elements:  # dedupe features that span tile edges
            all_elements[(el["type"], el["id"])] = el
        time.sleep(2)  # be polite to the free server
    if failed:
        print(f"\nWARNING: tiles {failed} failed. Rerun the script to retry only those.")
    if not all_elements:
        raise SystemExit("No data fetched. Wait a few minutes and rerun.")
    return list(all_elements.values())


def to_dataframe(elements):
    rows = []
    for e in elements:
        tags = e.get("tags", {})
        lat = e.get("lat") or e.get("center", {}).get("lat")
        lon = e.get("lon") or e.get("center", {}).get("lon")
        rows.append({
            "osm_id": f"{e['type']}/{e['id']}",
            "amenity": tags.get("amenity"),
            "name": tags.get("name"),
            "opening_hours": tags.get("opening_hours"),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "website": tags.get("website") or tags.get("contact:website"),
            "lat": lat,
            "lon": lon,
            "last_edit": pd.to_datetime(e.get("timestamp"), utc=True),
            "version": e.get("version"),
        })
    return pd.DataFrame(rows)


def completeness(df):
    fields = ["name", "opening_hours", "phone", "website"]
    out = df.groupby("amenity")[fields].agg(lambda s: round(s.notna().mean() * 100, 1))
    out["overall_complete_%"] = out.mean(axis=1).round(1)
    return out


def consistency(df):
    """Flag likely duplicates: same name + amenity within ~100m grid cell."""
    named = df.dropna(subset=["name", "lat", "lon"]).copy()
    named["name_key"] = named["name"].str.lower().str.strip()
    named["cell"] = list(zip(named["lat"].round(3), named["lon"].round(3)))
    dup = named.duplicated(subset=["name_key", "amenity", "cell"], keep=False)
    return named[dup].sort_values("name_key"), round(dup.sum() / len(df) * 100, 2)


def freshness(df):
    now = pd.Timestamp.now(tz="UTC")
    df = df.copy()
    df["age_days"] = (now - df["last_edit"]).dt.days
    summary = df.groupby("amenity")["age_days"].agg(
        median_age_days="median",
        pct_stale_2y=lambda s: round((s > 730).mean() * 100, 1),
    )
    return summary.round(1)


def plot(comp):
    ax = comp["overall_complete_%"].sort_values().plot(kind="barh", figsize=(8, 4))
    ax.set_xlabel("Average attribute completeness (%)")
    ax.set_title("OSM Hyderabad: POI completeness by type")
    plt.tight_layout()
    plt.savefig("completeness.png", dpi=150)


if __name__ == "__main__":
    print("Fetching from Overpass in tiles (a few minutes on first run)...")
    df = to_dataframe(fetch())
    print(f"\nFetched {len(df)} features")

    comp = completeness(df)
    dups, dup_pct = consistency(df)
    fresh = freshness(df)

    print("\n--- COMPLETENESS (%) ---\n", comp)
    print(f"\n--- CONSISTENCY ---\nPossible duplicates: {dup_pct}% of features")
    print("\n--- FRESHNESS ---\n", fresh)

    df.to_csv("hyderabad_pois_raw.csv", index=False)
    comp.to_csv("completeness_summary.csv")
    fresh.to_csv("freshness_summary.csv")
    dups.to_csv("possible_duplicates.csv", index=False)
    plot(comp)
    print("\nSaved CSVs and completeness.png")