"""
OpenStreetMap Data Quality Assessment: Hyderabad POIs
Measures completeness, consistency (duplicates), and freshness.

Setup:  pip install requests pandas matplotlib
Run:    python osm_quality_hyderabad.py
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
BBOX = "17.25,78.30,17.55,78.60"  # south, west, north, east (Hyderabad)
AMENITIES = "restaurant|cafe|hospital|pharmacy|bank|school"

QUERY = f"""
[out:json][timeout:180];
nwr["amenity"~"^({AMENITIES})$"]({BBOX});
out center meta;
"""


def fetch():
    resp = requests.post(
        OVERPASS_URL,
        data={"data": QUERY},
        headers={
            "User-Agent": "osm-quality-hyderabad/1.0 (Python requests)",
            "Accept": "application/json",
        },
        timeout=200,
    )
    if not resp.ok:
        detail = resp.text.strip()[:500] or "<empty response body>"
        raise requests.HTTPError(
            f"Overpass returned HTTP {resp.status_code} {resp.reason}: {detail}",
            response=resp,
        )
    return resp.json()["elements"]


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
    print("Fetching from Overpass (can take a minute)...")
    df = to_dataframe(fetch())
    print(f"Fetched {len(df)} features")

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