from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "figures"

GEOLOGY_MAP = FIGS / "localization_geology_mmv_map.png"
OUT = FIGS / "q1_fig1_geologic_context_and_data.png"


def main():
    events = pd.read_csv(DATA / "sgc_new_events_selected_event_summary_min4_non2700_nest300.csv")
    event_meta = (
        events[["event_id", "selection_group", "event_lat", "event_lon", "event_depth_km"]]
        .drop_duplicates("event_id")
        .copy()
    )
    reloc = pd.read_csv(DATA / "sgc_new_events_relocations_round2.csv")
    reloc = reloc.merge(event_meta[["event_id", "selection_group"]], on="event_id", how="left")
    stations = pd.read_csv(DATA / "q1_station_metadata_sgc_fdsn.csv")

    nest = event_meta["selection_group"].eq("bucaramanga_nest")
    shallow = ~nest

    fig = plt.figure(figsize=(8.0, 10.2), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.95, 1.05], hspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    img = plt.imread(GEOLOGY_MAP)
    # Keep the geologic map and scale bar, but remove the detailed stratigraphic
    # legend so the structural framework remains legible at manuscript size.
    img = img[:, : int(img.shape[1] * 0.50)]
    ax0.imshow(img)
    ax0.set_axis_off()
    ax0.text(
        0.015,
        0.975,
        "(a) Geological and structural framework",
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="0.4", boxstyle="round,pad=0.25", alpha=0.85),
    )

    ax1.scatter(
        event_meta.loc[shallow, "event_lon"],
        event_meta.loc[shallow, "event_lat"],
        s=8,
        c="0.18",
        alpha=0.42,
        linewidths=0,
        label="Selected earthquakes (20-120 km)",
        zorder=2,
    )
    ax1.scatter(
        event_meta.loc[nest, "event_lon"],
        event_meta.loc[nest, "event_lat"],
        s=14,
        facecolors="#2b83ba",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.75,
        label="Bucaramanga nest (120-170 km)",
        zorder=3,
    )
    ax1.scatter(
        stations["longitude"],
        stations["latitude"],
        marker="^",
        s=72,
        facecolors="red",
        edgecolors="black",
        linewidths=0.75,
        label="Stations",
        zorder=4,
    )
    for _, row in stations.iterrows():
        ax1.text(
            row["longitude"] + 0.018,
            row["latitude"] + 0.018,
            row["station"],
            fontsize=6.4,
            color="black",
            path_effects=[],
            zorder=5,
        )

    # Plotting domain used for the current tomography; the catalog-selection box is
    # slightly wider than the rounded plotting frame.
    lon_min, lon_max = -74.60, -72.55
    lat_min, lat_max = 6.20, 8.05
    ax1.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="#b2182b",
        lw=1.6,
        ls="--",
        label="Tomography domain",
        zorder=6,
    )
    ax1.text(-73.20, 7.53, "Bucaramanga\nnest", color="#2b83ba", fontsize=8.2, ha="center")
    ax1.set_xlim(-74.75, -72.45)
    ax1.set_ylim(6.05, 8.15)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("Longitude (deg)", fontsize=10)
    ax1.set_ylabel("Latitude (deg)", fontsize=10)
    ax1.set_title("(b) Selected earthquakes and station geometry", loc="left", fontsize=9.5, fontweight="bold")
    ax1.tick_params(labelsize=9)
    ax1.grid(True, color="0.82", lw=0.55, alpha=0.65)
    ax1.legend(loc="upper right", frameon=True, fontsize=7.8)

    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
