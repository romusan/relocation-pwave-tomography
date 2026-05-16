from __future__ import annotations

import argparse
import json
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pandas as pd
import requests
from obspy import read, read_events
from obspy.core.utcdatetime import UTCDateTime


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output_V2"
DEFAULT_CATALOG_CSV = (
    OUTPUT_DIR
    / "data"
    / "sgc_kml_2022_2026_rms_lt1_depth_gt20"
    / "sgc_sismos_area_2022_2026_depth_gt20_rms_lt1_strict.csv"
)
DEFAULT_STATIONS_CSV = OUTPUT_DIR / "data" / "domain_stations_v2.csv"
DEFAULT_OUT_DIR = OUTPUT_DIR / "data" / "sgc_waveforms_q_spectral"
CODE_DIR = OUTPUT_DIR / "codigos_finales"

DATASELECT_URL = "https://sismo.sgc.gov.co:8443/fdsnws/dataselect/1/query"
DEFAULT_PRE_P_SECONDS = 30.0
DEFAULT_POST_P_SECONDS = 180.0
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass
class DownloadConfig:
    catalog_csv: Path
    stations_csv: Path
    out_dir: Path
    dataselect_url: str
    pre_p_seconds: float
    post_p_seconds: float
    timeout_seconds: int
    retries: int
    sleep_seconds: float
    overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download miniSEED windows from SGC for the filtered earthquake catalog. "
            "Only P picks at the existing tomography stations are requested."
        )
    )
    parser.add_argument("--catalog-csv", type=Path, default=DEFAULT_CATALOG_CSV)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataselect-url", default=DATASELECT_URL)
    parser.add_argument("--pre-p", type=float, default=DEFAULT_PRE_P_SECONDS)
    parser.add_argument("--post-p", type=float, default=DEFAULT_POST_P_SECONDS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--event-id", default=None, help="Download one event id only.")
    parser.add_argument("--min-magnitude", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_token(value: str | int | float | None) -> str:
    text = "" if value is None else str(value)
    text = text.strip() or "NA"
    for char in "\\/:*?\"<>| ":
        text = text.replace(char, "_")
    return text


def request_with_retries(
    session: requests.Session,
    url: str,
    timeout: int,
    retries: int,
    sleep_seconds: float,
) -> tuple[int, bytes, str]:
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            return response.status_code, response.content, ""
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep_seconds * (attempt + 1))
    return 0, b"", last_error


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if local_name(child) == name:
            return child
    return None


def child_text(element: ET.Element | None, *path: str) -> str | None:
    current = element
    for name in path:
        if current is None:
            return None
        current = first_child(current, name)
    if current is None or current.text is None:
        return None
    return current.text.strip()


def parse_float(text: str | None) -> float | None:
    if text in (None, ""):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_sgc_seiscomp_xml(xml_path: Path):
    """Fallback parser for SGC SeisComP XML files that ObsPy may not map to events."""
    root = ET.parse(xml_path).getroot()

    picks = []
    for pick_element in root.iter():
        if local_name(pick_element) != "pick":
            continue
        waveform = first_child(pick_element, "waveformID")
        if waveform is None:
            continue
        pick_time_text = child_text(pick_element, "time", "value")
        if not pick_time_text:
            continue
        picks.append(
            SimpleNamespace(
                time=UTCDateTime(pick_time_text),
                phase_hint=child_text(pick_element, "phaseHint") or "",
                waveform_id=SimpleNamespace(
                    network_code=waveform.attrib.get("networkCode", ""),
                    station_code=waveform.attrib.get("stationCode", ""),
                    location_code=waveform.attrib.get("locationCode", ""),
                    channel_code=waveform.attrib.get("channelCode", ""),
                ),
            )
        )

    origins = [element for element in root.iter() if local_name(element) == "origin"]
    events = [element for element in root.iter() if local_name(element) == "event"]
    preferred_origin_id = None
    preferred_magnitude_id = None
    if events:
        preferred_origin_id = child_text(events[0], "preferredOriginID")
        preferred_magnitude_id = child_text(events[0], "preferredMagnitudeID")

    origin_element = None
    if preferred_origin_id:
        origin_element = next(
            (item for item in origins if item.attrib.get("publicID") == preferred_origin_id),
            None,
        )
    if origin_element is None and origins:
        origin_element = origins[0]
    if origin_element is None:
        raise ValueError("No origin found in SGC SeisComP XML.")

    origin_time = child_text(origin_element, "time", "value")
    latitude = parse_float(child_text(origin_element, "latitude", "value"))
    longitude = parse_float(child_text(origin_element, "longitude", "value"))
    depth = parse_float(child_text(origin_element, "depth", "value"))
    origin = SimpleNamespace(
        time=UTCDateTime(origin_time) if origin_time else None,
        latitude=latitude,
        longitude=longitude,
        depth=depth,
    )

    magnitude_elements = [
        element
        for element in root.iter()
        if local_name(element) == "magnitude" and element.attrib.get("publicID")
    ]
    magnitude_element = None
    if preferred_magnitude_id:
        magnitude_element = next(
            (
                item
                for item in magnitude_elements
                if item.attrib.get("publicID") == preferred_magnitude_id
            ),
            None,
        )
    if magnitude_element is None and magnitude_elements:
        magnitude_element = magnitude_elements[0]
    magnitude = None
    if magnitude_element is not None:
        magnitude = SimpleNamespace(
            mag=parse_float(child_text(magnitude_element, "magnitude", "value")),
            magnitude_type=child_text(magnitude_element, "type"),
        )

    event = SimpleNamespace(
        picks=picks,
        origins=[origin],
        magnitudes=[magnitude] if magnitude is not None else [],
        preferred_origin=lambda: origin,
        preferred_magnitude=lambda: magnitude,
    )
    return event, origin, magnitude


def parse_event_xml(xml_path: Path):
    try:
        catalog = read_events(str(xml_path))
        if len(catalog) == 0:
            raise IndexError("ObsPy returned an empty catalog.")
        event = catalog[0]
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude()
        if magnitude is None and event.magnitudes:
            magnitude = event.magnitudes[0]
        return event, origin, magnitude
    except Exception:
        return parse_sgc_seiscomp_xml(xml_path)


def first_domain_p_picks(event, domain_stations: set[str]) -> list[dict]:
    picks = []
    seen: set[str] = set()
    for pick in sorted(event.picks, key=lambda item: item.time):
        wid = pick.waveform_id
        station = (wid.station_code or "").strip()
        phase = (pick.phase_hint or "").upper().strip()
        channel = (wid.channel_code or "").strip()
        if station not in domain_stations:
            continue
        if station in seen:
            continue
        if not phase.startswith("P"):
            continue
        if not channel.endswith("Z"):
            continue
        picks.append(
            {
                "network": (wid.network_code or "CM").strip() or "CM",
                "station": station,
                "location": (wid.location_code or "*").strip() or "*",
                "channel": channel or "*Z",
                "phase": phase,
                "pick_time": pick.time,
            }
        )
        seen.add(station)
    return picks


def dataselect_url(config: DownloadConfig, pick: dict, start: UTCDateTime, end: UTCDateTime) -> str:
    params = {
        "net": pick["network"],
        "sta": pick["station"],
        "loc": pick["location"],
        "cha": pick["channel"],
        "starttime": start.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "endtime": end.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "format": "miniseed",
        "nodata": "204",
    }
    return f"{config.dataselect_url}?{urlencode(params)}"


def read_mseed_metadata(mseed_path: Path) -> dict:
    stream = read(str(mseed_path))
    trace_ids = []
    sampling_rates = []
    npts = []
    starts = []
    ends = []
    for trace in stream:
        trace_ids.append(trace.id)
        sampling_rates.append(float(trace.stats.sampling_rate))
        npts.append(int(trace.stats.npts))
        starts.append(str(trace.stats.starttime))
        ends.append(str(trace.stats.endtime))
    return {
        "n_traces": len(stream),
        "trace_ids": ";".join(trace_ids),
        "sampling_rates_hz": ";".join(str(item) for item in sampling_rates),
        "npts": ";".join(str(item) for item in npts),
        "trace_start_utc": ";".join(starts),
        "trace_end_utc": ";".join(ends),
    }


def load_catalog(args: argparse.Namespace) -> pd.DataFrame:
    catalog = pd.read_csv(args.catalog_csv)
    catalog["time_utc"] = pd.to_datetime(catalog["time_utc"], errors="coerce")
    catalog = catalog.sort_values(["time_utc", "event_id"]).reset_index(drop=True)
    if args.event_id:
        catalog = catalog.loc[catalog["event_id"].astype(str) == args.event_id].copy()
    if args.min_magnitude is not None:
        catalog = catalog.loc[catalog["magnitude"] >= args.min_magnitude].copy()
    if args.start_index:
        catalog = catalog.iloc[args.start_index :].copy()
    if args.max_events is not None:
        catalog = catalog.iloc[: args.max_events].copy()
    return catalog.reset_index(drop=True)


def write_csv_incremental(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", index=False, header=not path.exists(), encoding="utf-8")


def event_row_metadata(row: pd.Series, origin, magnitude) -> dict:
    depth_km = None
    if getattr(origin, "depth", None) is not None:
        depth_km = origin.depth / 1000 if abs(origin.depth) > 1000 else origin.depth
    return {
        "event_id": row["event_id"],
        "origin_time_utc": str(origin.time),
        "catalog_time_utc": str(row.get("time_utc", "")),
        "event_lat": origin.latitude,
        "event_lon": origin.longitude,
        "event_depth_km": depth_km if depth_km is not None else row.get("depth_km", None),
        "catalog_depth_km": row.get("depth_km", None),
        "magnitude": magnitude.mag if magnitude is not None else row.get("magnitude", None),
        "magnitude_type": magnitude.magnitude_type if magnitude is not None else row.get("magnitude_type", None),
        "catalog_phases": row.get("phases", None),
        "catalog_rms_s": row.get("rms_s", None),
        "region": row.get("region", ""),
    }


def download_event(
    session: requests.Session,
    config: DownloadConfig,
    row: pd.Series,
    domain_stations: set[str],
    manifest_csv: Path,
    event_summary_csv: Path,
) -> dict:
    event_id = str(row["event_id"])
    xml_dir = config.out_dir / "xml"
    mseed_dir = config.out_dir / "mseed_by_event" / event_id
    xml_dir.mkdir(parents=True, exist_ok=True)
    mseed_dir.mkdir(parents=True, exist_ok=True)

    xml_path = xml_dir / f"{event_id}.xml"
    xml_url = str(row["quakeml_url"])
    xml_status = None
    xml_bytes = xml_path.stat().st_size if xml_path.exists() else 0
    xml_error = ""
    if config.overwrite or not xml_path.exists() or xml_bytes == 0:
        xml_status, content, xml_error = request_with_retries(
            session,
            xml_url,
            config.timeout_seconds,
            config.retries,
            config.sleep_seconds,
        )
        xml_path.write_bytes(content)
        xml_bytes = len(content)
    else:
        xml_status = "cached"

    try:
        event, origin, magnitude = parse_event_xml(xml_path)
    except Exception as exc:
        origin = SimpleNamespace(
            time=row.get("time_utc", ""),
            latitude=row.get("lat", None),
            longitude=row.get("lon", None),
            depth=row.get("depth_km", None),
        )
        magnitude = SimpleNamespace(
            mag=row.get("magnitude", None),
            magnitude_type=row.get("magnitude_type", None),
        )
        event_meta = event_row_metadata(row, origin, magnitude)
        summary = {
            **event_meta,
            "event_id": event_id,
            "status": "XML_PARSE_ERROR",
            "xml_status": xml_status,
            "xml_bytes": xml_bytes,
            "xml_error": xml_error,
            "error": f"{type(exc).__name__}: {exc}",
            "p_picks_in_domain": 0,
            "successful_station_downloads": 0,
        }
        write_csv_incremental(event_summary_csv, summary)
        return summary

    event_meta = event_row_metadata(row, origin, magnitude)
    picks = first_domain_p_picks(event, domain_stations)
    if not picks:
        summary = {
            **event_meta,
            "status": "NO_DOMAIN_P_PICKS",
            "xml_status": xml_status,
            "xml_bytes": xml_bytes,
            "p_picks_in_domain": 0,
            "successful_station_downloads": 0,
            "failed_station_downloads": 0,
        }
        write_csv_incremental(event_summary_csv, summary)
        return summary

    success_count = 0
    failure_count = 0
    total_download_bytes = 0
    for pick in picks:
        start = pick["pick_time"] - config.pre_p_seconds
        end = pick["pick_time"] + config.post_p_seconds
        loc_token = safe_token(pick["location"])
        mseed_name = (
            f"{event_id}_{safe_token(pick['network'])}_{safe_token(pick['station'])}_"
            f"{loc_token}_{safe_token(pick['channel'])}.mseed"
        )
        mseed_path = mseed_dir / mseed_name
        url = dataselect_url(config, pick, start, end)

        http_status = None
        download_bytes = mseed_path.stat().st_size if mseed_path.exists() else 0
        request_error = ""
        if config.overwrite or not mseed_path.exists() or download_bytes == 0:
            http_status, content, request_error = request_with_retries(
                session,
                url,
                config.timeout_seconds,
                config.retries,
                config.sleep_seconds,
            )
            mseed_path.write_bytes(content)
            download_bytes = len(content)
            time.sleep(config.sleep_seconds)
        else:
            http_status = "cached"

        total_download_bytes += int(download_bytes)
        read_error = ""
        metadata = {
            "n_traces": 0,
            "trace_ids": "",
            "sampling_rates_hz": "",
            "npts": "",
            "trace_start_utc": "",
            "trace_end_utc": "",
        }
        if download_bytes > 0 and http_status not in (204, "204"):
            try:
                metadata = read_mseed_metadata(mseed_path)
            except Exception as exc:
                read_error = f"{type(exc).__name__}: {exc}"

        if metadata["n_traces"]:
            success_count += 1
            status = "OK"
        else:
            failure_count += 1
            status = "NO_READABLE_TRACE"
            if http_status in (204, "204"):
                status = "NODATA_204"
            elif request_error:
                status = "REQUEST_ERROR"
            elif read_error:
                status = "READ_ERROR"

        manifest_row = {
            **event_meta,
            "status": status,
            "network": pick["network"],
            "station": pick["station"],
            "location": pick["location"],
            "channel": pick["channel"],
            "phase": pick["phase"],
            "pick_time_utc": str(pick["pick_time"]),
            "window_start_utc": str(start),
            "window_end_utc": str(end),
            "http_status": http_status,
            "download_bytes": download_bytes,
            **metadata,
            "mseed_file": str(mseed_path),
            "xml_file": str(xml_path),
            "quakeml_url": xml_url,
            "dataselect_url": url,
            "request_error": request_error,
            "read_error": read_error,
        }
        write_csv_incremental(manifest_csv, manifest_row)

    summary = {
        **event_meta,
        "status": "DONE",
        "xml_status": xml_status,
        "xml_bytes": xml_bytes,
        "p_picks_in_domain": len(picks),
        "successful_station_downloads": success_count,
        "failed_station_downloads": failure_count,
        "download_bytes": total_download_bytes,
    }
    write_csv_incremental(event_summary_csv, summary)
    return summary


def build_success_inventory(manifest_csv: Path, inventory_csv: Path) -> int:
    if not manifest_csv.exists():
        pd.DataFrame().to_csv(inventory_csv, index=False)
        return 0
    manifest = pd.read_csv(manifest_csv, dtype=str)
    if manifest.empty or "status" not in manifest:
        manifest.to_csv(inventory_csv, index=False)
        return 0
    inventory = manifest.loc[manifest["status"] == "OK"].copy()
    dedup_columns = [
        "event_id",
        "network",
        "station",
        "location",
        "channel",
        "pick_time_utc",
    ]
    present = [column for column in dedup_columns if column in inventory.columns]
    if present:
        inventory = inventory.drop_duplicates(subset=present, keep="last")
    inventory.to_csv(inventory_csv, index=False, encoding="utf-8")
    return len(inventory)


def main() -> None:
    args = parse_args()
    config = DownloadConfig(
        catalog_csv=args.catalog_csv,
        stations_csv=args.stations_csv,
        out_dir=args.out_dir,
        dataselect_url=args.dataselect_url,
        pre_p_seconds=args.pre_p,
        post_p_seconds=args.post_p,
        timeout_seconds=args.timeout,
        retries=args.retries,
        sleep_seconds=args.sleep,
        overwrite=args.overwrite,
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    CODE_DIR.mkdir(parents=True, exist_ok=True)
    logs_dir = config.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = logs_dir / "waveform_download_manifest.csv"
    event_summary_csv = logs_dir / "event_download_summary.csv"
    inventory_csv = logs_dir / "spectral_q_waveform_inventory.csv"
    run_summary_json = logs_dir / "waveform_download_run_summary.json"

    catalog = load_catalog(args)
    stations = pd.read_csv(config.stations_csv)
    domain_stations = set(stations["station"].astype(str).str.strip())

    run_info = {
        "catalog_csv": str(config.catalog_csv),
        "stations_csv": str(config.stations_csv),
        "out_dir": str(config.out_dir),
        "dataselect_url": config.dataselect_url,
        "pre_p_seconds": config.pre_p_seconds,
        "post_p_seconds": config.post_p_seconds,
        "domain_stations": sorted(domain_stations),
        "selected_events": len(catalog),
        "start_index": args.start_index,
        "max_events": args.max_events,
        "event_id": args.event_id,
        "min_magnitude": args.min_magnitude,
        "overwrite": args.overwrite,
    }
    (logs_dir / "last_run_parameters.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    totals = {
        "events_processed": 0,
        "events_with_domain_p_picks": 0,
        "successful_station_downloads": 0,
        "failed_station_downloads": 0,
        "xml_parse_errors": 0,
        "events_without_domain_p_picks": 0,
    }
    session = requests.Session()
    session.headers.update({"User-Agent": "MMV-Q-tomography-waveform-downloader/1.0"})

    for i, (_, row) in enumerate(catalog.iterrows(), start=1):
        event_id = row["event_id"]
        print(f"[{i}/{len(catalog)}] {event_id} {row.get('time_utc', '')}", flush=True)
        summary = download_event(
            session,
            config,
            row,
            domain_stations,
            manifest_csv,
            event_summary_csv,
        )
        totals["events_processed"] += 1
        if summary["status"] == "XML_PARSE_ERROR":
            totals["xml_parse_errors"] += 1
        elif summary["status"] == "NO_DOMAIN_P_PICKS":
            totals["events_without_domain_p_picks"] += 1
        if int(summary.get("p_picks_in_domain", 0) or 0) > 0:
            totals["events_with_domain_p_picks"] += 1
        totals["successful_station_downloads"] += int(
            summary.get("successful_station_downloads", 0) or 0
        )
        totals["failed_station_downloads"] += int(summary.get("failed_station_downloads", 0) or 0)

    inventory_rows = build_success_inventory(manifest_csv, inventory_csv)
    final_summary = {
        **run_info,
        **totals,
        "inventory_rows_ok": inventory_rows,
        "manifest_csv": str(manifest_csv),
        "event_summary_csv": str(event_summary_csv),
        "inventory_csv": str(inventory_csv),
        "notes": [
            "The manifest contains one row per event-station P-pick request.",
            "The inventory contains only readable miniSEED traces and is the starting table for spectral-ratio Q processing.",
            "The script is resumable: rerunning it skips existing non-empty XML and miniSEED files unless --overwrite is used.",
        ],
    }
    run_summary_json.write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), CODE_DIR / Path(__file__).name)

    print(json.dumps(final_summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
