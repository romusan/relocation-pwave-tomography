"""Station elevation inventory for the 2026-10-11 elevation campaign.

Combines the three local station inventories (SGC CM network, the extended
regional inventory built for the 1D Caribbean campaign, and the CARMA campaign)
into a single station-key -> elevation (km) mapping. Stations absent from all
three get 0 km and are reported as missing by attach_station_elevations.
"""
from pathlib import Path
from regional.inputs import load_station_elevations, attach_station_elevations

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = [
    ROOT / 'datos/estaciones_cm.csv',
    ROOT / 'datos/caribe_1d_estaciones_completas_2026-10-10/estaciones_todas_regionales.csv',
    ROOT / 'datos/carma_caribe_2026-10-10/estaciones_carma.csv',
]


def elevation_map():
    return load_station_elevations(INVENTORY)


def with_elevations(picks):
    return attach_station_elevations(picks, elevation_map())
