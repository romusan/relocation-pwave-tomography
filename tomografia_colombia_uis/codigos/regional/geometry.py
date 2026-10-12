"""AEQD/WGS84 coordinates shared by association, fields and output."""
from dataclasses import dataclass
import numpy as np
from pyproj import CRS, Geod, Transformer


class Projection:
    def __init__(self, domain):
        self.lat0 = (domain['lat_min'] + domain['lat_max']) / 2
        self.lon0 = (domain['lon_min'] + domain['lon_max']) / 2
        self.crs = CRS.from_proj4(f'+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m')
        self.forward = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        self.inverse = Transformer.from_crs(self.crs, 'EPSG:4326', always_xy=True)

    def xy(self, lat, lon):
        x, y = self.forward.transform(lon, lat)
        return np.asarray(x) / 1000., np.asarray(y) / 1000.

    def latlon(self, x, y):
        lon, lat = self.inverse.transform(np.asarray(x) * 1000., np.asarray(y) * 1000.)
        return np.asarray(lat), np.asarray(lon)

    def audit(self, domain):
        lat, lon = np.meshgrid(np.linspace(domain['lat_min'], domain['lat_max'], 15),
                              np.linspace(domain['lon_min'], domain['lon_max'], 15))
        x, y = self.xy(lat.ravel(), lon.ravel())
        la, lo = self.latlon(x, y)
        rng = np.random.default_rng(20261009)
        a, b = rng.integers(0, len(x), size=(2, 3000))
        _, _, meters = Geod(ellps='WGS84').inv(lon.ravel()[a], lat.ravel()[a],
                                            lon.ravel()[b], lat.ravel()[b])
        plane = np.hypot(x[a] - x[b], y[a] - y[b])
        valid = meters > 1000
        return {'crs_wkt': self.crs.to_wkt(), 'roundtrip_max_deg': float(max(
            abs(la - lat.ravel()).max(), abs(lo - lon.ravel()).max())),
            'sampled_pairs': int(valid.sum()),
            'pair_distance_max_error_km': float(abs(plane[valid] - meters[valid] / 1000).max()),
            'pair_distance_max_relative_error_percent': float((abs(plane[valid] / (meters[valid] / 1000) - 1) * 100).max()),
            'depth_convention': 'positive down; stations at z=0 (elevation excluded explicitly)',
            'limitation': 'local Cartesian physics; Earth curvature not included in eikonal'}


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    @property
    def shape(self):
        return len(self.z), len(self.y), len(self.x)

    @property
    def spacing(self):
        return float(self.x[1] - self.x[0])

    def nodes(self, xyz):
        xyz = np.asarray(xyz)
        ijk = np.rint((xyz - [self.x[0], self.y[0], self.z[0]]) / self.spacing).astype(int)
        ijk = np.clip(ijk, 0, np.array(self.shape[::-1]) - 1)
        return ijk[..., ::-1]

    def xyz(self, kji):
        kji = np.asarray(kji)
        return np.stack([self.x[kji[..., 2]], self.y[kji[..., 1]], self.z[kji[..., 0]]], axis=-1)

    def linear(self, kji):
        k, j, i = np.asarray(kji).T
        return (k * len(self.y) + j) * len(self.x) + i


def grids(domain, projection, direct_step, inversion_step):
    ratio = inversion_step / direct_step
    if direct_step <= 0 or ratio < 1 or abs(ratio - round(ratio)) > 1e-10:
        raise ValueError('Inversion spacing must be an integer multiple of positive solver spacing')
    # Sample edges, not just corners: the projected boundary is curved.
    la = np.linspace(domain['lat_min'], domain['lat_max'], 101)
    lo = np.linspace(domain['lon_min'], domain['lon_max'], 101)
    x, y = projection.xy(np.r_[la, la, la * 0 + la[0], la * 0 + la[-1]],
                         np.r_[lo * 0 + lo[0], lo * 0 + lo[-1], lo, lo])
    # Align both grids to coarse spacing; one extra coarse cell of padding.
    bounds = [(np.floor(v.min() / inversion_step) * inversion_step - inversion_step,
               np.ceil(v.max() / inversion_step) * inversion_step + inversion_step) for v in (x, y)]
    zmax = np.ceil(domain['prof_max'] / inversion_step) * inversion_step
    def make(h):
        return Grid(np.arange(bounds[0][0], bounds[0][1] + h / 2, h),
                    np.arange(bounds[1][0], bounds[1][1] + h / 2, h),
                    np.arange(0, zmax + h / 2, h))
    return make(direct_step), make(inversion_step)


def candidate_nodes(grid, center, radius_km, domain, projection):
    r = int(np.ceil(radius_km / grid.spacing))
    offset = np.array(np.meshgrid(*([np.arange(-r, r + 1)] * 3), indexing='ij')).reshape(3, -1).T
    nodes = np.asarray(center) + offset
    nodes = nodes[np.all((nodes >= 0) & (nodes < grid.shape), axis=1)]
    xyz = grid.xyz(nodes)
    # Radius is per axis, exactly as in the reference, but expressed in km.
    inside = np.all(abs(xyz - grid.xyz(center)) <= radius_km + 1e-9, axis=1)
    lat, lon = projection.latlon(xyz[:, 0], xyz[:, 1])
    inside &= ((lat >= domain['lat_min']) & (lat <= domain['lat_max']) &
               (lon >= domain['lon_min']) & (lon <= domain['lon_max']) &
               (xyz[:, 2] >= domain['prof_min']) & (xyz[:, 2] <= domain['prof_max']))
    return nodes[inside]


def gap_deg(lat, lon, station_lat, station_lon):
    az, _, _ = Geod(ellps='WGS84').inv(np.full(len(station_lon), lon),
        np.full(len(station_lat), lat), station_lon, station_lat)
    az = np.sort(np.asarray(az) % 360)
    return float(np.diff(np.r_[az, az[0] + 360]).max()) if len(az) else 360.
