"""Explicit experimental FSM fields and station/model-specific caching."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
from .fsm_reference import fast_sweeping_3d


class FieldCache:
    def __init__(self, vp, grid, stations, directory, config):
        self.vp = np.ascontiguousarray(vp, dtype=np.float64)
        self.grid, self.stations, self.config = grid, stations, config
        if self.vp.shape != grid.shape or not np.isfinite(vp).all() or np.min(vp) <= 0:
            raise ValueError('Finite positive velocity matching the grid is required')
        solver_hash = hashlib.sha256(Path(__file__).with_name('fsm_reference.py').read_bytes()).hexdigest()
        content = hashlib.sha256(vp.tobytes())
        for axis in (grid.x, grid.y, grid.z):
            content.update(axis.tobytes())
        content.update(json.dumps({'solver_hash': solver_hash,
            'sweeps': config['fsm_sweeps'], 'phase': 'P', 'units': 'km,s'}).encode())
        self.model_key = content.hexdigest()
        # Short disk names keep Windows paths below MAX_PATH. Full identities
        # are verified in metadata, so a truncated-name collision cannot reuse
        # another model or station field silently.
        self.path = Path(directory) / self.model_key[:16]
        self.path.mkdir(parents=True, exist_ok=True)
        identity = self.path/'model_identity.json'
        if identity.exists() and json.loads(identity.read_text(encoding='utf-8'))['model_key'] != self.model_key:
            raise RuntimeError('Truncated model-cache hash collision')
        if not identity.exists():
            identity.write_text(json.dumps({'model_key':self.model_key}),encoding='utf-8')
        self.memory = OrderedDict()
        self.max_fields = max(1, int(config['time_field_memory_limit_mb'] * 1024**2 / self.vp.nbytes))
        self.calculated = self.loaded = 0
        self.seconds = 0.
        self.station_nodes = {str(row.station): tuple(grid.nodes([row.st_x_km, row.st_y_km, 0.]))
                              for row in stations.itertuples(index=False)}
        if 'st_elev_km' in stations.columns:
            e = stations[['station', 'st_elev_km']].drop_duplicates('station')
            self.station_elev_km = {str(s): float(v) for s, v in zip(e.station, e.st_elev_km)}
        else:
            self.station_elev_km = {s: 0. for s in self.station_nodes}

    def elevation_correction(self, station, kji):
        """First-order plane-wave correction (s) for a receiver above z=0.

        The eikonal grid starts at z=0 (sea level) and receivers are anchored
        to the surface node. A station at elevation e>0 is farther from any
        subsurface source and records later:
            t(x, y, -e) ~= t(x, y, 0) + e * qz
        with qz the vertical slowness component of the upgoing wave at the
        station column: the wave continues past sea level to the receiver.
        qz is estimated from the converged field: the first layer below the
        station gives the vertical derivative, and the horizontal slowness
        toward the event is removed in quadrature:
            qz = sqrt(s1^2 - ph^2),  ph = max(T[j+dj, i+di] - T[j, i], 0) / h
        Exact for a laterally homogeneous medium; first order otherwise. The
        correction is pair-dependent, so station statics cannot absorb it.
        """
        e = self.station_elev_km.get(str(station), 0.)
        kji = np.asarray(kji).reshape(-1, 3)
        if e == 0. or len(kji) == 0:
            return np.zeros(len(kji))
        k0, j0, i0 = self.station_nodes[str(station)]
        if k0 + 1 >= len(self.grid.z):
            return np.zeros(len(kji))
        field = self.get(station)
        h = self.grid.spacing
        t1 = float(field[k0 + 1, j0, i0])
        s1 = 0.5 * (1. / self.vp[k0, j0, i0] + 1. / self.vp[k0 + 1, j0, i0])
        xyz = self.grid.xyz(kji)
        di = np.clip(i0 + np.sign(np.round(xyz[:, 0] - self.grid.x[i0])).astype(int), 0, len(self.grid.x) - 1)
        dj = np.clip(j0 + np.sign(np.round(xyz[:, 1] - self.grid.y[j0])).astype(int), 0, len(self.grid.y) - 1)
        ph = np.maximum(field[k0 + 1, dj, di] - t1, 0.0) / h
        qz = np.sqrt(np.maximum(s1 * s1 - ph * ph, 0.0))
        return e * qz

    def get(self, station):
        if station in self.memory:
            self.memory.move_to_end(station)
            return self.memory[station]
        node = self.station_nodes[station]
        key = hashlib.sha256((station + str(node)).encode()).hexdigest()
        path = self.path / (key[:16] + '.npy')
        metadata = self.path / (key[:16] + '.json')
        start = perf_counter()
        if path.exists():
            if not metadata.exists() or json.loads(metadata.read_text(encoding='utf-8'))['station_key'] != key:
                raise RuntimeError('Invalid station-cache identity')
            t = np.load(path, mmap_mode='r', allow_pickle=False)
            if t.shape != self.grid.shape or t.dtype != np.float64 or t[node] != 0:
                raise RuntimeError(f'Invalid time field cache: {path}')
            self.loaded += 1
        else:
            h = self.grid.spacing
            t = fast_sweeping_3d(1 / self.vp, *node, h, h, h, self.config['fsm_sweeps'], 1e6)
            if not np.isfinite(t).all() or t.max() >= 1e6:
                raise RuntimeError('FSM did not reach the whole grid')
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, t, allow_pickle=False)
            temporary.replace(path)
            metadata.write_text(json.dumps({'station_key':key,'station':station,'node':list(map(int,node))}),encoding='utf-8')
            t = np.load(path, mmap_mode='r', allow_pickle=False)
            self.calculated += 1
            if self.calculated % 10 == 0:
                print(f'FSM {self.model_key[:8]}: {self.calculated} station fields calculated',flush=True)
        self.seconds += perf_counter() - start
        self.memory[station] = t
        while len(self.memory) > self.max_fields:
            self.memory.popitem(last=False)
        return t

    def predict(self, picks, event_nodes):
        result = np.empty(len(picks))
        for station, indices in picks.groupby('station', sort=False).indices.items():
            nodes = np.array([event_nodes[eid] for eid in picks.iloc[indices].event_id])
            result[indices] = self.get(station)[tuple(nodes.T)] + self.elevation_correction(station, nodes)
        return result

    def stats(self):
        return {'model_key': self.model_key, 'fields_calculated': self.calculated,
                'fields_loaded': self.loaded, 'field_seconds': self.seconds,
                'resident_field_limit': self.max_fields}

    def close(self):
        # Explicitly release Windows file mappings before temporary directories
        # are cleaned up or before a caller archives a completed run.
        for field in self.memory.values():
            if isinstance(field, np.memmap):
                field._mmap.close()
        self.memory.clear()
