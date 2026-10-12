"""C++ or reference downhill tracing and coarse-grid kernels."""
import ctypes
import numpy as np
from scipy import sparse
from relocation import _native

_configured = False


def trace(field, start, finish, spacing, backend='cpp'):
    global _configured
    shape = field.shape
    capacity = sum(shape) * 4
    if backend == 'cpp':
        lib = _native()
        if not _configured:
            arr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
            idx = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
            lib.trace_ray.argtypes = [arr, *([ctypes.c_int64] * 5), ctypes.c_double,
                                      ctypes.c_int64, idx, arr]
            lib.trace_ray.restype = ctypes.c_int64
            _configured = True
        indices = np.empty(capacity, dtype=np.int64)
        lengths = np.empty(capacity)
        n = lib.trace_ray(field, *shape, int(start), int(finish), spacing, capacity, indices, lengths)
        if n < 0:
            return None
        return indices[:n], lengths[:n]
    if backend != 'reference':
        raise ValueError('Ray backend must be cpp or reference')
    indices, lengths = [], []
    current = int(start)
    for _ in range(capacity):
        if current == finish:
            return np.asarray(indices, dtype=np.int64), np.asarray(lengths)
        k, j, i = np.unravel_index(current, shape)
        minimum, following, distance = field[k, j, i], current, 0
        for dk in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for di in (-1, 0, 1):
                    kk, jj, ii = k + dk, j + dj, i + di
                    if (dk == dj == di == 0 or kk < 0 or jj < 0 or ii < 0 or
                            kk >= shape[0] or jj >= shape[1] or ii >= shape[2]):
                        continue
                    if field[kk, jj, ii] < minimum:
                        minimum = field[kk, jj, ii]
                        following = (kk * shape[1] + jj) * shape[2] + ii
                        distance = spacing * np.sqrt(dk*dk + dj*dj + di*di)
        if following == current:
            return None
        indices.append(current)
        lengths.append(distance)
        current = following
    return None


def mapping(direct, coarse):
    k, j, i = np.indices(direct.shape)
    ratio = coarse.spacing / direct.spacing
    ck, cj, ci = np.rint(k / ratio).astype(int), np.rint(j / ratio).astype(int), np.rint(i / ratio).astype(int)
    return ((ck * len(coarse.y) + cj) * len(coarse.x) + ci).ravel()


def kernels(cache, picks, nodes, coarse, backend='cpp', q_mode=False):
    direct = cache.grid
    map_flat = mapping(direct, coarse)
    rr, cc, vv = [], [], []
    failed = []
    mismatch = []
    for station, ids in picks.groupby('station', sort=False).indices.items():
        field = cache.get(station)
        receiver = int(direct.linear(np.array([cache.station_nodes[station]]))[0])
        for row in ids:
            start = int(direct.linear(np.array([nodes[picks.iloc[row].event_id]]))[0])
            path = trace(field, start, receiver, direct.spacing, backend)
            if path is None or len(path[0]) == 0:
                failed.append(int(row))
                continue
            indices, length = path
            qweights = length / cache.vp.ravel()[indices]
            mismatch.append(float(qweights.sum() - field.ravel()[start]))
            columns = map_flat[indices]
            weight = qweights if q_mode else length
            unique, inverse = np.unique(columns, return_inverse=True)
            sums = np.bincount(inverse, weights=weight)
            rr.extend([int(row)] * len(unique)); cc.extend(unique); vv.extend(sums)
    matrix = sparse.csr_matrix((vv, (rr, cc)), shape=(len(picks), int(np.prod(coarse.shape))))
    return matrix, {'failed_rows': failed,
        'ray_integral_minus_field_median_s': float(np.median(mismatch)) if mismatch else None,
        'ray_integral_minus_field_p95_abs_s': float(np.percentile(np.abs(mismatch), 95)) if mismatch else None,
        'limitation': '26-neighbour ray lengths approximate FSM sensitivity; integral differs from field time'}
