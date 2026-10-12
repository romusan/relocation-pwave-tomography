"""Sparse regularized updates, gauges and explicit LSMR termination."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
import pandas as pd


def laplacian(mask):
    active = np.flatnonzero(mask.ravel())
    position = np.full(mask.size, -1, dtype=int)
    position[active] = np.arange(len(active))
    rr, cc, vv = [], [], []
    for row, old in enumerate(active):
        k, j, i = np.unravel_index(old, mask.shape)
        neighbors = []
        for dk, dj, di in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
            kk, jj, ii = k+dk, j+dj, i+di
            if 0 <= kk < mask.shape[0] and 0 <= jj < mask.shape[1] and 0 <= ii < mask.shape[2]:
                col = position[(kk * mask.shape[1] + jj) * mask.shape[2] + ii]
                if col >= 0:
                    neighbors.append(col)
        if neighbors:
            rr.extend([row] * (1 + len(neighbors)))
            cc.extend([row, *neighbors])
            vv.extend([float(len(neighbors)), *([-1.] * len(neighbors))])
    return sparse.csr_matrix((vv, (rr, cc)), shape=(len(active), len(active)))


def regularized(A, b, mask, station_matrix, sigma, config, q=False, whitener=None):
    ncell, nsta = A.shape[1], station_matrix.shape[1]
    if ncell == 0:
        raise ValueError('No active inversion cells; change selection or run a coverage diagnostic first')
    sigma = np.asarray(sigma, float)
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Observation uncertainties must be finite and positive')
    W = sparse.diags(1 / sigma) if whitener is None else sparse.csr_matrix(whitener)
    if W.shape != (A.shape[0], A.shape[0]):
        raise ValueError('Whitening operator must match observation rows')
    blocks = [W @ sparse.hstack([A, station_matrix], format='csr')]
    rhs = [W @ np.asarray(b)]
    smooth = config['q_laplace_weight'] if q else config['laplace_weight']
    damp = config['q_cell_damping'] if q else config['cell_damping']
    static_damp = config['q_site_damping'] if q else config['station_damping']
    blocks.append(sparse.hstack([smooth * laplacian(mask), sparse.csr_matrix((ncell, nsta))]))
    rhs.append(np.zeros(ncell))
    blocks.append(sparse.hstack([damp * sparse.eye(ncell), sparse.csr_matrix((ncell, nsta))]))
    rhs.append(np.zeros(ncell))
    if nsta:
        blocks.append(sparse.hstack([sparse.csr_matrix((nsta, ncell)), static_damp * sparse.eye(nsta)]))
        rhs.append(np.zeros(nsta))
    augmented = sparse.vstack(blocks, format='csr')
    result = lsmr(augmented, np.concatenate(rhs), atol=config['lsmr_atol'],
                  btol=config['lsmr_btol'], maxiter=config['lsmr_maxiter'])
    info = {'istop': int(result[1]), 'iterations': int(result[2]),
            'normr': float(result[3]), 'normar': float(result[4]), 'acond': float(result[6]),
            'converged': int(result[1]) in (0, 1, 2, 4, 5),
            'shape': list(augmented.shape), 'nnz': int(augmented.nnz)}
    if not info['converged']:
        raise RuntimeError(f'LSMR failed to converge: {info}')
    return result[0][:ncell], result[0][ncell:], info


def station_design(picks, enabled=True):
    stations = sorted(picks.station.unique())
    # Hard gauge: lexicographically first station fixed to zero.
    free = stations[1:] if enabled else []
    position = {sta: i for i, sta in enumerate(free)}
    rr, cc = [], []
    for row, sta in enumerate(picks.station):
        if sta in position:
            rr.append(row); cc.append(position[sta])
    S = sparse.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(picks), len(free)))
    return S, free, stations[0] if stations else None


def active_mask(matrix, grid, domain, projection, minimum_hits):
    hits = np.asarray((matrix != 0).sum(axis=0)).ravel()
    k, j, i = np.indices(grid.shape)
    lat, lon = projection.latlon(grid.x[i.ravel()], grid.y[j.ravel()])
    inside = ((lat >= domain['lat_min']) & (lat <= domain['lat_max']) &
              (lon >= domain['lon_min']) & (lon <= domain['lon_max']) &
              (grid.z[k.ravel()] >= domain['prof_min']) & (grid.z[k.ravel()] <= domain['prof_max']))
    return ((hits >= minimum_hits) & inside).reshape(grid.shape), hits.reshape(grid.shape)


def linear_resolution(A, mask, coarse, config, seed):
    """Fixed-ray linear diagnostics only; not nonlinear resolution evidence."""
    active = np.flatnonzero(mask)
    k, j, i = np.unravel_index(active, mask.shape)
    # Alternating 2-node blocks; checkerboard expressed in slowness s/km.
    imposed = np.where(((k//2 + j//2 + i//2) % 2) == 0, .003, -.003)
    rng = np.random.default_rng(seed)
    data = A @ imposed + rng.normal(0, .03, A.shape[0])
    recovered, _, info = regularized(A, data, mask, sparse.csr_matrix((A.shape[0], 0)),
                                     np.ones(A.shape[0]), config)
    corr = float(np.corrcoef(imposed, recovered)[0, 1]) if len(active) > 1 and np.std(recovered) > 0 else None
    center = int(np.argmax(np.asarray(A.power(2).sum(axis=0)).ravel()))
    spike = np.zeros(len(active)); spike[center] = .003
    spike_rec,_,spike_info = regularized(A,A@spike,mask,sparse.csr_matrix((A.shape[0],0)),
                                       np.ones(A.shape[0]),config)
    coordinates = np.column_stack([coarse.x[i],coarse.y[j],coarse.z[k]])
    energy = spike_rec**2
    spread = float(np.sqrt(np.sum(energy*np.sum((coordinates-coordinates[center])**2,axis=1))/np.sum(energy))) if energy.sum()>0 else None
    return {'kind': 'linear fixed-ray checkerboard; same forward/inverse operator',
            'block_width_km': 2 * coarse.spacing, 'noise_sigma_s': .03,
            'correlation': corr, 'relative_l2_error': float(np.linalg.norm(recovered-imposed)/np.linalg.norm(imposed)),
            'lsmr': info, 'linear_spike': {'active_index':center,
                'amplitude_ratio_at_imposed_cell':float(spike_rec[center]/spike[center]),
                'energy_spread_km':spread,'lsmr':spike_info}}, imposed, recovered


def lcurve_diagnostic(A,b,mask,station_matrix,config):
    """Linearized sensitivity at final geometry; does not tune on holdout."""
    rows=[]
    L=laplacian(mask)
    for smooth in (config['laplace_weight']/3,config['laplace_weight'],config['laplace_weight']*3):
        for damp in (config['cell_damping']/3,config['cell_damping'],config['cell_damping']*3):
            cfg=dict(config,laplace_weight=smooth,cell_damping=damp)
            row={'laplace_weight':smooth,'cell_damping':damp,
                 'configured_values':smooth==config['laplace_weight'] and damp==config['cell_damping']}
            try:
                dx,ds,info=regularized(A,b,mask,station_matrix,np.full(A.shape[0],config['sigma_p_s']),cfg)
                row.update(data_residual_l2_s=float(np.linalg.norm(b-A@dx-station_matrix@ds)),
                           slowness_update_l2_s_per_km=float(np.linalg.norm(dx)),
                           roughness_l2=float(np.linalg.norm(L@dx)),
                           static_update_l2_s=float(np.linalg.norm(ds)),converged=info['converged'],
                           iterations=info['iterations'],istop=info['istop'])
            except RuntimeError as exc:
                row.update(converged=False,error=str(exc))
            rows.append(row)
    return pd.DataFrame(rows)
