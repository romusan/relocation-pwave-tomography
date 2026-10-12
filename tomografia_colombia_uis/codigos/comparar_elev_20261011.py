"""Paired prediction comparison and common-reference velocity sections.

Elevation-campaign variant of comparar_inicios_1d_3d.py: the two run folders
and the prior initial model are parameters, everything else (pairing, event
bootstrap, common 1D anomaly reference, assertions) is unchanged.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from regional.inputs import sha256
from regional.geometry import Projection

ROOT = Path(__file__).resolve().parents[1]


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main_argv(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--old', type=Path, required=True)
    ap.add_argument('--new', type=Path, required=True)
    ap.add_argument('--prior', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(argv)
    old, new, prior_path, out = a.old, a.new, a.prior, a.output
    out.mkdir(parents=True, exist_ok=False)
    s1 = json.loads((old / 'summary.json').read_text(encoding='utf-8'))
    s3 = json.loads((new / 'summary.json').read_text(encoding='utf-8'))
    for key in ['domain', 'config', 'train_events', 'holdout_events', 'train_picks',
                'holdout_picks', 'direct_shape', 'inversion_shape']:
        assert s1[key] == s3[key], key
    split1 = pd.read_csv(old / 'events_split.csv').set_index('event_id').sort_index()
    split3 = pd.read_csv(new / 'events_split.csv').set_index('event_id').sort_index()
    pd.testing.assert_frame_equal(split1, split3)
    p1 = pd.read_csv(old / 'predictions.csv', low_memory=False)
    p3 = pd.read_csv(new / 'predictions.csv', low_memory=False)
    shared = p1.merge(p3, on=['event_id', 'station', 'fase', 'raw_row'],
                      suffixes=('_1d', '_3d'), validate='one_to_one')
    assert len(shared) == len(p1) == len(p3)
    for key in ['observed_s', 'ev_lat', 'ev_lon', 'ev_prof_km', 'st_x_km', 'st_y_km',
                'source_x_km', 'source_y_km']:
        np.testing.assert_allclose(shared[key + '_1d'], shared[key + '_3d'], rtol=0, atol=1e-10)
    assert (shared.split_1d == shared.split_3d).all()
    held = shared[shared.split_1d.eq('holdout')].copy()
    np.testing.assert_array_equal(held.origin_shift_s_1d, 0)
    np.testing.assert_array_equal(held.origin_shift_s_3d, 0)
    held['sse_1d'] = held.residual_s_1d ** 2
    held['sse_3d'] = held.residual_s_3d ** 2
    cluster = held.groupby('event_id').agg(sse_1d=('sse_1d', 'sum'),
                                           sse_3d=('sse_3d', 'sum'), n=('sse_1d', 'size'))
    rng = np.random.default_rng(20261010)
    samples = rng.integers(0, len(cluster), size=(3000, len(cluster)))
    c = cluster.to_numpy()
    aggregate = c[samples].sum(axis=1)
    delta = np.sqrt(aggregate[:, 1] / aggregate[:, 2]) - np.sqrt(aggregate[:, 0] / aggregate[:, 2])
    ci = np.quantile(delta, [.025, .975]).tolist()
    cluster['rms_1d'] = np.sqrt(cluster.sse_1d / cluster.n)
    cluster['rms_3d'] = np.sqrt(cluster.sse_3d / cluster.n)
    cluster.to_csv(out / 'comparacion_por_sismo_reservado.csv')
    held.to_csv(out / 'observaciones_reservadas_comunes.csv', index=False)
    comparisons = []
    for name, s in [('1D original', s1), ('1D más placas continuas', s3)]:
        comparisons.append(dict(initialization=name, round=0,
            train_rms_s=s['baseline']['train_rms_fixed_catalog_s'],
            holdout_rms_s=s['baseline']['holdout_rms_fixed_catalog_origin_s']))
        comparisons.extend(dict(initialization=name, round=h['round'],
            train_rms_s=h['train_rms_after_update_s'],
            holdout_rms_s=h['holdout_rms_fixed_catalog_origin_s']) for h in s['history'])
    pd.DataFrame(comparisons).to_csv(out / 'rms_por_ronda.csv', index=False)
    sources = {}
    for run in [old, new]:
        for f, h in json.loads((run / 'manifest.json').read_text(encoding='utf-8')).items():
            assert sha256(f) == h, f
        for name in ['manifest.json', 'summary.json', 'vp_final.npz', 'predictions.csv', 'events_split.csv']:
            f = run / name
            sources[str(f.resolve())] = sha256(f)
    oldmodel = np.load(old / 'vp_final.npz')
    newmodel = np.load(new / 'vp_final.npz')
    prior = np.load(prior_path)
    np.testing.assert_array_equal(prior['vp_background_1d'], oldmodel['vp_initial'])
    np.testing.assert_array_equal(newmodel['vp_initial'], prior['vp'])
    for name in ['xg', 'yg', 'zg']:
        np.testing.assert_array_equal(oldmodel[name], newmodel[name])
    result = dict(same_observations=True, same_original_1d_background=True,
        same_parameters_and_split=True, reserved_events=len(cluster), reserved_P=len(held),
        train_P=len(shared) - len(held),
        final_train_1d_rms_s=s1['history'][-1]['train_rms_after_update_s'],
        final_train_3d_rms_s=s3['history'][-1]['train_rms_after_update_s'],
        final_holdout_1d_rms_s=rms(held.residual_s_1d),
        final_holdout_3d_rms_s=rms(held.residual_s_3d),
        holdout_delta_3d_minus_1d_s=rms(held.residual_s_3d) - rms(held.residual_s_1d),
        event_cluster_bootstrap_delta_95percent_ci_s=ci,
        held_events_with_lower_rms_3d=int((cluster.rms_3d < cluster.rms_1d).sum()),
        bootstrap_replicates=3000,
        initial_holdout_1d_rms_s=s1['baseline']['holdout_rms_fixed_catalog_origin_s'],
        initial_holdout_3d_rms_s=s3['baseline']['holdout_rms_fixed_catalog_origin_s'],
        active_final_cells_1d=int(oldmodel['active'].sum()),
        active_final_cells_3d=int(newmodel['active'].sum()),
        limitations=[
            '3D geometry and local contrasts use earlier training results; this is a conditioned initialization comparison, not independent geological verification',
            'Reserved set unchanged and no parameter selection performed using these results',
            'Slab2 may share catalog inputs; statistical CI measures event sampling only, not systematic model uncertainty',
            'Identical first-order FSM and approximate ray limitations remain',
            'Station elevations applied as a first-order sub-grid receiver correction on both arms'])
    result['holdout_improvement_percent'] = 100 * (1 - result['final_holdout_3d_rms_s'] / result['final_holdout_1d_rms_s'])
    result['train_improvement_percent'] = 100 * (1 - result['final_train_3d_rms_s'] / result['final_train_1d_rms_s'])
    pr = Projection(s1['domain'])
    start = np.array(pr.xy(3.9, -78.2))
    end = np.array(pr.xy(6.81, -73.1))
    length = float(np.linalg.norm(end - start))
    distance = np.linspace(0, length, 150)
    depth = np.arange(0, 181, 5)
    xy = start[None, :] + distance[:, None] / length * (end - start)[None, :]
    zz, dd = np.meshgrid(depth, distance, indexing='ij')
    points = np.c_[zz.ravel(), np.broadcast_to(xy[:, 1], zz.shape).ravel(),
                   np.broadcast_to(xy[:, 0], zz.shape).ravel()]
    axes = (oldmodel['zg'], oldmodel['yg'], oldmodel['xg'])
    v0 = RegularGridInterpolator(axes, oldmodel['vp_initial'])(points).reshape(zz.shape)
    models = [('Inicio 1D original', oldmodel['vp_initial']),
              ('Inicio 1D más placas', prior['vp']),
              ('Vp final desde 1D', oldmodel['vp']),
              ('Vp final desde 3D', newmodel['vp'])]
    fields = []
    for title, volume in models:
        v = RegularGridInterpolator(axes, volume)(points).reshape(zz.shape)
        anomaly = 100 * (v / v0 - 1)
        fields.append((title, v, anomaly))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    for ax, (title, v, anomaly) in zip(axs.ravel(), fields):
        im = ax.pcolormesh(distance, depth, anomaly, cmap='RdBu_r', vmin=-15, vmax=15, shading='auto')
        ax.set_title(title)
        ax.set_ylim(180, 0)
        ax.set_xlabel('Distancia Pacífico–Bucaramanga km')
        ax.set_ylabel('Profundidad km')
    fig.colorbar(im, ax=axs.ravel().tolist(), label='dVp respecto al mismo perfil 1D original %', shrink=.85)
    fig.suptitle('Comparación con referencia común y sin exageración de las anomalías')
    fig.savefig(out / 'seccion_comparacion_1d_3d.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, color in [('1D original', '#555555'), ('1D más placas continuas', '#1581a0')]:
        d = pd.DataFrame(comparisons)
        d = d[d.initialization.eq(name)]
        ax.plot(d['round'], d.holdout_rms_s, 'o-', color=color, label=name + ' reservados')
        ax.plot(d['round'], d.train_rms_s, 'o--', color=color, alpha=.6, label=name + ' entrenamiento')
    ax.set(xlabel='Ronda 0 antes de invertir', ylabel='RMS s', xticks=[0, 1, 2, 3])
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out / 'rms_comparacion.png', dpi=160)
    plt.close(fig)
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[x[0] for x in fields],
                        shared_xaxes=True, shared_yaxes=True)
    for i, (title, v, anomaly) in enumerate(fields):
        fig.add_trace(go.Heatmap(x=distance, y=depth, z=anomaly, coloraxis='coloraxis',
            customdata=v, hovertemplate='Distancia %{x:.1f} km<br>Prof. %{y:.1f} km<br>dVp %{z:.2f}%<br>Vp %{customdata:.3f} km/s<extra>' + title + '</extra>'),
            row=i // 2 + 1, col=i % 2 + 1)
    fig.update_yaxes(autorange='reversed', title='Profundidad km')
    fig.update_xaxes(title='Distancia km')
    fig.update_layout(height=800, title='Sección Pacífico–Bucaramanga con referencia 1D común',
        coloraxis=dict(colorscale='RdBu_r', cmin=-15, cmax=15,
                       colorbar=dict(title='dVp %')))
    note = '<p>Mismos sismos, parámetros y validación. Todas las anomalías se calculan respecto al perfil 1D ORIGINAL. Las secciones muestran el modelo completo; fuera de cobertura, el Vp puede conservar el prior y no debe interpretarse como estructura resuelta.</p>'
    html = fig.to_html(include_plotlyjs=True).replace('<body>', '<body>' + note)
    (out / 'comparacion_1d_3d.html').write_text(html, encoding='utf-8')
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    sources[str(prior_path.resolve())] = sha256(prior_path)
    (out / 'comparacion.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'input_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    main_argv(sys.argv[1:])
