from pathlib import Path
HERE=Path(__file__).resolve().parent
source=HERE/'regional/pipeline.py';dest=HERE/'regional/pipeline_consistent.py'
assert not dest.exists(),'Do not overwrite a recorded implementation'
text=source.read_text(encoding='utf-8')
text=text.replace('from .rays import kernels, mapping','from .rays_consistent import kernels, mapping, active_mask')
text=text.replace('from .inversion import active_mask, station_design','from .inversion import station_design')
text=text.replace("diagnostic, imposed, recovered = linear_resolution(last_A, last_mask, coarse, config, config['seed'])","diagnostic, imposed, recovered = linear_resolution(last_A, last_mask, coarse, config, config['seed'])\n    diagnostic['kind'] = 'linear fixed discrete-Jacobian checkerboard; same forward/inverse linear operator'")
text=text.replace("'status': 'experimental_first_order_FSM'","'status': 'consistent_discrete_first_order_FSM'")
text=text.replace("'26-neighbour backtrace is an approximate kernel; inspect ray-integral mismatch.'","'Exact discrete Jacobian verified against finite differences; grid accuracy remains separate.'")
dest.write_text(text,encoding='utf-8');print(dest)
