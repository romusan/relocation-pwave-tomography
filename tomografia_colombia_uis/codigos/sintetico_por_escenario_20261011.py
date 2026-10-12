"""Run one frozen synthetic case; parallelism does not change its data or fit."""
from robustez_20261011 import *

if __name__=='__main__':
    amp={'03':.03,'08':.08}[sys.argv[1]]
    original=Path(__file__).with_name('sinteticos_no_lineales_20261011.py')
    source=original.read_text(encoding='utf-8')
    assert source.count('for amp in [.03,.08]:')==1
    source=source.replace('for amp in [.03,.08]:',f'for amp in [{amp!r}]:')
    lines=source.splitlines();modified=[]
    for line in lines:
        if line.startswith("    save(OUT/'nonlinear_synthetics.json'"):
            modified.append("    if all((OUT/f'synthetic_{int(a*100):02d}pct/recovery_metrics.json').exists() for a in [.03,.08]):")
            line='    '+line
        modified.append(line)
    save(OUT/f'scenario_{int(100*amp):02d}_worker.json',dict(amplitude=amp,sequential_script_sha256=digest(original),wrapper_sha256=digest(__file__),changes='amplitude selection and final aggregate written only when both cases finished; same seeded generator and inverse algorithm'))
    ns={'__name__':'__main__','__file__':str(original)}
    exec(compile('\n'.join(modified),str(original)+' [one scenario]','exec'),ns)
