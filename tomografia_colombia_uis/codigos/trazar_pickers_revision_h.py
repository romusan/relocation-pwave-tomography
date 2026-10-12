"""Match selected arrivals against their recorded source files, without new QC."""
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'resultados/campana_perfil_suavidad_20261011h'
p=pd.read_csv(ROOT/'resultados/campana_factorizada_cotas_20261011g/selected_picks.csv',low_memory=False)
keys=['event_id','net','sta','fase','pick_time','prob']
def index(q):
    q=q[keys].copy()
    q['pick_time']=pd.to_datetime(q.pick_time,utc=True,format='mixed').dt.as_unit('ns').astype('int64')
    q['prob']=pd.to_numeric(q.prob).round(6)
    return pd.MultiIndex.from_frame(q)
target=index(p);original=p.copy();original['net']=original.original_net.fillna(original.net);target_original=index(original)
matches=[[] for _ in range(len(p))];files=[]
for path in sorted((ROOT/'datos').rglob('picks*.csv')):
    q=pd.read_csv(path,low_memory=False)
    if not set(keys)<=set(q):continue
    idx=index(q);found=target.isin(idx)|target_original.isin(idx)
    if found.any():
        rel=path.relative_to(ROOT).as_posix()
        files.append(dict(path=rel,selected_matches=int(found.sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for i in found.nonzero()[0]:matches[i].append(rel)
q=p[['raw_row','event_id','station','pick_time','prob']].copy()
q['matching_source_files']=[';'.join(m) for m in matches]
q.to_csv(C/'arrival_source_matches.csv',index=False)
automatic=[i for i,m in enumerate(matches) if any('phasenet' in s or 'picks_nuevos' in s for s in m)]
published=[i for i,m in enumerate(matches) if any('publicados' in s or 'regionales_sgc' in s or 'picks_regionales.csv' in s for s in m) and p.prob.iloc[i]==1]
report=dict(rows=len(p),unmatched_rows=sum(not m for m in matches),automatic_source_rows=len(automatic),published_source_rows=len(published),both_classes_rows=len(set(automatic)&set(published)),files=files,note='Exact event/station/phase/UTC pick timestamp/probability match using selected or explicitly recorded original network; retains IR-to-CM reconciliation. Source names alone do not establish neural model version; generator scripts and weight manifests must also be inspected.')
(C/'arrival_source_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(report,indent=2))
