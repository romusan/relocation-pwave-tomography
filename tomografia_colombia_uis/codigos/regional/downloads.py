"""Window-aware waveform cache; transient failures are never empty markers."""
from pathlib import Path
import hashlib
import json
import time
import requests


def download(session, url, params, directory, retries=3, pause=0.4):
    identity = json.dumps({'url':url,'params':params},sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    data = directory/(key+'.mseed')
    state = directory/(key+'.json')
    if state.exists():
        status = json.loads(state.read_text(encoding='utf-8'))
        if status.get('identity') != identity:
            raise RuntimeError('Waveform cache identity mismatch')
        if status.get('status') == 'no_data':
            return None,status
        if status.get('status') == 'downloaded' and data.exists() and data.stat().st_size == status.get('bytes'):
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            if digest == status.get('sha256'):
                return data,status
    attempts=[]
    for attempt in range(retries):
        try:
            response=session.get(url,params=params,timeout=(10,120))
            code=response.status_code
            attempts.append({'http':code,'bytes':len(response.content)})
            if code==204:
                result={'status':'no_data','http':code}
                break
            if code==200 and len(response.content)>2000:
                temp=data.with_suffix('.tmp')
                temp.write_bytes(response.content)
                temp.replace(data)
                result={'status':'downloaded','http':code,'bytes':len(response.content),
                        'sha256':hashlib.sha256(response.content).hexdigest()}
                break
            # 4xx except 408/429 needs request/service correction, not a false
            # "no data" marker. It remains retryable in a later campaign.
            result={'status':'request_error' if 400<=code<500 and code not in (408,429) else 'temporary_error','http':code}
            if result['status']=='request_error':
                break
        except requests.RequestException as exc:
            attempts.append({'exception':type(exc).__name__,'message':str(exc)})
            result={'status':'temporary_error','http':None}
        if attempt+1<retries:
            time.sleep(pause*(2**attempt))
    result.update(identity=identity,attempts=attempts,checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    temp=state.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2),encoding='utf-8')
    temp.replace(state)
    return (data if result['status']=='downloaded' else None),result
