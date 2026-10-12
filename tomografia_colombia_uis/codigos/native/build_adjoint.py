from pathlib import Path
import subprocess,os,json,hashlib,shutil
HERE=Path(__file__).resolve().parent
compiler=shutil.which('g++') or 'C:/msys64/ucrt64/bin/g++.exe'
out=HERE/('eikonal_adjoint.dll' if os.name=='nt' else 'eikonal_adjoint.so')
flags=['-O3','-std=c++17','-shared']+(['-static','-static-libgcc','-static-libstdc++'] if os.name=='nt' else ['-fPIC'])
cmd=[compiler,*flags,str(HERE/'eikonal_adjoint.cpp'),'-o',str(out)]
env=os.environ.copy();env['PATH']=str(Path(compiler).parent)+os.pathsep+env.get('PATH','')
subprocess.run(cmd,check=True,env=env)
(HERE/'build_adjoint_info.json').write_text(json.dumps({'command':cmd,'source_sha256':hashlib.sha256((HERE/'eikonal_adjoint.cpp').read_bytes()).hexdigest(),'binary_sha256':hashlib.sha256(out.read_bytes()).hexdigest()},indent=2),encoding='utf-8')
print(out)
