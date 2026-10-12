from pathlib import Path
import subprocess,os,hashlib,json,shutil
here=Path(__file__).resolve().parent
compiler=shutil.which('g++') or 'C:/msys64/ucrt64/bin/g++.exe'
out=here/('factored_eikonal.dll' if os.name=='nt' else 'factored_eikonal.so')
cmd=[compiler,'-O3','-std=c++17','-shared',*(['-static','-static-libgcc','-static-libstdc++'] if os.name=='nt' else ['-fPIC']),str(here/'factored_eikonal.cpp'),'-o',str(out)]
env=os.environ.copy();env['PATH']=str(Path(compiler).parent)+os.pathsep+env.get('PATH','')
subprocess.run(cmd,env=env,check=True)
(here/'build_factored_info.json').write_text(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256((here/'factored_eikonal.cpp').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
print(out)
