"""Build the optional C++17 kernel. Does not install packages or alter PATH."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compiler', help='Path to a GCC-compatible C++ compiler')
    a = ap.parse_args()
    compiler = a.compiler or shutil.which('g++') or shutil.which('clang++')
    if not compiler and os.name == 'nt':
        candidate = Path('C:/msys64/ucrt64/bin/g++.exe')
        if candidate.exists():
            compiler = str(candidate)
    if not compiler:
        sys.exit('No C++ compiler found. Supply --compiler; the numpy backend is available.')
    compiler = str(Path(compiler).resolve())
    out = HERE / ('relocation.dll' if os.name == 'nt' else 'relocation.so')
    flags = ['-O3', '-std=c++17', '-shared']
    if os.name == 'nt':
        flags += ['-static', '-static-libgcc', '-static-libstdc++']
    else:
        flags += ['-fPIC']
    cmd = [compiler, *flags, str(HERE / 'relocation.cpp'), '-o', str(out)]
    env = os.environ.copy()
    env['PATH'] = str(Path(compiler).parent) + os.pathsep + env.get('PATH', '')
    subprocess.run(cmd, check=True, env=env)
    version = subprocess.check_output([compiler, '--version'], env=env, text=True)
    (HERE / 'build_info.json').write_text(json.dumps({
        'compiler': version.splitlines()[0], 'command': cmd,
        'platform': platform.platform(), 'python': sys.version,
        'source_sha256': hashlib.sha256((HERE/'relocation.cpp').read_bytes()).hexdigest(),
    }, indent=2), encoding='utf-8')
    print(out)


if __name__ == '__main__':
    main()
