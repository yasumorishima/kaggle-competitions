# Generalist Gaming Agent submission (yasunorim). The agent below is ours (gga/agent.py in our repo);
# this header finds the competition's harness and the footer hands the agent to the inference server.
import glob, os, sys
from pathlib import Path
_srv = sorted(glob.glob('/kaggle/input/**/kaggle_evaluation/gga/gga_inference_server.py', recursive=True))[0]
_root = str(Path(_srv).parents[2])
os.environ.setdefault('GGA_ROOT', _root)
for p in (_root,):
    if p not in sys.path:
        sys.path.insert(0, p)
print('harness', _root, flush=True)

# The TPU image has no grpc_tools, which kaggle_evaluation needs to compile its gRPC stubs at import.
# Our dataset gga-wheels carries grpcio-tools/grpcio for Python 3.10-3.13 in a protobuf-4 line (1.62.3) and a
# protobuf-5+ line (1.71.0); install the one that matches this image's protobuf.
try:
    import grpc_tools  # noqa: F401
except ImportError:
    import subprocess
    _wh = sorted(glob.glob('/kaggle/input/**/grpcio_tools-*.whl', recursive=True))
    _wd = os.path.dirname(_wh[0]) if _wh else None
    try:
        import google.protobuf as _pb
        _major = int(_pb.__version__.split('.')[0])
    except Exception:
        _major = 5
    _v = '1.62.3' if _major < 5 else '1.71.0'
    _pkgs = [f'grpcio-tools=={_v}']
    try:
        import grpc  # noqa: F401
    except ImportError:
        _pkgs.append(f'grpcio=={_v}')
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        _pkgs.append('setuptools')
    print('installing', _pkgs, 'from', _wd, 'protobuf major', _major, flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', '--find-links', _wd] + _pkgs, check=True)
    import importlib; importlib.invalidate_caches()
