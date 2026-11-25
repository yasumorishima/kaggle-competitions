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

# The TPU image (11-25 probe: Python 3.12, grpc 1.85.1, protobuf 7.37) lacks grpc_tools and polars, and maybe
# pyarrow, pydantic and opentelemetry, which kaggle_evaluation imports. Our dataset gga-wheels carries them;
# install only what fails to import. grpcio-tools must match the image's grpc (an older one writes gencode for an
# older protobuf major, which protobuf 7 rejects), so it goes in on its own with --no-deps.
import importlib, subprocess
_wh = sorted(glob.glob('/kaggle/input/**/grpcio_tools-*.whl', recursive=True))
_wd = os.path.dirname(_wh[0]) if _wh else None
def _pip(args):
    print('installing', args, flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--find-links', _wd] + args, check=True)
    importlib.invalidate_caches()
try:
    import grpc_tools  # noqa: F401
except ImportError:
    try:
        import grpc
        _gv = grpc.__version__
    except ImportError:
        _gv = None
    _have = [os.path.basename(w).split('-')[1] for w in _wh]
    if _gv in _have:
        _pip(['--no-deps', f'grpcio-tools=={_gv}'])
    else:
        try:
            import google.protobuf as _pb
            _major = int(_pb.__version__.split('.')[0])
        except Exception:
            _major = 5
        _v = '1.62.3' if _major < 5 else '1.71.0'
        _pip(['--no-deps', f'grpcio-tools=={_v}'] + ([] if _gv else [f'grpcio=={_v}']))
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        try: _pip(['--no-deps', 'setuptools'])
        except Exception as e: print('setuptools', e, flush=True)
for _mod, _pkg in (('polars', 'polars'), ('pyarrow', 'pyarrow'), ('pydantic', 'pydantic'),
                   ('opentelemetry.sdk', 'opentelemetry-sdk')):
    try:
        importlib.import_module(_mod)
    except ImportError:
        _pip([_pkg])
