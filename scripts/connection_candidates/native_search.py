"""Optional compiled search, cached by source hash; pure Python fallback remains."""
import ctypes,hashlib,subprocess,tempfile
from pathlib import Path
import numpy as np
_LIB=None

def search(cost,source,target,radius,origin):
    global _LIB
    if _LIB is None:
        source_path=Path(__file__).with_name('pixel_astar.cpp');key=hashlib.sha256(source_path.read_bytes()).hexdigest()[:16];lib_path=Path(tempfile.gettempdir())/f'xjtlu_corridor_{key}.so'
        if not lib_path.exists():subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(source_path),'-o',str(lib_path)],check=True,capture_output=True)
        _LIB=ctypes.CDLL(str(lib_path));_LIB.corridor_astar.argtypes=[ctypes.POINTER(ctypes.c_float),ctypes.c_int,ctypes.c_int,*([ctypes.c_double]*5),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int)];_LIB.corridor_astar.restype=ctypes.c_int
    a=np.ascontiguousarray(cost,dtype=np.float32);h,w=a.shape;out=np.empty(h*w,np.int32);expanded=ctypes.c_int();s=np.asarray(source)-origin;t=np.asarray(target)-origin
    n=_LIB.corridor_astar(a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),w,h,*s,*t,radius,out.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),ctypes.byref(expanded));k=out[:n][::-1];path=np.stack([k%w,k//w],axis=1)+origin
    return path.tolist() if n else None,expanded.value
