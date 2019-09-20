import glob
import os
import shutil
import subprocess
import tempfile

from cffi import FFI

ffi = FFI()

with open(os.path.join("src", "ssht_numba", "src", "ssht.h")) as fl:
    cdef_str = fl.read()
    ffi.cdef(cdef_str)

fftw_path = os.getenv("FFTW_PATH", None)

extra_link_args = []
if fftw_path:
    extra_link_args.append(["-L" + fftw_path])


ssht_src_files = sorted(glob.glob(os.path.join("ssht", "src", "c", "*.c")))
ssht_src_files = [
    fl
    for fl in ssht_src_files
    if not fl.endswith("ssht_about.c") and not fl.endswith("ssht_test.c")
]

ssht_inc_files = sorted(glob.glob(os.path.join("ssht", "src", "c", "*.h")))

makeopts = [
    "-std=c99",
    "-Wall",
    "-O3",
    '-DSSHT_VERSION="1.2b1"',
    '-DSSHT_BUILD="`git rev-parse HEAD`"',
    "-fPIC",
]

if "NO_OPENMP" not in os.environ:
    makeopts += ["-fopenmp"]

inc_string = ""
for inc in ssht_inc_files:
    inc_string += '#include "{}"\n'.format(os.path.basename(inc))

include_dirs = [os.path.join("ssht", "src", "c")]

ffi.set_source(
    "ssht_numba._ssht_cffi",
    inc_string,
    sources=ssht_src_files,
    libraries=["fftw3"],
    include_dirs=include_dirs,
    extra_link_args=extra_link_args,
    extra_compile_args=makeopts,
)


if __name__ == "__main__":
    ffi.compile(verbose=True)
