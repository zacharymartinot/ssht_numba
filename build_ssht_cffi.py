import glob
import os
import shutil
import subprocess
import tempfile

from cffi import FFI

ffi = FFI()
this_dir = ""  # os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(this_dir, "src", "ssht_numba", "src", "ssht.h")) as fl:
    cdef_str = fl.read()
    ffi.cdef(cdef_str)

fftw_path = os.getenv("FFTW_PATH", None)

extra_link_args = []
if fftw_path:
    extra_link_args.append(["-L" + fftw_path])


ssht_src_files = sorted(glob.glob(os.path.join(this_dir, "ssht", "src", "c", "*.c")))
ssht_src_files = [fl for fl in ssht_src_files if not fl.endswith("ssht_about.c")]

ssht_inc_files = sorted(glob.glob(os.path.join(this_dir, "ssht", "src", "c", "*.h")))
ssht_inc_files = [fl for fl in ssht_inc_files if not fl.endswith("ssht_about.h")]

# with open(os.path.join(this_dir, 'ssht', 'makefile')) as fl:
#     for line in fl.readlines():
#         if line.lstrip().startswith("OPT"):
#             makeopts = line.split("=", 1)[-1].split(" -")
# makeopts = ["-"+mk for mk in makeopts if (mk and mk != 'pedantic')]
# makeopts += ['-fPIC']

makeopts = [
    "-std=c99",
    "-Wall",
    "-O3",
    "-fopenmp",
    '-DSSHT_VERSION="1.2b1"',
    '-DSSHT_BUILD="`git rev-parse HEAD`"',
    "-fPIC",
]

inc_string = ""
for inc in ssht_inc_files:
    inc_string += '#include "{}"\n'.format(os.path.basename(inc))

include_dirs = [os.path.join(this_dir, "ssht", "src", "c")]

print(makeopts)
print(inc_string)
print(ssht_src_files)

ffi.set_source(
    "ssht_numba._ssht_cffi",
    inc_string,
    sources=ssht_src_files,
    libraries=["fftw3"],
    include_dirs=include_dirs,
    extra_link_args=extra_link_args,
    extra_compile_args=makeopts,
)


# def build():
#     cwd = os.getcwd()
#
#     ffi.compile(os.path.join(cwd, "ssht_numba"), verbose=True)
#
#     files_to_delete = ["_ssht_cffi.o", "_ssht_cffi.c"]
#     build_directory = ssht_path.split("/")[1]
#
#     for file in files_to_delete:
#         # print os.path.join(cwd, file)
#         os.remove(os.path.join(cwd, "ssht_numba", file))
#
#     # print os.path.join(cwd, build_directory)
#     shutil.rmtree(os.path.join(cwd, "ssht_numba", build_directory))


if __name__ == "__main__":
    ffi.compile(verbose=True)
