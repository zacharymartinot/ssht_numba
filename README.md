This module builds a `cffi` extension module for some of the functions in [SSHT](https://github.com/astro-informatics/ssht), and a set of wrapper functions
which can be included in other `numba` functions compiled in nopython mode.

## Requirements

- [SSHT](https://github.com/astro-informatics/ssht)
- [FFTW](http://www.fftw.org/download.html)

Python:
- `numpy`
- `numba`
- `cffi`
- `pyyaml`


## Installation

FFTW must be compiled seperately, but only the SSHT source files are required.

In the `sources_config.yaml` file,
- add the path to the SSHT top level directory as `ssht_path`
- add the path to the FFTW top level directory as `fftw_path`

Then run `python build_ssht_cffi.py`, which will produce a `_ssht_cffi.so` file.
