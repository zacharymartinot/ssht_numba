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

FFTW must be compiled separately, but only the SSHT source files are required.

In the `sources_config.yaml` file,
- add the path to the SSHT top level directory as `ssht_path`
- add the path to the directory containing the FFTW library as `fftw_path`

The package can then be installed with `pip install .`. During setup the
extension module `ssht_numba/_ssht_cffi.so` will be built from SSHT source.
