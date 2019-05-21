from setuptools import setup
from build_ssht_cffi import build

build()

setup(
    name='ssht_numba',
    version='0.0.1',
    description='Numba njit-compilable wrappers for SSHT.',
    url='https://github.com/zacharymartinot/ssht_numba',
    author='Zachary Martinot',
    author_email='zmarti@sas.upenn.edu',
    packages=['ssht_numba'],
    zip_safe=False)
