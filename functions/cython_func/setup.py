#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize("EMbase_c.pyx"),
    include_dirs=[numpy.get_include()],
)    