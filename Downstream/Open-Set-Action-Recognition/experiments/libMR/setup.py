from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import sys
import numpy
#ext_modules = [Extension("libmr", ["libmr.pyx", "MetaRecognition.cpp"])]

setup(
      ext_modules = cythonize(Extension('libmr',
                                        ["libmr.pyx",
                                         "MetaRecognition.cpp",
                                         "weibull.c"
                                         ],
                                        include_dirs = [".", numpy.get_include()],
                                        language="c++",
                  )),
      data_files = [('.', ['MetaRecognition.h', 'weibull.h'])],

)
