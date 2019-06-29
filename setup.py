from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Compiler import Options
import numpy

"""
    ~$: python setup.py bdist_egg

    manually build Cython extension
    ~$: python setup.py build_ext -if
"""

# set False while developing
Options.annotate = True

extensions = [
    Extension(
        # path of .so
        "supervised.regression_tree",
        ["supervised/regression_tree.pyx"],
        # include_dirs=['/some/path/to/include/'],
        # libraries=['ext_C_libs'],
        # library_dirs=['/some/path/to/include/'],
    ),
    Extension(
        # path of .so
        "supervised.gbdt_tree",
        ["supervised/gbdt_tree.pyx"],
    ),
]

setup(
    name='ml-toys',
    version='0.9',
    description='a tiny lib of machine learning',
    author='zyf',
    author_email='dtcf@163.com',
    url='https://github.com/two2er/ml-toys',
    packages=find_packages(),

    setup_requires=['numpy', 'cython'],

    ext_modules=cythonize(extensions,
                          # path of generated C files
                          build_dir='build'),
    include_dirs=[numpy.get_include()],
)
