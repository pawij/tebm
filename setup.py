# Author: Peter Wijeratne (p.wijeratne@pm.me)

import setuptools
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

class build_ext(build_ext):

    def finalize_options(self):
        from Cython.Build import cythonize
        import numpy as np
        import numpy.distutils

        self.distribution.ext_modules[:] = cythonize("**/*.pyx")
        for ext in self.distribution.ext_modules:
            for k, v in np.distutils.misc_util.get_info("npymath").items():
                setattr(ext, k, v)
            ext.include_dirs = [np.get_include()]

        super().finalize_options()

    def build_extensions(self):
        try:
            self.compiler.compiler_so.remove("-Wstrict-prototypes")
        except (AttributeError, ValueError):
            pass
        super().build_extensions()


setup(
    name="tebm",
    description="Temporal Event-Based Models in Python with scikit-learn like API",
    maintainer="Peter Wijeratne",
    url="https://github.com/pawij/tebm",
    license="Academic Use License (TBC)",
    cmdclass={"build_ext": build_ext},
    py_modules=[],
    packages=find_packages("lib"),
    package_dir={"": "lib"},
    ext_modules=[Extension("", [])],
    package_data={},
    python_requires=">=3.5",
    setup_requires=[
        "Cython",
        "numpy>=1.10",
        "setuptools_scm>=3.3",
    ],
    install_requires=[
        "numpy>=1.10",
        "scipy>=0.15",
    ],
)
