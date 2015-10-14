from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os.path

# Adjust to point to your SRILM build directory
SRILM_DIR = "/Users/erik/srilm-1.7.1"
# Adjust to match your architecture -- if unsure, build SRILM and then see
# what subdirectories you have in SRILM_DIR/lib/.
SRILM_ARCH = "macosx"

SRILM_INCLUDE_DIR = os.path.join(SRILM_DIR, "include")
SRILM_LIB_DIR = os.path.join(SRILM_DIR, "lib", SRILM_ARCH)

setup(
            cmdclass = {'build_ext': build_ext},
                ext_modules = [
                          Extension("srilm",
                                              ["srilm.pyx"],
                                                              language="c++",
                                                                              include_dirs=[SRILM_INCLUDE_DIR],
                                                                                              libraries=["oolm", "dstruct", "misc", "z"],
                                                                                                              library_dirs=[SRILM_LIB_DIR],
                                                                                                                              extra_compile_args=['-fPIC'],
                                                                                                                                              )
                                ],
                )
# setup(
    # cmdclass = {'build_ext': build_ext},
    # ext_modules = [
      # Extension("srilm",
                # ["srilm.pyx"],
                # language="c++",
                # include_dirs=[SRILM_INCLUDE_DIR],
                # extra_link_args=["-L" + SRILM_LIB_DIR, '-liboolm.a', '-libdstruct', '-libmisc'],
                # )
      # ],
# )

