#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['bob.blitz', 'bob.core']))
from bob.blitz.extension import Extension, Library, build_ext

version = '2.0.0a0'

setup(

    name='bob.sp',
    version=version,
    description='Bindings for Bob\'s signal processing utilities',
    url='http://github.com/bioidiap/bob.sp',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,

    install_requires=[
      'setuptools',
      'bob.blitz',
      'bob.core'
    ],

    namespace_packages=[
      "bob",
      ],

    ext_modules = [
      Extension("bob.sp.version",
        [
          "bob/sp/version.cpp",
          ],
        version = version,
        bob_packages = ['bob.core'],
        ),

      Library('bob.sp.bob_sp',
        [
          "bob/sp/cpp/DCT1DNaive.cpp",
          "bob/sp/cpp/DCT2D.cpp",
          "bob/sp/cpp/DCT2DNaive.cpp",
          "bob/sp/cpp/FFT1D.cpp",
          "bob/sp/cpp/FFT2DNaive.cpp",
          "bob/sp/cpp/DCT1D.cpp",
          "bob/sp/cpp/FFT1DNaive.cpp",
          "bob/sp/cpp/FFT2D.cpp",
          "bob/sp/cpp/fftpack.c"
        ],
        version = version,
        bob_packages = ['bob.core'],
      ),

      Extension("bob.sp._library",
        [
          "bob/sp/quantization.cpp",
          "bob/sp/extrapolate.cpp",
          "bob/sp/fft1d.cpp",
          "bob/sp/fft2d.cpp",
          "bob/sp/ifft1d.cpp",
          "bob/sp/ifft2d.cpp",
          "bob/sp/fft.cpp",
          "bob/sp/dct1d.cpp",
          "bob/sp/dct2d.cpp",
          "bob/sp/idct1d.cpp",
          "bob/sp/idct2d.cpp",
          "bob/sp/dct.cpp",
          "bob/sp/main.cpp",
          ],
        version = version,
        bob_packages = ['bob.core'],
        ),
      ],

    cmdclass = {
      'build_ext': build_ext
    },

    classifiers = [
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Software Development :: Libraries :: Python Modules',
      ],

    )
