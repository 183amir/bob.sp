#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Thu 30 Jan 08:45:49 2014 CET

from setuptools import setup, find_packages, dist
dist.Distribution(dict(setup_requires=['xbob.blitz']))
from xbob.blitz.extension import Extension

packages = ['bob-sp >= 1.2.2']
version = '2.0.0a0'

setup(

    name='xbob.sp',
    version=version,
    description='Bindings for Bob\'s signal processing utilities',
    url='http://github.com/anjos/xbob.sp',
    license='BSD',
    author='Andre Anjos',
    author_email='andre.anjos@idiap.ch',

    long_description=open('README.rst').read(),

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
      'setuptools',
      'xbob.blitz',
    ],

    namespace_packages=[
      "xbob",
      ],

    ext_modules = [
      Extension("xbob.sp._library",
        [
          "xbob/sp/quantization.cpp",
          "xbob/sp/extrapolate.cpp",
          "xbob/sp/fft1d.cpp",
          "xbob/sp/fft2d.cpp",
          "xbob/sp/ifft1d.cpp",
          "xbob/sp/ifft2d.cpp",
          "xbob/sp/fft.cpp",
          "xbob/sp/dct1d.cpp",
          "xbob/sp/dct2d.cpp",
          "xbob/sp/idct1d.cpp",
          "xbob/sp/idct2d.cpp",
          "xbob/sp/dct.cpp",
          "xbob/sp/main.cpp",
          ],
        packages = packages,
        version = version,
        ),
      ],

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
