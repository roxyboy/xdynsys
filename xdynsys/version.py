from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "xdynsys: Python version of functions to utilize the ROMS outputs."
# Long description will go up on the pypi page
long_description = """
xdynsys
========
Python version of functions to utilize the Regional Ocean Model outputs.
License
=======
``xdynsys`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.
All trademarks referenced herein are property of their respective holders.
Copyright (c) 2017--, Henk Dijkstra, Utrecht University
"""

NAME = "xdynsys"
MAINTAINER = "Takaya Uchida"
MAINTAINER_EMAIL = "takaya@ldeo.columbia.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/roxyboy/xdynsys"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Takaya Uchida"
AUTHOR_EMAIL = "takaya@ldeo.columbia.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'xdynsys': [pjoin('data', '*')]}
REQUIRES = ["numpy", "xarray", "dask"]
