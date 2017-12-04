#!/usr/bin/env python
import distutils
from distutils.core import setup

description = "LSS related DESC Python utilities"

setup(name="lssutils",
      version="0.1",
      description=description,
      url="https://github.com/LSSTDESC/LSS_utils",
      author="Javier Sanchez for DESC",
      author_email="francs1@uci.edu",
      packages=['lssutils'])

