import sys
import os
try:
    from setuptools import setup
except:
    from distutils.core import setup

setup(
    name            = 'vis_sim',
    version         = '0.1',
    license         = 'BSD',
    description     = 'radio interferometric visibility simulator',
    author          = 'Nick Kern',
    url             = "http://github.com/nkern/vis_sim",
    packages        = ['vis_sim']
    )


