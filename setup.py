import sys
import os
import subprocess

try:
    from setuptools import setup
except:
    from distutils.core import setup

# write git-hash
githash = subprocess.check_output(["git", "describe", "--always"]).strip()
with open('vis_sim/hash.txt', 'w') as f:
    f.write(str(githash))

# setup vars
setup(
    name            = 'vis_sim',
    version         = '0.1',
    license         = 'BSD',
    description     = 'radio interferometric visibility simulator',
    author          = 'Nick Kern',
    url             = "http://github.com/nkern/vis_sim",
    packages        = ['vis_sim']
    )


