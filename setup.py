import sys
import os
from setuptools import setup
import json
from vis_sim import version

data = [version.git_hash]
with open(os.path.join('vis_sim', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

# setup vars
setup(
    name            = 'vis_sim',
    version         = '0.1',
    license         = 'BSD',
    description     = 'radio interferometric visibility simulator',
    author          = 'Nick Kern',
    url             = "http://github.com/nkern/vis_sim",
    include_package_data = True,
    package_data    = {'vis_sim': ['GIT_INFO']},
    packages        = ['vis_sim']
    )


