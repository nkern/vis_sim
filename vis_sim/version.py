"""
version.py
---------


"""
import sys
import os
import subprocess
import json


def get_version_info():
    vis_sim_dir = os.path.dirname(os.path.realpath(__file__))

    try:
	git_hash = subprocess.check_output(['git', '-C', vis_sim_dir, 'describe', '--always'], stderr=subprocess.STDOUT).strip()
    except:
        try:
            git_file = os.path.join(vis_sim_dir, 'GIT_INFO')
            with open(git_file) as f:
                data = json.load(f)
            git_hash = data[0].encode('UTF8')
        except:
            git_hash = ''


    version_info = {'git_hash': git_hash}

    return version_info


version_info = get_version_info()
git_hash = version_info['git_hash']

