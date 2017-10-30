# load modules
from .beam_model import Beam_Model
from .sky_model import Sky_Model
from .vis_sim import Vis_Sim

# try to load githash
try:
    with open('hash.txt', 'r') as f:
        __githash__ = f.read()
except:
    __githash__ = ''
