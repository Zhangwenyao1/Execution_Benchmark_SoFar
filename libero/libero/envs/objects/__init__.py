import re

from libero.libero.envs.base_object import OBJECTS_DICT, VISUAL_CHANGE_OBJECTS_DICT

from .hope_objects import *
from .google_scanned_objects import *
from .articulated_objects import *
from .turbosquid_objects import *
from .site_object import SiteObject
from .target_zones import *

import sys
# from Open6Dor.SpatialObject import *
if len(sys.argv) > 1 and sys.argv[1] == "sofar":
    from sofar_execution.SpatialObject import *
    print("Successfully imported spatialobject.")
    sys.argv.pop(1)
else:
    print("spatialobject library not imported. Pass '1' as argument to import it.")




def get_object_fn(category_name):
    return OBJECTS_DICT[category_name.lower()]


def get_object_dict():
    return OBJECTS_DICT
