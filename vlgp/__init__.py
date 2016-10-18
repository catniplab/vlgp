import sys
import warnings

from .core import *
from .selection import *

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    warnings.warn('Python >= 3.5 is required.')
