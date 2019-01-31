from .api import *

import sys
import logging
import warnings

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    logger.warning(str(sys.version_info))
    warnings.warn("Python 3.5 or later is required.")
