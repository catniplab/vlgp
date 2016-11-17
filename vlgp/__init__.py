import sys
import logging
import warnings


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('vlgp.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

logger.info('Module loaded')

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    logger.warning(str(sys.version_info))
    warnings.warn('Python 3.5 or later is required.')


from .api import *
# from .selection import *
