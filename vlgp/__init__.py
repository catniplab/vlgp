import sys
import logging
import warnings


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# create a file file_handler
file_handler = logging.FileHandler('vlgp.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.INFO)
# logger.addHandler(stdout_handler)

logger.info('Module loaded')

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    logger.warning(str(sys.version_info))
    warnings.warn('Python 3.5 or later is required.')


from .api import *
# from .selection import *
