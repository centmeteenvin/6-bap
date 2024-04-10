import torch
import logging
from transformers import logging as hf_logging

hf_logging.set_verbosity_warning()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(f"logs./{__name__}.log")
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)