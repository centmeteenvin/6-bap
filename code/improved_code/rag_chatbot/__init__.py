from .text_source import *  # noqa: E402, F403
from .text_cleanup import *  # noqa: E402, F403
from .similarity_search import *  # noqa: F403
from .cache import MODULE_CACHE_DIR as MODULE_CACHE_DIR
from .chatbot import *  # noqa: F403
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
