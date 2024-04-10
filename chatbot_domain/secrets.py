from dotenv import load_dotenv
from os import getenv
load_dotenv()
OPEN_AI_API = getenv('OPEN_AI_API')
HUGGINGFACE_ACCESS = getenv('HUGGINGFACE_ACCESS')