"""This module extracts tokens from the environment into named variables"""
import dotenv
import os
dotenv.load_dotenv()
hf_token_env_name = "HUGGING_FACE_TOKEN"

class Secrets:
    """This class stores the necessary secrets as static properties so they can be loaded in lazily"""
    
    @staticmethod
    def hf_token() -> str:
        token = os.getenv(hf_token_env_name)
        if token is None or len(token) == 0:
            raise Exception(f"The {hf_token_env_name} was requested but not set in the env")
        return token