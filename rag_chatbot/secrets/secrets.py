"""This module extracts tokens from the environment into named variables"""
import dotenv
import os
dotenv.load_dotenv()
hf_token_env_name = "HUGGING_FACE_TOKEN"
open_ai_key_name = "OPEN_AI_API_KEY"

class Secrets:
    """This class stores the necessary secrets as static properties so they can be loaded in lazily"""
    
    @staticmethod
    def hf_token() -> str:
        return Secrets._getEnv(hf_token_env_name)
    
    @staticmethod
    def openAIKey() -> str:
        return Secrets._getEnv(open_ai_key_name)

    @staticmethod
    def _getEnv(envName: str) -> str:
        """Retrieves and checks a key from the environment"""
        env = os.getenv(envName) 
        if env is None or len(env) == 0:
            raise Exception(f"The {envName} was requested but not set in the env")
        return env