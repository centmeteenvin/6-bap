import os
from rag_chatbot.secrets.secrets import Secrets, hf_token_env_name

def test_hf_token():
    current_hf_variable = os.getenv(hf_token_env_name)
    os.environ[hf_token_env_name] = "foo"
    assert Secrets.hf_token() == "foo"
    if current_hf_variable is not None:
        os.environ[hf_token_env_name] = current_hf_variable