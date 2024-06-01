import re
import string
from rag_chatbot.text_cleanup.text_cleanup import TextTransformer


class TrimTextTransformer(TextTransformer):
    """Removes all leading and trailing whitespace"""
    def transform(self, text: str) -> str:
        return text.strip()

class AlphaNumericalTextTransformer(TextTransformer):
    """Removes all none alpha numerical characters from the string, it does retain punctuation"""
    def transform(self, text: str) -> str:
        pattern = f"[^{re.escape(string.ascii_letters + string.digits + string.whitespace + string.punctuation)}]"
        return re.sub(pattern, '', text)
