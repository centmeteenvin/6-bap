import torch
from rag_chatbot.secrets.secrets import Secrets
from rag_chatbot.similarity_search.text_search import TextSearch
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import TextSource

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
import numpy

class DPREncoder():
    """This class contains two encoders, one to encode a question into embeddings and one for the context"""

    def __init__(self) -> None:
        """Loads the necessary encoders. One thing that is needed is that the hugging face token is passed"""
        self.ctxTokenizer : DPRContextEncoderTokenizerFast = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", token=Secrets.hf_token())
        self.ctxEncoder : DPRContextEncoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", token=Secrets.hf_token())
        self.questionTokenizer : DPRContextEncoderTokenizerFast =  DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base", token=Secrets.hf_token())
        self.questionEncoder : DPRContextEncoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", token = Secrets.hf_token())

    def encodeContext(self, data: str) -> numpy.ndarray:
        """Encode the given data string with the context encoder and return it as a numpy array"""
        with torch.no_grad():
            return self.ctxEncoder(**self.ctxTokenizer(data, return_tensors="pt"))[0][0].numpy()
    
    def encodeQuestion(self, data: str) -> numpy.ndarray:
        """Encode the given data string with the question encoder and return it as a numpy array"""
        with torch.no_grad():
            return self.questionEncoder(**self.questionTokenizer(data, return_tensors="pt"))[0][0].numpy()

class DPRTextSearch(TextSearch):
    """
    This type of search uses the Facebook DPR model in combination with a Faiss index on the dataset to do similarity search.
    The DPR embeddings calculations are expensive so they are cached behind the screens. You can run the clear_cache script to reduce the cache.
    """
    def __init__(self, source: TextSource, cleanup: TextCleanup) -> None:
        super().__init__(source, cleanup)