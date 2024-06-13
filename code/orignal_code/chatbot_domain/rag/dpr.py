from numpy.core.multiarray import array as array
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from abc import ABC, abstractmethod
import numpy
class DPR(ABC):
    """
    DPR is responsible for:
    1. encode questions to prepare them for similarity search.
    2. encode context (knowledge) to prepare them for similarity search.
    """
    @abstractmethod
    def encodeContext(self, context) -> numpy.array:
        """
        Encodes the context.
        """
        pass
    
    @abstractmethod
    def encodeQuestion(self, question) -> numpy.array:
        """
        Encodes the question.
        """
        pass
    
class FacebookDPR(DPR):
    
    def __init__(self) -> None:
        super().__init__()
        self._contextEncoder : DPRContextEncoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", device_map = 'cuda')
        self._contextTokenizer : DPRContextEncoderTokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", device_map = 'cuda')
        
        self._questionEncoder: DPRQuestionEncoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", device_map = 'cuda')
        self._questionTokenizer : DPRQuestionEncoderTokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base", device_map = 'cuda')
        
    def encodeContext(self, context: list[str]) -> numpy.array:
        # encoded = []
        # for item in context:
        #     tokenizedContext = self._contextTokenizer(item, return_tensors='pt', truncation=True).to('cuda')
        #     encodedContext = self._contextEncoder(**tokenizedContext)[0][0]
        #     encoded.append(encodedContext.cpu().detach().numpy())
        # return encoded
        return self._contextEncoder(**self._contextTokenizer(context, return_tensors = "pt", truncation=True).to('cuda'))[0][0].cpu().detach().numpy()

    def encodeQuestion(self, question) -> numpy.array:
        return self._questionEncoder(**self._questionTokenizer(question, return_tensors='pt', truncation=True).to('cuda'))[0][0].cpu().detach().numpy()