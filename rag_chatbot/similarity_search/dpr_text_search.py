import hashlib
import re
import torch
import os
from rag_chatbot.cache.cache import MODULE_CACHE_DIR
from rag_chatbot.secrets.secrets import Secrets
from rag_chatbot.similarity_search.text_search import QueryResult, TextSearch
from rag_chatbot.text_cleanup.text_cleanup import TextCleanup
from rag_chatbot.text_source.text_source import TextSource

from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from datasets import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DPREncoder():
    """This class contains two encoders, one to encode a question into embeddings and one for the context"""

    def __init__(self) -> None:
        """Loads the necessary encoders. One thing that is needed is that the hugging face token is passed"""
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.ctxTokenizer : DPRContextEncoderTokenizerFast = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", token=Secrets.hf_token())
        self.ctxEncoder : DPRContextEncoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", token=Secrets.hf_token())
        self.questionTokenizer : DPRContextEncoderTokenizerFast =  DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base", token=Secrets.hf_token())
        self.questionEncoder : DPRContextEncoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", token = Secrets.hf_token())
        self.ctxEncoder.to(self.device)
        self.questionEncoder.to(self.device)
        self.maxLength = 1024

    def encodeContext(self, data: str) -> torch.Tensor:
        """Encode the given data string with the context encoder and return it as a torch tensor located on cuda if possible"""
        with torch.no_grad():
            return self.ctxEncoder(
                **self.ctxTokenizer(
                    data, return_tensors="pt", max_length=self.maxLength, padding=True, truncation=True
                    ).to(self.device)
                )[0]
    
    def encodeQuestion(self, data: str) -> torch.Tensor:
        """Encode the given data string with the question encoder and return it as a torch tensor located on cuda if possible"""
        with torch.no_grad():
            return self.questionEncoder(
                **self.questionTokenizer(
                    data, return_tensors="pt", max_length=self.maxLength, padding=True, truncation=True
                    ).to(self.device)
                )[0]

class DPRTextSearch(TextSearch):
    """
    This type of search uses the Facebook DPR model in combination with a Faiss
    index on the dataset to do similarity search. The DPR embeddings
    calculations are expensive so they are cached behind the screens. You can
    run the clear_cache script to reduce the cache. We calculate two datasets,
    one that text and references and a second one that contains sentence level
    embeddings and a reference to the text and reference in the first dataset.
    """
    def __init__(self, source: TextSource, cleanup: TextCleanup) -> None:
        super().__init__(source, cleanup)
        self.dataset : Dataset = None
        self.embeddingsDataset : Dataset = None
        self.encoder = DPREncoder()
        # calculate the combined hash of the source and the cleanup to check if it is a new combination
        id = hashlib.md5((source.id + cleanup.id()).encode()).hexdigest()
        self.cache_path = MODULE_CACHE_DIR / id
        # If the cache path did not already exist we now we are handling a new case so we need to encode our text source.
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)
            self.preprocess()
            self.save()
        else:
            self.load()

    def preprocess(self) -> None:
        """This function preprocess the text source by adding the embeddings to it."""
        dataDict = {
            "text": [],
            "reference": [],
        }
        embeddingsDict = {
            "sentence": [],
            "refId": [],
        }
        sentenceEndings = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)(?=\s|$)')
        for i, (text, reference) in enumerate(self.text):
            dataDict["text"].append(text)
            dataDict["reference"].append(reference.asDict())
            for sentence in sentenceEndings.split(text):
                embeddingsDict["sentence"].append(sentence)
                embeddingsDict["refId"].append(i)
        self.dataset = Dataset.from_dict(dataDict)
        self.embeddingsDataset = Dataset.from_dict(embeddingsDict)
        self.embeddingsDataset = self.embeddingsDataset.map(lambda examples: {"embeddings" : self.encoder.encodeContext(examples["sentence"])}, batched=True, batch_size=16)
        self.embeddingsDataset.add_faiss_index("embeddings")

    def save(self) -> None:
        """This function saves the current datasets and their indices"""
        self.dataset.save_to_disk((self.cache_path / 'data.set').as_posix())
        self.embeddingsDataset.save_faiss_index('embeddings', (self.cache_path / 'embeddings.index').as_posix())
        self.embeddingsDataset.drop_index('embeddings')
        self.embeddingsDataset.save_to_disk((self.cache_path / 'embeddings.set').as_posix())
        self.embeddingsDataset.load_faiss_index('embeddings', (self.cache_path / 'embeddings.index').as_posix())

    def load(self) -> None:
        """This function loads the datasets from the cache_folder"""
        self.dataset = Dataset.load_from_disk((self.cache_path / 'data.set').as_posix())
        self.embeddingsDataset = Dataset.load_from_disk((self.cache_path / 'embeddings.set').as_posix())
        self.embeddingsDataset.load_faiss_index('embeddings', (self.cache_path / 'embeddings.index').as_posix())


    def findNCClosest(self, query: str, n: int) -> list[QueryResult]:
        assert n <= len(self.dataset), "n must be smaller then the amount of samples the dataset contains"

        # n refers to dataset samples but we have our index on the embeddings
        # dataset, therefore we need to keep searching larger and larger amounts
        # of the dataset until we have enough unique refId's
        encodedQuery = self.encoder.encodeQuestion(query).cpu().numpy()
        i = n
        while True:
            print(f"searching {i} examples")
            result = self.embeddingsDataset.get_nearest_examples('embeddings', encodedQuery, i)
            if len(set(result.examples['refId'])) >= n:
                break
            i = i*2
            
        # Now we know that the result variable contains enough samples to
        # reconstruct a dataset retrieval of length n. Now we need to find the refIds with the highest scores
        highestScoringRefIds = []
        highestScoringRefIdScores = []
        refIdsFound = 0
        for score, refId in zip(result.scores, result.examples["refId"]):
            if refId not in highestScoringRefIds:
                highestScoringRefIds.append(refId)
                highestScoringRefIdScores.append(score)
                refIdsFound += 1
                if refIdsFound == n:
                    break

        # Now we retrieve the corresponding texts and references.
        results = []
        for score, refId in zip(highestScoringRefIdScores, highestScoringRefIds):
            datasetRow = self.dataset[refId]
            results.append(
                QueryResult(datasetRow['text'], datasetRow['reference'], score)
            )
        return results



        
