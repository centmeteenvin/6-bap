import re
from copy import deepcopy
from random import shuffle
from pypdf import PdfReader, PageObject
from . import logger
from datasets import Dataset, DatasetDict, load_from_disk

Paragraph = str
Sentence = str

def parseData(filePath: str, startPage: int = 1, endPage: int = None) -> DatasetDict:
    """
    takes a filePath and a range of pages.
    returns a 2 lists, one containing sentences and another alineas.
    """
    logger.info("Reading pdf-file")
    reader: PdfReader = PdfReader(filePath)
    pages : list[PageObject] = reader.pages
    _endPage = endPage
    if endPage is None:
        _endPage = len(pages)
    pages = pages[startPage - 1: _endPage - 1]
    sentenceCounter = 0
    result : list[tuple[Paragraph, list[Sentence]]] = []
    for page in pages:
        text: str = page.extract_text()
        text = ''.join((char for char in text if char.isalnum() or char.isspace() or char in '.?!\n'))
        text = re.sub(r'(?<=[^.?!])\n', ' ', text) # remove all the newlines that are not preceded by a .?!
        paragraphs = re.split('[\n]', text) # split on the remaining newlines aka newlines at the end of a sentence = paragraph
        for paragraph in paragraphs:
            sentences = re.split(r'[.?!]', paragraph)  # split on punctuation aka split the sentences.
            sentenceCounter += len(sentences)
            result.append((paragraph, sentences))
        
    logger.info(f"Finished reading pdf.\nExtracted {len(result)} paragraphs and {sentenceCounter} sentences")
    return result


def _dataEntriesToDataSet(data: list[tuple[Paragraph, list[Sentence]]]) -> Dataset:
    dictionary = {'sentence': [], 'paragraph': []}
    for paragraph, sentences in data:
        for sentence in sentences:
            dictionary['sentence'].append(sentence)
            dictionary['paragraph'].append(paragraph)
    return Dataset.from_dict(dictionary)
        

def createDataSet(data: list[tuple[Paragraph, list[Sentence]]], evaluationShare: float) -> DatasetDict:
    """
    Data is a list of strings,
    this is shuffled and then split according to the evaluation share into a training and evaluation dataset.
    will return a DatasetDict with 'train' and 'evaluation' keys,
    data will be copied and not be modified directly 
    """
    logger.info("creating dataset")
    dataCopy = deepcopy(data)
    shuffle(dataCopy)
    splitIndex = int(len(dataCopy) * evaluationShare)
    evaluationData, trainingData = dataCopy[:splitIndex], dataCopy[splitIndex:]
    return DatasetDict({
        'train': _dataEntriesToDataSet(trainingData),
        'evaluation': _dataEntriesToDataSet(evaluationData)
    })

def saveDatasetDict(data: DatasetDict, location: str) -> None:
    """Saves a dataset dict object to a location, it should not contain any indices"""
    data.save_to_disk(location)
    
def saveDataset(data: Dataset, location: str, index : str | None = None) -> None:
    """
    Save a dataset object to a location, if it is indexed, the index parameter should be the column name of the index.
    """
    if index is not None:
        data.save_faiss_index(index, location + "/faiss.index")
        data.drop_index(index)
    data.save_to_disk(location)
    if index is not None:
        data.load_faiss_index(index, location + "/faiss.index")
    
def loadDataSetDictFromDisk(filePath: str) -> DatasetDict:
    logger.info(f"Loading dataset dict from {filePath}")
    return DatasetDict.load_from_disk(filePath)

def loadDataSetFromDisk(filepath: str) -> Dataset:
    logger.info(f"Loading dataset from {filepath}")
    return load_from_disk(filepath)