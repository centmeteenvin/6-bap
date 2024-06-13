import pathlib
from rag_chatbot.text_source.html_text_source import HTMLTextSource
from rag_chatbot.text_source.pdf_text_source import PDFTextSource
import logging

import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()



def testPDFTextExtraction():
    source: PDFTextSource = PDFTextSource("./test\\resources\\attention_is_all_you_need.pdf")
    result = source.text
    assert len(result) == 15, "Expect 15 pages to be read out"
    firstPage = result[0]
    assert firstPage[1].documentPath == "c:\\Users\\vince\\programmeren\\projects\\6-bap\\test\\resources\\attention_is_all_you_need.pdf"
    assert firstPage[1].page == 0
    

def testHTMLTextExtraction(caplog):
    # caplog.set_level(logging.DEBUG)
    with pytest.raises(Exception):
        HTMLTextSource("https://foo.bar", ["/"])
    
    url = "https://arxiv.org/" # with trailing /
    with pytest.raises(AssertionError):
        source = HTMLTextSource(url, ["/"])
    url = "https://arxiv.org"
    with pytest.raises(AssertionError):
        source = HTMLTextSource(url, ["+"])
    url = "https://arxiv.org"
    paths = ["/html/1706.03762v7"]
    source = HTMLTextSource(url, paths)
    text = source.text
    logger.debug(text)
    assert text[0][1].url == url+paths[0], "reference urls must match"
    assert "attention" in text[0][0], "I hope the word attention is present in the text extraction of the attention is all you need paper :)"

def testTextSourceId():
    textSourceHashPath = pathlib.Path("./test/resources/text_source_hashes")
    if not textSourceHashPath.exists():
        textSourceHashPath.mkdir(parents=True)
    
    pdfSource = PDFTextSource('./test/resources/attention_is_all_you_need.pdf')
    pdfSourceHashPath = textSourceHashPath / "pdf_source_hash.txt"
    htmlSource = HTMLTextSource("https://kokodobu.org", "/")
    htmlSourceHashPath = textSourceHashPath / "html_source_hash.txt"
    if len(list(textSourceHashPath.iterdir())) == 0:
        with open(pdfSourceHashPath, 'w') as f:
            f.write(pdfSource.id)
        with open(htmlSourceHashPath, 'w') as f:
            f.write(htmlSource.id)
        assert False, "This test had no correct setup and must be run at least twice"

    with open(pdfSourceHashPath, 'r') as f:
        pdfSourceId = f.read()
    with open(htmlSourceHashPath, 'r') as f:
        htmlSourceId = f.read()

    assert pdfSourceId == pdfSource.id
    assert htmlSourceId == htmlSource.id
    assert pdfSourceId != htmlSourceId
    
    
