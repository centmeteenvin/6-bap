import os
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
from rag_chatbot.text_source.pdf_text_source import PDFTextSource


def testPDFTextExtraction():
    source: PDFTextSource = PDFTextSource("./test\\resources\\attention_is_all_you_need.pdf")
    result = source.text
    assert len(result) == 15, "Expect 15 pages to be read out"
    firstPage = result[0]
    assert firstPage[1].documentPath == "c:\\Users\\vince\\programmeren\\projects\\6-bap\\test\\resources\\attention_is_all_you_need.pdf"
    assert firstPage[1].page == 0
    

