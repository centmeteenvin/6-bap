import hashlib
from typing import Hashable
from .text_source import Reference, TextSource
from pypdf import PdfReader
import os

class PDFReference(Reference):
    """A reference to a pdf file"""
    def __init__(self, documentPath :str, page : int) -> None:
        super().__init__()
        assert os.path.exists(documentPath), "The document path does not exist"
        assert os.path.isabs(documentPath), "The document path is not absolute"
        assert page >= 0, "The page number is smaller then 0"
        self.documentPath = documentPath
        self.page = page

    def get(self) -> str:
        return f"""
PDF source: {self.documentPath}
Page: {self.page}
"""
    def __repr__(self) -> str:
        return self.get()

class PDFTextSource(TextSource):
    def __init__(self, filePath: str) -> None:
        super().__init__()
        assert os.path.exists(filePath), "The file should exist"
        assert os.path.isfile(filePath), "The file should not be a dir"
        self.filePath = os.path.abspath(filePath)


    @property
    def text(self) -> list[tuple[str, Reference]]:
        reader = PdfReader(self.filePath)
        output = []
        for page in reader.pages:
            text = page.extract_text()
            reference = PDFReference(self.filePath, page.page_number)
            output.append((text, reference))
        return output
    
    @property
    def id(self) -> Hashable:
        return hashlib.md5(self.filePath.encode()).hexdigest()