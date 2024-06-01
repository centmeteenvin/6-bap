import hashlib
from typing import Hashable
from bs4 import BeautifulSoup
import requests
import urllib.request
from rag_chatbot.text_source.text_source import Reference, TextSource
import logging
logger = logging.getLogger(__name__)

class HTMLReference(Reference):
    """This class defines the reference for html sources"""
    def __init__(self, url: str) -> None:
        super().__init__()
        assert isinstance(url, str)
        self.url = url
    def get(self) -> str:
        return f"URL: {self.url}"
    

class HTMLTextSource(TextSource):
    """
    This Text source is able to fetch text from an html site
    This class takes two parameters, a website and a list of paths it needs to extract the data from
    """
    def __init__(self, host:str, paths: list[str]) -> None:
        """
        The host is the base of the url, the paths is a list of paths that must be visited.
        The host should not include a trailing '/' and the paths should start with a '/'
        """
        super().__init__()
        # Test if the site exists
        assert isinstance(host, str), "The url must be a string"
        assert not host.endswith('/'), "The host URL should not end in a /"
        for path in paths:
            assert path.startswith('/'), "The path should start with a /"
            assert urllib.request.urlopen(host + path).getcode() == 200, "The site must be available"
        self.host = host
        self.paths = paths

    @property
    def text(self) -> list[tuple[str, Reference]]:
        try:
            result = []
            for path in self.paths:
                response : requests.Response= requests.get(self.host + path)
                if response.status_code != 200:
                    raise Exception("The status code was not 200", response)
                body = response.text
                logger.debug(body)
                soup = BeautifulSoup(body, features="html.parser")
                # kill all script and style elements
                for script in soup(["script", "style"]):
                    script.extract()    # rip it out
                text = soup.get_text()
                result.append((text, HTMLReference(self.host+path)))
            return result
        except Exception as e:
            logger.error("An exception occurred while fetching html page text")
            raise e
        
    @property
    def id(self) -> Hashable:
        return hashlib.md5((self.host + "".join(self.paths)).encode()).hexdigest()