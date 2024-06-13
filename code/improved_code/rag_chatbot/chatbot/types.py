from typing import Generator

Stream = Generator[str, None, None]
"""This represents a synchronous stream of string bits, concatenation without separators should equal the normal response output"""
