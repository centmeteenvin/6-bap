import pathlib
from rag_chatbot.cache.cache import MODULE_CACHE_DIR, clearCache, createCacheSubDir


def test_cache():
    assert MODULE_CACHE_DIR.exists(), "The Cache dir should always exist when importing it"
    clearCache()
    assert not any(MODULE_CACHE_DIR.iterdir()), "Assure the cache dir is empty"
    fooPath = pathlib.Path("./foo")
    fooPath = createCacheSubDir(fooPath)
    assert fooPath.exists() and fooPath.is_dir(), "Assure the fake dir was creating"
    clearCache()
    assert not any(MODULE_CACHE_DIR.iterdir()), "Assure the cache dir is empty again"