import shutil
import appdirs
import pathlib

MODULE_CACHE_DIR = pathlib.Path(appdirs.user_cache_dir(appname="rag_chatbot"))

# If the path does not exist create it.
if not MODULE_CACHE_DIR.exists():
    MODULE_CACHE_DIR.mkdir(parents=True)

def createCacheSubDir(path: pathlib.Path) -> pathlib.Path:
    """
    The given RELATIVE path is created in the cache directory, it returns the absolute path.
    If the given path needs to create multiple paths, it will do so. 
    """
    assert not path.is_absolute(), "The path should be relative"
    assert MODULE_CACHE_DIR.exists(), "The cache dir should exist"
    absPath = MODULE_CACHE_DIR / path
    absPath.mkdir(parents=True, exist_ok=True)
    return absPath

def get_size(path: pathlib.Path) -> int:
    """
    Calculate the total size of a given path (file or directory).
    
    :param path: Path object (file or directory)
    :return: Size in bytes
    """
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
    else:
        raise ValueError(f"The path '{path}' is neither a file nor a directory")

def clear_directory(directory):
    """
    Remove all contents of a directory without deleting the directory itself.

    :param directory: Path object of the directory to clear
    """
    if not directory.is_dir():
        raise ValueError(f"The path '{directory}' is not a directory")
    
    for item in directory.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def clearCache(needsConfirmation = False) -> None:
    """This function will clear the entire cache directory, but only it's contents"""
    if needsConfirmation:
        question = f"Do you want to delete {MODULE_CACHE_DIR} which contains {get_size(MODULE_CACHE_DIR)/(1024*1024):0.4f} MB?\n[yes/no]: "
        confirmation = input(question)
        while confirmation not in ["yes", "no"]:
            confirmation = input(question)
        if confirmation == "no":
            return
    clear_directory(MODULE_CACHE_DIR)    
