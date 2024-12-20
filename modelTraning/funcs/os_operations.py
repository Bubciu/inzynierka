from os import listdir
from platform import system

# Hubert is the only person here who uses MacOS and have issues with paths, hence this oh so important variable
IS_WINDOWS = True if system() == 'Windows' else False   # just don't use it
# Apparently doesn't work as well... Sad, but a sacrifice I'm willing to take


def files_in_directory(dir_path: str, file_extension: str):
    """
    Functions finds all files with specified extension in specified location
    :param dir_path: path to directory
    :param file_extension: extension of seeking files
    :return: list of files in string format
    """
    return [file for file in listdir(dir_path) if file.endswith(file_extension)]
