import re
from typing import List, Tuple

def read_file_input(file_path: str) -> str:
    """
    Reads text input from a file.

    Args:
    file_path (str): Path to the input file.

    Returns:
    str: Text content of the file.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError(f"File not found: {file_path}")

def read_string_input(input_string: str) -> str:
    """
    Reads text input from a string.

    Args:
    input_string (str): Input string.

    Returns:
    str: Input string (no processing needed).
    """
    return input_string

def tokenize_text(text: str) -> List[str]:
    """
    Tokenizes text into words.

    Args:
    text (str): Input text.

    Returns:
    List[str]: List of words.
    """
    return re.findall(r'\b\w+\b', text.lower())

def split_sentences(text: str) -> List[str]:
    """
    Splits text into sentences.

    Args:
    text (str): Input text.

    Returns:
    List[str]: List of sentences.
    """
    return re.split(r'[.!?]\s*', text)

def preprocess_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Preprocesses text by tokenizing and splitting sentences.

    Args:
    text (str): Input text.

    Returns:
    Tuple[List[str], List[str]]: Tuple of tokenized words and split sentences.
    """
    words = tokenize_text(text)
    sentences = split_sentences(text)
    return words, sentences

def process_file(file_path: str) -> Tuple[List[str], List[str], str]:
    """
    Processes text from a file.

    Args:
    file_path (str): Path to the input file.

    Returns:
    Tuple[List[str], List[str], str]: Tuple of tokenized words, split sentences, and original text.
    """
    text = read_file_input(file_path)
    words, sentences = preprocess_text(text)
    return words, sentences, text

def process_string(input_string: str) -> Tuple[List[str], List[str], str]:
    """
    Processes text from a string.

    Args:
    input_string (str): Input string.

    Returns:
    Tuple[List[str], List[str], str]: Tuple of tokenized words, split sentences, and original text.
    """
    text = read_string_input(input_string)
    words, sentences = preprocess_text(text)
    return words, sentences, text