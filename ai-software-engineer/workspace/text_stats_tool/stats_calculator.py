import re
from text_processor import preprocess_text

def count_words(text):
    """
    Counts the number of words in the given text.

    Args:
        text (str): The text to count words from.

    Returns:
        int: The number of words in the text.
    """
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def count_sentences(text):
    """
    Counts the number of sentences in the given text.

    Args:
        text (str): The text to count sentences from.

    Returns:
        int: The number of sentences in the text.
    """
    sentences = re.split(r'[.!?]', text)
    return len([s for s in sentences if s.strip()])

def count_characters(text):
    """
    Counts the number of characters in the given text.

    Args:
        text (str): The text to count characters from.

    Returns:
        int: The number of characters in the text.
    """
    return len(text)

def calculate_statistics(text):
    """
    Calculates word, sentence, and character counts for the given text.

    Args:
        text (str): The text to calculate statistics for.

    Returns:
        dict: A dictionary containing the word count, sentence count, and character count.
    """
    if not text:
        return {'word_count': 0, 'sentence_count': 0, 'character_count': 0}

    word_count = count_words(text)
    sentence_count = count_sentences(text)
    character_count = count_characters(text)

    return {'word_count': word_count, 'sentence_count': sentence_count, 'character_count': character_count}

def main(text):
    """
    Calculates and prints the text statistics for the given text.

    Args:
        text (str): The text to calculate statistics for.
    """
    text = preprocess_text(text)
    stats = calculate_statistics(text)
    print(f"Word count: {stats['word_count']}")
    print(f"Sentence count: {stats['sentence_count']}")
    print(f"Character count: {stats['character_count']}")

if __name__ == "__main__":
    main("This is a test. This test is only a test.")