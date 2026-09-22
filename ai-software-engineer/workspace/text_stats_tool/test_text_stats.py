import pytest
from text_processor import process_file, process_string
from stats_calculator import calculate_word_count, calculate_sentence_count, calculate_character_count, calculate_stats

def test_empty_input():
    assert calculate_word_count("") == 0
    assert calculate_sentence_count("") == 0
    assert calculate_character_count("") == 0

def test_single_word_input():
    assert calculate_word_count("hello") == 1
    assert calculate_sentence_count("hello") == 0
    assert calculate_character_count("hello") == 5

def test_single_sentence_input():
    assert calculate_word_count("Hello world!") == 2
    assert calculate_sentence_count("Hello world!") == 1
    assert calculate_character_count("Hello world!") == 13

def test_multiple_sentences_input():
    text = "Hello world! This is a test. Testing 1, 2, 3."
    assert calculate_word_count(text) == 9
    assert calculate_sentence_count(text) == 3
    assert calculate_character_count(text) == 46

def test_file_input(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello world! This is a test. Testing 1, 2, 3.")
    assert calculate_word_count(process_file(file_path)) == 9
    assert calculate_sentence_count(process_file(file_path)) == 3
    assert calculate_character_count(process_file(file_path)) == 46

def test_string_input():
    text = "Hello world! This is a test. Testing 1, 2, 3."
    assert calculate_word_count(process_string(text)) == 9
    assert calculate_sentence_count(process_string(text)) == 3
    assert calculate_character_count(process_string(text)) == 46

def test_calculate_stats():
    text = "Hello world! This is a test. Testing 1, 2, 3."
    stats = calculate_stats(text)
    assert stats["word_count"] == 9
    assert stats["sentence_count"] == 3
    assert stats["character_count"] == 46

def test_invalid_input():
    with pytest.raises(TypeError):
        calculate_word_count(123)
    with pytest.raises(TypeError):
        calculate_sentence_count(123)
    with pytest.raises(TypeError):
        calculate_character_count(123)