**Text Statistics Tool**
=========================

A Python application for calculating word, sentence, and character counts in text input.

**Description**
---------------

This project provides a modular text statistics tool that takes user input (file or string) and calculates various statistics, including word count, sentence count, and character count. The application is designed with a separate file for text processing, statistics calculation, and testing.

**Installation/Setup**
----------------------

To use this project, simply clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/text_stats_tool.git
cd text_stats_tool
pip install -r requirements.txt
```

**Running the Application**
---------------------------

To run the application, execute the `main.py` file and provide the input text:

```bash
python main.py --help
```

Usage:
```bash
python main.py [-f FILE | -s STRING]
```

* `-f FILE`: Provide a file path as input
* `-s STRING`: Provide a string as input

Example:
```bash
python main.py -f example.txt
```

**Running Tests**
-----------------

To run the unit tests, execute the `test_text_stats.py` file:

```bash
python test_text_stats.py
```

**File Descriptions**
----------------------

### text_processor.py

Contains functions to read and preprocess text input, including tokenization and sentence splitting. This file includes:

* `read_file(input_file)`: Reads text from a file
* `read_string(input_string)`: Reads text from a string
* `preprocess_text(text)`: Tokenizes and splits text into sentences

### stats_calculator.py

Calculates word, sentence, and character counts based on the preprocessed text. This file includes:

* `calculate_word_count(text)`: Calculates the word count
* `calculate_sentence_count(text)`: Calculates the sentence count
* `calculate_character_count(text)`: Calculates the character count
* `calculate_all_stats(text)`: Calculates all statistics

### main.py

Entry point for the application. Takes user input (file or string), calls `text_processor` to preprocess the text, and then calls `stats_calculator` to calculate the statistics. Prints the results to the console.

### test_text_stats.py

Unit tests for the `text_processor` and `stats_calculator` functions. Includes tests for edge cases, such as empty input, single-word input, and input with multiple sentences.