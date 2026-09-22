import sys
from text_processor import process_file, process_string
from stats_calculator import calculate_stats

def main():
    """
    Entry point for the application.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <file_or_string>")
        sys.exit(1)

    input_data = sys.argv[1]

    try:
        if input_data.endswith('.txt'):
            text = process_file(input_data)
        else:
            text = process_string(input_data)
    except Exception as e:
        print(f"Error processing input: {e}")
        sys.exit(1)

    if not text:
        print("No text to process.")
        sys.exit(0)

    stats = calculate_stats(text)

    print("Text Statistics:")
    print(f"Words: {stats['word_count']}")
    print(f"Sentences: {stats['sentence_count']}")
    print(f"Characters: {stats['char_count']}")

if __name__ == "__main__":
    main()