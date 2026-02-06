"""Main entry point for the application."""

from utils import process_data
from config import DATABASE_URL, API_KEY


def main():
    """Main function to run the application."""
    print("Starting application...")
    print(f"Database URL: {DATABASE_URL}")
    print(f"API Key: {API_KEY}")
    
    # Test the process_data function
    test_data = {"name": "test", "value": 42}
    result = process_data(test_data)
    
    print("\nProcessed data:")
    print(result)


if __name__ == "__main__":
    main()