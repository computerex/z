"""Utility functions for data processing."""

from typing import Dict, Any, List, Optional
from config import DATABASE_URL, API_KEY


def validate_data(data: Any, required_keys: Optional[List[str]] = None) -> bool:
    """Validate that data is a dictionary and contains required keys.
    
    This function checks if the input data is a dictionary and optionally
    verifies that it contains all specified required keys. Useful for
    ensuring data integrity before processing.
    
    Args:
        data: The data to validate. Can be any type.
        required_keys: Optional list of keys that must be present in the data.
                      If None, only checks that data is a dictionary.
    
    Returns:
        bool: True if data is a dictionary and contains all required keys,
              False otherwise.
              
    Example:
        >>> validate_data({"name": "test"}, ["name"])
        True
        >>> validate_data({"name": "test"}, ["name", "age"])
        False
        >>> validate_data("not a dict")
        False
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys is None:
        return True
    
    return all(key in data for key in required_keys)


def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process the input data using configuration values.
    
    This function takes raw data and enhances it with configuration information
    from the application settings. It creates a new dictionary containing the
    original data along with metadata about the database connection and API key.
    
    Args:
        data: The raw data dictionary to be processed. Can contain any
              key-value pairs relevant to the application.
        
    Returns:
        Dict[str, Any]: A new dictionary containing:
            - 'original': The input data unchanged
            - 'database': The DATABASE_URL from config
            - 'api_key_prefix': First 7 characters of API_KEY for display
            - 'status': A string indicating processing was completed
            
    Example:
        >>> process_data({"name": "test", "value": 42})
        {'original': {'name': 'test', 'value': 42}, 
         'database': 'postgresql://localhost:5432/mydb',
         'api_key_prefix': 'sk-1234',
         'status': 'processed'}
    """
    # Validate input data before processing
    if not validate_data(data):
        raise ValueError("Data must be a dictionary")
    
    # Simulate processing with config values
    processed = {
        "original": data,
        "database": DATABASE_URL,
        "api_key_prefix": API_KEY[:7],
        "status": "processed"
    }
    return processed