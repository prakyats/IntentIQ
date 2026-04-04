import re
import string

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase, removing special characters,
    and removing extra spaces.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters (punctuation)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Remove digits (optional, but often helpful for intent)
    text = re.sub(r"\d+", " ", text)
    
    # Remove extra spaces and leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

if __name__ == "__main__":
    test_text = "  Need pricing details urgently! @#$% 123  "
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{clean_text(test_text)}'")
