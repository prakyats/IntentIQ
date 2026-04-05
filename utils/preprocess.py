import re
import string

def clean_text(text: str) -> str:
    """
    Cleans the input text by converting to lowercase, removing generic special characters,
    and removing extra spaces. Preserves keywords by using a surgical punctuation removal.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation but keep alphanumeric and spaces
    # This preserves keywords like "price", "demo", "error" without special char interference
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra spaces and leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

if __name__ == "__main__":
    test_text = "  Need pricing details urgently! @#$% 123  "
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{clean_text(test_text)}'")
