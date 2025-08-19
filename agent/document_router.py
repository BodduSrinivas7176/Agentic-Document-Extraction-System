import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

def classify_document_type(document_text: str) -> str:
    """
    Classifies the type of the document (invoice, medical_bill, or prescription)
    based on its text content using an LLM.

    Args:
        document_text: The full text content of the document.

    Returns:
        A string representing the detected document type ('invoice', 'medical_bill',
        'prescription'), or 'unknown' if classification fails.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Use a snippet of the document text for classification to save tokens
    # and focus on overall content rather than details.
    text_snippet = document_text[:2000] # Take first 2000 characters

    # System prompt to instruct the LLM on its role
    system_prompt = (
        "You are an expert document classifier. Your task is to accurately identify "
        "the type of document from the provided text. Choose from 'invoice', 'medical_bill', "
        "or 'prescription'. Respond with ONLY the document type, no other text or explanation."
    )

    # User prompt providing the document text snippet
    user_prompt = f"""
    Analyze the following document text and determine its type:

    Document Text:
    {text_snippet}

    Document Type (invoice, medical_bill, or prescription):
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Using gpt-4o-mini for classification
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0 # Extremely low temperature for deterministic classification
        )

        detected_type = completion.choices[0].message.content.strip().lower()

        # Validate the detected type against expected categories
        allowed_types = ['invoice', 'medical_bill', 'prescription']
        if detected_type in allowed_types:
            print(f"Detected document type: {detected_type}")
            return detected_type
        else:
            print(f"LLM returned an unexpected type: {detected_type}. Defaulting to 'unknown'.")
            return 'unknown'

    except Exception as e:
        print(f"An error occurred during document type classification: {e}")
        return 'unknown'

