import os
import json
from openai import OpenAI
from pydantic import BaseModel

# Import the schemas you created in models/schemas.py
from models.schemas import InvoiceSchema, MedicalBillSchema, PrescriptionSchema

# Load environment variables (API keys)
from dotenv import load_dotenv
load_dotenv()

def extract_document_data(document_text: str, document_schema: BaseModel) -> dict:
    """
    Extracts structured data from document text using an LLM based on a Pydantic schema.

    Args:
        document_text: The full text content of the document.
        document_schema: The Pydantic BaseModel schema for the document type.

    Returns:
        A dictionary containing the extracted data.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Convert the Pydantic schema to a JSON Schema string for the LLM
    schema_json = json.dumps(document_schema.model_json_schema())

    # Create the prompt for the LLM
    # The system prompt sets the context and role for the LLM
    system_prompt = (
        "You are a highly accurate data extraction agent. "
        "Your task is to extract information from the provided document text "
        "and return it as a JSON object that strictly conforms to the given JSON schema. "
        "Do not include any other text, explanations, or formatting. "
        "If a field is not found, use a null value unless the schema requires a specific type."
    )

    # The user prompt contains the document text and the JSON schema
    user_prompt = f"""
    Document Text:
    {document_text}

    JSON Schema:
    {schema_json}
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Recommended model for this task
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1 # Low temperature for consistent, factual output
        )
        
        # Parse the JSON string from the response
        extracted_data = json.loads(completion.choices[0].message.content)
        return extracted_data
    
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        return {}
