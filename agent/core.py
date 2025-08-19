import os
import json
from typing import Dict, Any, Optional, List
from utils.confidence_scorer import calculate_confidence_scores

# Import your tools and schemas
from utils.tools import process_document
from agent.document_router import classify_document_type
from agent.extraction_chain import extract_document_data
from models.schemas import InvoiceSchema, MedicalBillSchema, PrescriptionSchema, ExtractedField, Source # Import all schemas

class DocumentExtractionAgent:
    def __init__(self):
        # Initialize any common resources or configurations here
        # For this agent, the OpenAI client is initialized within each function
        # but you could centralize it if preferred.
        pass

    def run_extraction(self, file_path: str, optional_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs the full document extraction process.

        Args:
            file_path: The path to the document file (PDF or image).
            optional_fields: A list of specific fields to extract (not yet fully implemented
                             for dynamic field extraction, but can be added later).

        Returns:
            A dictionary containing the extracted data, document type, confidence scores,
            and QA notes.
        """
        print(f"Starting extraction for: {file_path}")
        
        # 1. Document Ingestion and OCR
        try:
            document_data = process_document(file_path)
            raw_text = document_data.get("text", "")
            word_boxes = document_data.get("word_boxes", [])
            if not raw_text:
                return self._create_error_output("OCR failed to extract text.", "ocr_failed")
            print("OCR completed successfully.")
        except Exception as e:
            return self._create_error_output(f"Error during OCR processing: {e}", "ocr_error")

        # 2. Document Routing: Detect Document Type
        detected_doc_type = classify_document_type(raw_text)
        if detected_doc_type == 'unknown':
            return self._create_error_output(f"Could not classify document type.", "classification_failed")
        print(f"Document classified as: {detected_doc_type}")

        # Map detected type to the corresponding Pydantic schema
        document_schema = None
        if detected_doc_type == 'invoice':
            document_schema = InvoiceSchema
        elif detected_doc_type == 'medical_bill':
            document_schema = MedicalBillSchema
        elif detected_doc_type == 'prescription':
            document_schema = PrescriptionSchema
        else:
            # This case should ideally be caught by the 'unknown' check above
            return self._create_error_output(f"No schema defined for document type: {detected_doc_type}", "schema_missing")

        # 3. Information Extraction
        extracted_raw_data = {}
        try:
            print(f"Extracting data using {document_schema.__name__}...")
            extracted_raw_data = extract_document_data(raw_text, document_schema)
            # Validate extracted data against the Pydantic schema
            # This step ensures the LLM's output conforms to the schema's types
            document_model_instance = document_schema.model_validate(extracted_raw_data)
            extracted_raw_data = document_model_instance.model_dump() # Convert back to dict if needed
            print("Data extraction completed and validated against schema.")
        except Exception as e:
            return self._create_error_output(f"Error during data extraction or schema validation: {e}. Raw extracted: {extracted_raw_data}", "extraction_error")

        # 4. (Placeholder) Validation & Post-processing (e.g., sum checks, date formats)
        # This will be implemented in the next step.
        qa_notes = {"passed_rules": [], "failed_rules": [], "notes": "Validation not yet fully implemented."}
        
        # 5. (Placeholder) Confidence Scoring
        # This will be implemented in a later step.
        overall_confidence = 0.0 # Placeholder
        fields_with_confidence = self._format_extracted_data(extracted_raw_data, word_boxes)
        
        # Assemble the final output structure as per requirements
        final_output = {
            "doc_type": detected_doc_type,
            "fields": fields_with_confidence,
            "overall_confidence": overall_confidence, # Will be calculated later
            "qa": qa_notes
        }
        
        print("Extraction process finished.")
        return final_output

    def _format_extracted_data(self, extracted_data: Dict[str, Any], word_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats the extracted data into the 'fields' list as required by the prompt,
        including dummy confidence and source information for now.

        Args:
            extracted_data: The dictionary of extracted data from the LLM.
            word_boxes: List of word bounding boxes from OCR.

        Returns:
            A list of dictionaries, each representing an extracted field with name, value,
            confidence, and source.
        """
        formatted_fields = []
        for key, value in extracted_data.items():
            # Handle nested structures like line_items or medications
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                for i, item_dict in enumerate(value):
                    for sub_key, sub_value in item_dict.items():
                        # A very simplistic way to find bbox for nested items
                        # In a real scenario, this would involve more sophisticated logic
                        # to match extracted value to specific word_boxes.
                        bbox_for_value = self._find_bbox_for_value(str(sub_value), word_boxes)
                        formatted_fields.append(ExtractedField(
                            name=f"{key}.{i}.{sub_key}",
                            value=str(sub_value),
                            confidence=0.5, # Placeholder
                            source=Source(page=bbox_for_value['page_num'], bbox=bbox_for_value['bbox']) if bbox_for_value else None
                        ).model_dump())
            else:
                # Find bounding box for the top-level field
                bbox_for_value = self._find_bbox_for_value(str(value), word_boxes)
                formatted_fields.append(ExtractedField(
                    name=key,
                    value=str(value),
                    confidence=0.5, # Placeholder
                    source=Source(page=bbox_for_value['page_num'], bbox=bbox_for_value['bbox']) if bbox_for_value else None
                ).model_dump())
        return formatted_fields

    def _find_bbox_for_value(self, value: str, word_boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        A very simplistic heuristic to find a bounding box for an extracted value.
        This would need significant improvement for production use.
        It finds the first word box that contains the value as a substring.
        """
        if not value or not word_boxes:
            return None
        # Normalize value for searching (e.g., remove currency symbols, extra spaces)
        normalized_value = re.sub(r'[^\w\s.]', '', value).strip().lower()

        # Iterate through word_boxes to find a matching word or phrase
        for box_info in word_boxes:
            normalized_word = re.sub(r'[^\w\s.]', '', box_info['word']).strip().lower()
            if normalized_value in normalized_word or normalized_word in normalized_value:
                return {
                    "page_num": box_info['page_num'],
                    "bbox": box_info['bbox']
                }
        return None


    def _create_error_output(self, message: str, error_code: str) -> Dict[str, Any]:
        """Helper to create a standardized error output."""
        print(f"ERROR: {message}")
        return {
            "doc_type": "error",
            "fields": [],
            "overall_confidence": 0.0,
            "qa": {
                "passed_rules": [],
                "failed_rules": [error_code],
                "notes": message
            }
        }

