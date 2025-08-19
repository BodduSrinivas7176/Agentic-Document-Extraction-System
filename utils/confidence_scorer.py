import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import re # Added for _find_bbox_for_value

def calculate_confidence_scores(
    extracted_data: Dict[str, Any],
    qa_results: Dict[str, Any],
    llm_raw_outputs: List[Dict[str, Any]], # Raw LLM outputs for self-consistency check
    word_boxes: List[Dict[str, Any]] # For OCR confidence if available
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Calculates confidence scores for each extracted field and an overall document confidence.

    Args:
        extracted_data: The final, validated dictionary of extracted data.
        qa_results: The dictionary containing passed_rules and failed_rules from validation.
        llm_raw_outputs: A list of raw dictionaries from multiple LLM extraction runs
                         for self-consistency check. Expected to contain the same structure
                         as extracted_data, but for multiple runs.
        word_boxes: List of word bounding box info from OCR, potentially containing OCR confidence.

    Returns:
        A tuple:
        - List of dictionaries, each representing an ExtractedField with updated confidence.
        - Overall document confidence score (float).
    """
    
    # Calculate LLM Consistency Score for each field
    # This requires running the LLM multiple times (e.g., 3-5 times) for the same input
    # and comparing the extracted values.
    llm_consistency_scores = _calculate_llm_consistency(extracted_data, llm_raw_outputs)

    # Convert extracted_data into the list of fields format with initial formatting
    # This is a bit of a re-formatting step to match the desired output
    formatted_fields_with_confidence: List[Dict[str, Any]] = []

    # Iterate through the final extracted_data (from the main successful LLM run)
    # and populate confidence and source info.
    for key, value in extracted_data.items():
        # Handle nested structures like line_items or medications
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            for i, item_dict in enumerate(value):
                for sub_key, sub_value in item_dict.items():
                    field_name = f"{key}.{i}.{sub_key}"
                    confidence = _calculate_field_composite_confidence(
                        field_name,
                        str(sub_value), # Value as string for consistency check
                        llm_consistency_scores.get(field_name, 0.0), # Default 0 if not in consistency
                        qa_results,
                        word_boxes # Pass word_boxes for potential OCR confidence
                    )
                    
                    # Placeholder for source: In a real app, you'd match sub_value to its bbox from OCR
                    # For now, we'll use a very basic bbox lookup
                    source_info = _find_bbox_for_value(str(sub_value), word_boxes)

                    formatted_fields_with_confidence.append({
                        "name": field_name,
                        "value": str(sub_value), # Ensure value is string
                        "confidence": round(confidence, 2), # Round for display
                        "source": source_info # Source with bbox
                    })
        else:
            field_name = key
            confidence = _calculate_field_composite_confidence(
                field_name,
                str(value), # Value as string for consistency check
                llm_consistency_scores.get(field_name, 0.0), # Default 0 if not in consistency
                qa_results,
                word_boxes # Pass word_boxes for potential OCR confidence
            )

            # Placeholder for source: In a real app, you'd match value to its bbox from OCR
            source_info = _find_bbox_for_value(str(value), word_boxes)

            formatted_fields_with_confidence.append({
                "name": field_name,
                "value": str(value), # Ensure value is string
                "confidence": round(confidence, 2), # Round for display
                "source": source_info # Source with bbox
            })

    # Calculate overall confidence
    overall_confidence = _calculate_overall_confidence(formatted_fields_with_confidence, qa_results)

    return formatted_fields_with_confidence, overall_confidence


def _calculate_llm_consistency(extracted_data: Dict[str, Any], llm_raw_outputs: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates consistency score for each field based on multiple LLM runs.
    A simple approach: count how many times the primary extracted value appears.
    """
    consistency_scores = {}
    if not llm_raw_outputs:
        return consistency_scores # No runs to compare

    num_runs = len(llm_raw_outputs)

    # Flatten the data for easier comparison
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and all(isinstance(item, dict) for item in v):
                for i, item_val in enumerate(v):
                    items.extend(flatten_dict(item_val, f"{new_key}.{i}", sep=sep).items())
            else:
                items.append((new_key, str(v))) # Convert all values to string for comparison
        return dict(items)

    flattened_main_data = flatten_dict(extracted_data)
    flattened_all_outputs = [flatten_dict(output) for output in llm_raw_outputs]

    for field_name, primary_value in flattened_main_data.items():
        match_count = 0
        for output in flattened_all_outputs:
            if output.get(field_name) == primary_value:
                match_count += 1
        consistency_scores[field_name] = match_count / num_runs if num_runs > 0 else 0.0
    
    return consistency_scores

def _calculate_field_composite_confidence(
    field_name: str,
    field_value: str,
    llm_consistency_score: float,
    qa_results: Dict[str, Any],
    word_boxes: List[Dict[str, Any]]
) -> float:
    """
    Calculates a composite confidence score for a single field.
    Weights: LLM Consistency (70%), Validation (20%), OCR Quality (10%)
    """
    # 1. LLM Consistency (70%)
    # This is passed directly as llm_consistency_score

    # 2. Validation Score (20%)
    validation_score = 1.0 # Assume valid unless a specific failed rule applies
    # This part needs to be improved if specific validation rules are tied to field names.
    # For now, we'll give a full score if no general validation failures.
    # In a more advanced setup, you'd check qa_results for rules specific to `field_name`.
    # For simplicity, if ANY relevant validation rule failed, it penalizes.
    
    # A simple check: if the field name implies a date/number/email etc., check if its basic format rule failed
    if "date" in field_name and "format_invalid" in str(qa_results.get('failed_rules', [])): # very basic
        validation_score = 0.5 # Partial penalty for format issues
    if "numeric_invalid" in str(qa_results.get('failed_rules', [])) and ("amount" in field_name or "total" in field_name or "subtotal" in field_name or "charge" in field_name):
         validation_score = 0.5 # Partial penalty for format issues

    # If any specific rule tied to the field's content failed, we could apply a penalty.
    # e.g., if f"{field_name}_format_invalid" in qa_results['failed_rules']

    # 3. OCR Quality (10%)
    # This is highly dependent on what your OCR tool provides.
    # PyTesseract's image_to_data provides 'conf' for word confidence.
    ocr_confidence_avg = _get_ocr_confidence_for_value(field_value, word_boxes) # Returns 0-1.0
    
    # Apply weights
    confidence = (0.70 * llm_consistency_score) + \
                 (0.20 * validation_score) + \
                 (0.10 * ocr_confidence_avg)
    
    return confidence

def _get_ocr_confidence_for_value(value: str, word_boxes: List[Dict[str, Any]]) -> float:
    """
    Attempts to find the OCR confidence for the words making up the extracted value.
    This requires 'conf' (confidence) to be available in word_boxes, which
    pytesseract.image_to_data provides. PyMuPDF does not provide confidence per word by default.
    So, this will mostly apply to image-based OCR.
    """
    if not value or not word_boxes:
        return 0.0

    matching_word_confs = []
    normalized_value = value.strip().lower()

    for box_info in word_boxes:
        normalized_word_in_box = box_info.get('word', '').strip().lower()
        # Check if the extracted value is present in the OCR'd words or vice-versa
        # This is a simple substring match; more robust methods involve token alignment
        if normalized_value in normalized_word_in_box or normalized_word_in_box in normalized_value:
            # Assume 'conf' is available if OCR provided it. Otherwise, assume 1.0 (perfect)
            # If the OCR tool (like PyMuPDF) doesn't provide word confidence, this will default.
            confidence = box_info.get('conf', 100) / 100.0 # Tesseract conf is 0-100
            matching_word_confs.append(confidence)
            
    if matching_word_confs:
        return np.mean(matching_word_confs)
    return 0.0 # No matching OCR words found

def _find_bbox_for_value(value: str, word_boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Helper function to find the bounding box and page number for an extracted value.
    This is a simplistic match; a more robust solution would align tokens.
    """
    if not value or not word_boxes:
        return None
    
    normalized_value_tokens = re.findall(r'\b\w+\b', value.lower()) # Simple tokenization

    # Find words that are part of the value and aggregate their bboxes
    matching_boxes = []
    matching_page = None
    
    for box_info in word_boxes:
        box_word_norm = box_info.get('word', '').lower()
        if box_word_norm in normalized_value_tokens: # Check if a word in box matches a token
            matching_boxes.append(box_info['bbox'])
            if matching_page is None: # Capture page from first match
                matching_page = box_info['page_num']

    if matching_boxes:
        # Aggregate bounding boxes: find min x0, min y0, max x1, max y1
        x0 = min(b[0] for b in matching_boxes)
        y0 = min(b[1] for b in matching_boxes)
        x1 = max(b[2] for b in matching_boxes)
        y1 = max(b[3] for b in matching_boxes)
        return {"page": matching_page, "bbox": [x0, y0, x1, y1]}
    return None


def _calculate_overall_confidence(
    fields_with_confidence: List[Dict[str, Any]],
    qa_results: Dict[str, Any]
) -> float:
    """
    Calculates the overall document confidence based on field confidences and validation results.
    """
    if not fields_with_confidence:
        return 0.0

    # Define weights for critical fields (adjust as needed)
    field_weights = {
        "invoice_number": 1.5, "total_amount": 2.0, "vendor_name": 1.5, # Invoice
        "patient_name": 1.5, "total_charges": 2.0, "amount_due": 1.5, # Medical Bill
        "prescription_date": 1.5, "doctor_name": 1.5, "medications": 2.0 # Prescription
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for field in fields_with_confidence:
        field_name = field["name"]
        field_confidence = field["confidence"]
        
        # Use a higher weight for critical fields, default to 1.0 for others
        weight = field_weights.get(field_name.split('.')[0], 1.0) # Check top-level key for weight
        
        weighted_sum += field_confidence * weight
        total_weight += weight

    overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Apply penalty for failed validation rules
    num_failed_rules = len(qa_results.get("failed_rules", []))
    if num_failed_rules > 0:
        # Simple linear penalty: 10% penalty per failed rule, up to max 50%
        penalty = min(0.1 * num_failed_rules, 0.5)
        overall_score *= (1 - penalty)

    return round(overall_score, 2)
