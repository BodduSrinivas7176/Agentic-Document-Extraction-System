import re
from typing import Dict, Any, List, Callable, Tuple
from datetime import datetime

# Define validation functions for different types of checks

def _is_valid_date(date_string: str, date_format: str = "%Y-%m-%d") -> bool:
    """Checks if a string is a valid date according to a given format."""
    try:
        datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False

def _is_valid_decimal(value: Any) -> bool:
    """Checks if a value can be converted to a decimal (float)."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def _is_valid_email(email: str) -> bool:
    """Checks if a string is a valid email address."""
    # Simple regex for email validation
    return re.match(r"[^@]+@[^@]+\.[^@]+", email) is not None

def _is_valid_phone(phone_number: str) -> bool:
    """Checks if a string is a valid phone number (basic check)."""
    # Allows for digits, spaces, hyphens, and parentheses. Adjust as needed.
    return re.match(r"^\+?[\d\s\-\(\)]+$", phone_number) is not None

# --- Document-specific Validation Rules ---

def validate_invoice(extracted_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Applies validation rules specific to Invoice documents.

    Args:
        extracted_data: The dictionary of data extracted by the LLM.

    Returns:
        A tuple containing two lists: passed_rules and failed_rules.
    """
    passed_rules = []
    failed_rules = []

    # Rule 1: Validate date formats
    if 'invoice_date' in extracted_data and _is_valid_date(extracted_data['invoice_date']):
        passed_rules.append("invoice_date_format_valid")
    else:
        failed_rules.append("invoice_date_format_invalid")

    if 'due_date' in extracted_data and extracted_data['due_date'] and _is_valid_date(extracted_data['due_date']):
        passed_rules.append("due_date_format_valid")
    else:
        # Only fail if due_date was present but invalid. If null, it's not a failure.
        if 'due_date' in extracted_data and extracted_data['due_date'] is not None:
             failed_rules.append("due_date_format_invalid")


    # Rule 2: Validate numerical amounts
    if 'total_amount' in extracted_data and _is_valid_decimal(extracted_data['total_amount']):
        passed_rules.append("total_amount_numeric_valid")
    else:
        failed_rules.append("total_amount_numeric_invalid")

    if 'subtotal' in extracted_data and (extracted_data['subtotal'] is None or _is_valid_decimal(extracted_data['subtotal'])):
        passed_rules.append("subtotal_numeric_valid")
    else:
        if 'subtotal' in extracted_data and extracted_data['subtotal'] is not None:
             failed_rules.append("subtotal_numeric_invalid")

    if 'tax_amount' in extracted_data and (extracted_data['tax_amount'] is None or _is_valid_decimal(extracted_data['tax_amount'])):
        passed_rules.append("tax_amount_numeric_valid")
    else:
        if 'tax_amount' in extracted_data and extracted_data['tax_amount'] is not None:
             failed_rules.append("tax_amount_numeric_invalid")

    # Rule 3: Cross-field validation (sum of line items equals total)
    # This requires both total_amount and line_items to be present and parsable
    if 'total_amount' in extracted_data and 'line_items' in extracted_data:
        try:
            expected_total = float(extracted_data['total_amount'])
            calculated_line_items_total = sum(
                float(item.get('line_total', 0)) for item in extracted_data['line_items'] if _is_valid_decimal(item.get('line_total'))
            )
            # Allow for small floating point discrepancies
            if abs(expected_total - calculated_line_items_total) < 0.01:
                passed_rules.append("line_items_sum_matches_total")
            else:
                failed_rules.append("line_items_sum_mismatch")
        except (ValueError, TypeError):
            failed_rules.append("line_items_sum_check_failed_parsing")
    else:
        failed_rules.append("line_items_or_total_missing_for_sum_check")

    return passed_rules, failed_rules

def validate_medical_bill(extracted_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Applies validation rules specific to Medical Bill documents.
    """
    passed_rules = []
    failed_rules = []

    # Rule 1: Validate date of service formats
    if 'date_of_service_start' in extracted_data and _is_valid_date(extracted_data['date_of_service_start']):
        passed_rules.append("service_start_date_format_valid")
    else:
        failed_rules.append("service_start_date_format_invalid")

    if 'total_charges' in extracted_data and _is_valid_decimal(extracted_data['total_charges']):
        passed_rules.append("total_charges_numeric_valid")
    else:
        failed_rules.append("total_charges_numeric_invalid")

    if 'amount_due' in extracted_data and _is_valid_decimal(extracted_data['amount_due']):
        passed_rules.append("amount_due_numeric_valid")
    else:
        failed_rules.append("amount_due_numeric_invalid")

    # Rule: Check if total charges roughly equals (insurance_paid + amount_due) if all present
    if all(k in extracted_data and _is_valid_decimal(extracted_data[k]) for k in ['total_charges', 'amount_due']) \
       and 'insurance_paid' in extracted_data and (extracted_data['insurance_paid'] is None or _is_valid_decimal(extracted_data['insurance_paid'])):
        try:
            total_charges = float(extracted_data['total_charges'])
            amount_due = float(extracted_data['amount_due'])
            insurance_paid = float(extracted_data['insurance_paid']) if extracted_data['insurance_paid'] is not None else 0.0

            if abs(total_charges - (amount_due + insurance_paid)) < 0.01:
                passed_rules.append("charges_balance_check_valid")
            else:
                failed_rules.append("charges_balance_check_mismatch")
        except (ValueError, TypeError):
            failed_rules.append("charges_balance_check_failed_parsing")
    else:
        # Only add a failed rule if the fields were expected but not suitable for check
        pass # This check is optional if fields are missing

    return passed_rules, failed_rules

def validate_prescription(extracted_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Applies validation rules specific to Prescription documents.
    """
    passed_rules = []
    failed_rules = []

    # Rule 1: Validate prescription date format
    if 'prescription_date' in extracted_data and _is_valid_date(extracted_data['prescription_date']):
        passed_rules.append("prescription_date_format_valid")
    else:
        failed_rules.append("prescription_date_format_invalid")

    # Rule 2: Check if at least one medication is listed
    if 'medications' in extracted_data and isinstance(extracted_data['medications'], list) and len(extracted_data['medications']) > 0:
        passed_rules.append("at_least_one_medication_listed")
    else:
        failed_rules.append("no_medication_listed")

    # Rule 3: Basic check for key fields in each medication detail
    if 'medications' in extracted_data and isinstance(extracted_data['medications'], list):
        for i, med in enumerate(extracted_data['medications']):
            if not med.get('drug_name'):
                failed_rules.append(f"medication_{i}_drug_name_missing")
            else:
                passed_rules.append(f"medication_{i}_drug_name_present")
            
            if not med.get('dosage'):
                failed_rules.append(f"medication_{i}_dosage_missing")
            else:
                passed_rules.append(f"medication_{i}_dosage_present")
            
            if not med.get('frequency'):
                failed_rules.append(f"medication_{i}_frequency_missing")
            else:
                passed_rules.append(f"medication_{i}_frequency_present")

    return passed_rules, failed_rules


# Main validation dispatcher
def run_validation(doc_type: str, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatches to the appropriate validation function based on document type.

    Args:
        doc_type: The classified document type ('invoice', 'medical_bill', 'prescription').
        extracted_data: The dictionary of data extracted by the LLM.

    Returns:
        A dictionary with 'passed_rules' and 'failed_rules'.
    """
    passed_rules: List[str] = []
    failed_rules: List[str] = []

    if doc_type == 'invoice':
        passed, failed = validate_invoice(extracted_data)
        passed_rules.extend(passed)
        failed_rules.extend(failed)
    elif doc_type == 'medical_bill':
        passed, failed = validate_medical_bill(extracted_data)
        passed_rules.extend(passed)
        failed_rules.extend(failed)
    elif doc_type == 'prescription':
        passed, failed = validate_prescription(extracted_data)
        passed_rules.extend(passed)
        failed_rules.extend(failed)
    else:
        failed_rules.append(f"no_validation_rules_for_type_{doc_type}")

    return {
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
        "notes": f"{len(failed_rules)} validation rule(s) failed." if failed_rules else "All primary validation rules passed."
    }

