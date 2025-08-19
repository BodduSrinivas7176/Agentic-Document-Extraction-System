from pydantic import BaseModel, Field, condecimal, conlist
from typing import Optional, List, Tuple, Literal # Import Literal

# Define a common source field for all extracted data
class Source(BaseModel):
    page: Optional[int] = Field(None, description="The page number where the field was found.")
    bbox: Optional[Tuple[float, float, float, float]] = Field(None, description="Bounding box coordinates [x1, y1, x2, y2] of the field.")

# Define a generic field structure for consistent output
class ExtractedField(BaseModel):
    name: str = Field(..., description="The name of the extracted field.")
    value: str = Field(..., description="The extracted value of the field.")
    confidence: Optional[float] = Field(None, description="Confidence score for this specific field (0.0-1.0).")
    source: Optional[Source] = Field(None, description="Source location details (page, bounding box).")

# --- Invoice Schema ---
class LineItem(BaseModel):
    description: str = Field(..., description="Description of the line item.")
    quantity: int = Field(..., description="Quantity of the item.")
    unit_price: condecimal(decimal_places=2) = Field(..., description="Unit price of the item.")
    line_total: condecimal(decimal_places=2) = Field(..., description="Total amount for this line item.")

class InvoiceSchema(BaseModel):
    # Changed const=True to Literal["invoice"]
    doc_type: Literal["invoice"] = Field("invoice", description="The type of the document.")
    vendor_name: str = Field(..., description="The name of the company that issued the invoice.")
    invoice_number: str = Field(..., description="The unique invoice identification number.")
    invoice_date: str = Field(..., description="The date the invoice was issued (e.g., YYYY-MM-DD).")
    due_date: Optional[str] = Field(None, description="The date the payment is due (e.g., YYYY-MM-DD).")
    total_amount: condecimal(decimal_places=2) = Field(..., description="The total amount due on the invoice.")
    subtotal: Optional[condecimal(decimal_places=2)] = Field(None, description="The subtotal before taxes and discounts.")
    tax_amount: Optional[condecimal(decimal_places=2)] = Field(None, description="The total tax amount on the invoice.")
    currency: str = Field(..., description="The currency of the amounts (e.g., USD, EUR).")
    line_items: List[LineItem] = Field(..., description="A list of detailed line items on the invoice.")


# --- Medical Bill Schema ---
class MedicalService(BaseModel):
    service_date: str = Field(..., description="Date when the medical service was rendered (e.g., YYYY-MM-DD).")
    description: str = Field(..., description="Description of the medical service or procedure.")
    amount: condecimal(decimal_places=2) = Field(..., description="Cost for this specific service.")
    cpt_code: Optional[str] = Field(None, description="CPT (Current Procedural Terminology) code for the service.")

class MedicalBillSchema(BaseModel):
    # Changed const=True to Literal["medical_bill"]
    doc_type: Literal["medical_bill"] = Field("medical_bill", description="The type of the document.")
    patient_name: str = Field(..., description="The full name of the patient.")
    patient_id: Optional[str] = Field(None, description="The patient's identification number.")
    date_of_service_start: str = Field(..., description="The start date of the service period (e.g., YYYY-MM-DD).")
    date_of_service_end: Optional[str] = Field(None, description="The end date of the service period (e.g., YYYY-MM-DD).")
    provider_name: str = Field(..., description="The name of the healthcare provider or facility.")
    total_charges: condecimal(decimal_places=2) = Field(..., description="The total amount charged for all services.")
    amount_due: condecimal(decimal_places=2) = Field(..., description="The amount the patient is responsible for paying.")
    insurance_paid: Optional[condecimal(decimal_places=2)] = Field(None, description="Amount paid by insurance.")
    services: List[MedicalService] = Field(..., description="A list of individual medical services rendered.")


# --- Prescription Schema ---
class MedicationDetail(BaseModel):
    drug_name: str = Field(..., description="Name of the prescribed medication.")
    strength: str = Field(..., description="Strength of the medication (e.g., 250mg, 50mg/5ml).")
    dosage: str = Field(..., description="Dosage instructions (e.g., 1 tablet, 5ml).")
    frequency: str = Field(..., description="How often the medication should be taken (e.g., daily, twice a day).")
    route: Optional[str] = Field(None, description="Route of administration (e.g., oral, topical).")
    dispense_quantity: Optional[str] = Field(None, description="Quantity to be dispensed (e.g., #30, 1 bottle).")
    refills: Optional[str] = Field(None, description="Number of refills allowed.")

class PrescriptionSchema(BaseModel):
    # Changed const=True to Literal["prescription"]
    doc_type: Literal["prescription"] = Field("prescription", description="The type of the document.")
    patient_name: str = Field(..., description="The full name of the patient.")
    patient_dob: Optional[str] = Field(None, description="Patient's date of birth (e.g., YYYY-MM-DD).")
    prescription_date: str = Field(..., description="The date the prescription was issued (e.g., YYYY-MM-DD).")
    doctor_name: str = Field(..., description="The name of the prescribing doctor.")
    doctor_license: Optional[str] = Field(None, description="Doctor's license or NPI number.")
    medications: List[MedicationDetail] = Field(..., description="A list of prescribed medications with their details.")

