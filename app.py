import streamlit as st
import os
import json
from agent.core import DocumentExtractionAgent
from models.schemas import InvoiceSchema, MedicalBillSchema, PrescriptionSchema

# Ensure the 'data' directory exists for uploads
UPLOAD_DIR = "uploaded_temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="Agentic Document Extractor",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("📄 Agentic Document Extraction Challenge")
st.markdown("Upload a document (Invoice, Medical Bill, or Prescription) to extract key information with confidence scores.")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload a PDF or Image (PNG, JPG, JPEG)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=False
)

optional_fields_input = st.text_area(
    "Optional: Specify fields to extract (comma-separated, e.g., PatientName, DoctorName)",
    help="This feature is a 'nice-to-have' for the challenge and is not fully utilized in this demo's current implementation, but can be expanded."
)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File '{uploaded_file.name}' uploaded successfully.")

    if st.button("Extract Data"):
        with st.spinner("Processing document... This may take a moment as the LLM is running."):
            agent = DocumentExtractionAgent()
            # Pass optional fields if provided, otherwise None
            optional_fields_list = [field.strip() for field in optional_fields_input.split(',')] if optional_fields_input else None
            
            # Run the extraction agent
            result = agent.run_extraction(file_path, optional_fields_list)

            st.subheader("Extraction Results:")
            
            if result.get("doc_type") == "error":
                st.error(f"Extraction Failed: {result['qa']['notes']}")
                st.json(result) # Show full error details
            else:
                st.write(f"**Document Type Detected:** `{result.get('doc_type', 'N/A')}`")
                st.write(f"**Overall Confidence:** {result.get('overall_confidence', 0.0) * 100:.2f}%")
                st.progress(result.get('overall_confidence', 0.0), text=f"Overall Confidence: {result.get('overall_confidence', 0.0) * 100:.2f}%")

                st.subheader("Extracted Fields:")
                fields = result.get("fields", [])
                if fields:
                    for field in fields:
                        st.markdown(f"**{field['name']}**: `{field['value']}`")
                        st.progress(field['confidence'], text=f"Confidence: {field['confidence'] * 100:.2f}%")
                        if field['source'] and field['source'].get('bbox'):
                            bbox = [round(c, 2) for c in field['source']['bbox']]
                            st.markdown(f"<small>Page: {field['source'].get('page', 'N/A')}, BBox: {bbox}</small>", unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download JSON Output",
                        data=json.dumps(result, indent=2),
                        file_name=f"{uploaded_file.name}_extracted.json",
                        mime="application/json"
                    )
                else:
                    st.warning("No fields extracted.")

                st.subheader("Quality Assurance (QA) Report:")
                with st.expander("View QA Details"):
                    qa = result.get("qa", {})
                    st.write("**Passed Rules:**")
                    if qa.get("passed_rules"):
                        for rule in qa["passed_rules"]:
                            st.success(f"- {rule}")
                    else:
                        st.info("No rules passed or no specific rules defined for this document type.")

                    st.write("**Failed Rules:**")
                    if qa.get("failed_rules"):
                        for rule in qa["failed_rules"]:
                            st.error(f"- {rule}")
                    else:
                        st.info("No rules failed.")
                    
                    st.write(f"**Notes:** {qa.get('notes', 'N/A')}")
