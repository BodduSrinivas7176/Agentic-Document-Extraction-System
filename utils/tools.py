import fitz # PyMuPDF
from PIL import Image
import pytesseract
import io
import os
import re

# Ensure tesseract is in your PATH or specify its path here
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
# Uncomment the line above and provide the correct path if you face TesseractNotFound errors.

def process_document(file_path: str) -> dict:
    """
    Processes a document (PDF or image) to extract text and bounding box information.

    Args:
        file_path: The path to the document file.

    Returns:
        A dictionary containing:
            'text': The extracted plain text content.
            'word_boxes': A list of dictionaries, each with 'word', 'bbox', and 'page_num'.
                          Bounding boxes are [x0, y0, x1, y1] for text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found at: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return _process_pdf(file_path)
    elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        return _process_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only PDF and common image formats are supported.")

def _process_pdf(pdf_path: str) -> dict:
    """
    Extracts text and word-level bounding boxes from a PDF document using PyMuPDF.
    """
    full_text = []
    word_boxes = []

    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            # Extract text
            page_text = page.get_text()
            full_text.append(page_text)

            # Extract word-level bounding boxes
            # format of word: (x0, y0, x1, y1, word, block_no, line_no, word_no)
            words_on_page = page.get_text("words")
            for word_info in words_on_page:
                word = word_info[4]
                bbox = [word_info[0], word_info[1], word_info[2], word_info[3]] # [x0, y0, x1, y1]
                if word.strip(): # Ensure the word is not just whitespace
                    word_boxes.append({
                        "word": word,
                        "bbox": bbox,
                        "page_num": page_num + 1 # 1-indexed page number
                    })
        doc.close()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return {"text": "", "word_boxes": []} # Return empty if error

    return {
        "text": "\n".join(full_text),
        "word_boxes": word_boxes
    }

def _process_image(image_path: str) -> dict:
    """
    Extracts text and word-level bounding boxes from an image using Tesseract OCR.
    """
    full_text = ""
    word_boxes = []

    try:
        img = Image.open(image_path)
        
        # Perform OCR and get detailed data including bounding boxes
        # output_type=pytesseract.Output.DICT gives a dictionary of data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Extract plain text
        full_text = pytesseract.image_to_string(img)

        n_boxes = len(data['text'])
        for i in range(n_boxes):
            # Tesseract provides bounding box in (left, top, width, height)
            # Convert to (x0, y0, x1, y1)
            if int(data['conf'][i]) > 0: # Only include words with confidence > 0
                word = data['text'][i]
                if word.strip(): # Ensure word is not just whitespace
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    bbox = [x, y, x + w, y + h]
                    word_boxes.append({
                        "word": word,
                        "bbox": bbox,
                        "page_num": 1 # Images are single-page
                    })
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not in your PATH. Please install it or specify the path in tools.py.")
        print("For Windows: https://tesseract-ocr.github.io/tessdoc/Installation.html#windows")
        print("For macOS: brew install tesseract")
        return {"text": "", "word_boxes": []}
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return {"text": "", "word_boxes": []}

    return {
        "text": full_text,
        "word_boxes": word_boxes
    }

