from scripts.get_text_from_pdf import extract_text_from_pdf
from scripts.nlp_preprocessing import clean_text
# Constants
pdf_file_path = './Knowledge Graph/'
pdf_file_name = 'extracting_wisdom.pdf'

if __name__ == "__main__":
    # Extracts textual information from PDF file
    extracted_text = extract_text_from_pdf(pdf_file_path + pdf_file_name)
    # Cleans the extracted text
    cleaned_text = clean_text(extracted_text)
    print(cleaned_text)
