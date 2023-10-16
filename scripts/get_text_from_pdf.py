import fitz


def extract_text_from_pdf(pdf_file_path):
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        return str(e)


# Example usage:
pdf_file_path = '../Knowledge Graph/'
pdf_file_name = '14-Emmanuel-Petit-Christophe-Leveque.pdf'
extracted_text = extract_text_from_pdf(pdf_file_path+pdf_file_name)
print(extracted_text)
