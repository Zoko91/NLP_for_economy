import fitz


def extract_text_from_pdf(pdf_file_path, skip_pages=0, end_pages=0):
    try:
        doc = fitz.open(pdf_file_path)
        text = ""
        total_pages = doc.page_count
        for page_num in range(skip_pages, total_pages - end_pages):
            if page_num < skip_pages:
                continue
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    print("Hello World!")
    pdf_file_path = '../Knowledge Graph/'
    pdf_file_name = '14-Emmanuel-Petit-Christophe-Leveque.pdf'
    extracted_text = extract_text_from_pdf(pdf_file_path + pdf_file_name)
    print(extracted_text)
