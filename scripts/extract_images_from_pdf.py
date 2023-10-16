import os
import fitz


def extract_images_from_pdf(pdf_path, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    with fitz.open(pdf_path) as doc:
        # Iterate over each page in the PDF
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Get the image list on the page
            image_list = page.get_images()

            # Save the images
            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)

                # Get the image data and properties
                image_data = base_image["image"]
                image_ext = base_image["ext"]
                image_colorspace = base_image["colorspace"]

                # Determine the image type
                if image_colorspace == "/DeviceRGB":
                    image_type = "RGB"
                else:
                    image_type = "P"

                # Create a filename for the image
                if image_type == "RGB":
                    image_filename = f"image_page{page_num + 1}_{image_index}.png"
                else:
                    image_filename = f"image_page{page_num + 1}_{image_index}.jpeg"

                # Save the image
                output_path = os.path.join(output_dir, image_filename)
                with open(output_path, "wb") as image_file:
                    image_file.write(image_data)


if __name__ == "__main__":
    # Provide the path to the PDF file

    pdf_file = "EuroChoices - 2022 - Miller - Creating Conditions for Harnessing the Potential of Transitions to Agroecology in Europe and.pdf"
    pdf_path = f"../Knowledge Graph/{pdf_file}"
    # Provide the output directory where the extracted images will be saved
    output_directory = f"images/{pdf_file}"

    # Extract images from the PDF file
    extract_images_from_pdf(pdf_path, output_directory)
