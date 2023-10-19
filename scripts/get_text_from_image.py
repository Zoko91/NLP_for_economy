import easyocr
import os
from PIL import Image


def create_txt_from_image(image_path):
    # Load the image
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, paragraph=True)

    # OCR post-processing
    grouped_text = group_text_regions(result)

    # Sort the grouped text based on vertical position
    grouped_text.sort(key=lambda x: x[0][0][1])  # Sort based on the y-coordinate of the first bounding box point

    # Get the base name of the image file
    image_basename = os.path.basename(image_path)

    # Get the directory path of the image file
    image_directory = os.path.dirname(image_path)

    # Specify the output directory for the text file
    output_directory = os.path.join(image_directory, "transcriptions")

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Create a text file name by replacing the image extension with .txt
    text_file_name = os.path.splitext(image_basename)[0] + '.txt'

    # Create the full path of the text file
    text_file_path = os.path.join(output_directory, text_file_name)

    # Write the result to the text file
    with open(text_file_path, 'w', encoding='utf-8') as file:
        for group in grouped_text:
            header_line = group[0][1]
            paragraph_lines = [line[1] for line in group[1:]]
            paragraph_text = header_line + '\n' + '\n'.join(paragraph_lines) + '\n\n'
            file.write(paragraph_text)
            print(paragraph_text)

    '''# Windows only, might need to be changed for other OS

    # Open the image file
    image = Image.open(image_path)
    image.show()

    # Open the text file
    os.startfile(text_file_path)'''


def group_text_regions(result, threshold_distance=20):
    grouped_text = []
    current_group = []

    for i in range(len(result)):
        if i == 0:
            current_group.append(result[i])
        else:
            _, y1, _, _ = result[i][0]
            _, y2, _, _ = result[i - 1][0]

            if isinstance(y1, list) and isinstance(y2, list):
                y1 = y1[1]
                y2 = y2[1]

            distance = abs(y1 - y2)

            if distance <= threshold_distance:
                current_group.append(result[i])
            else:
                grouped_text.append(current_group)
                current_group = [result[i]]

    grouped_text.append(current_group)

    return grouped_text


# Example usage
if __name__ == '__main__':
    image_path = "AI.png"
    create_txt_from_image(image_path)
