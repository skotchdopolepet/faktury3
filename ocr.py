import os
from PIL import Image
import pytesseract

# If Tesseract is not on your PATH, uncomment and set the path to the tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Example path on Linux/Mac
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Example path on Windows

# Path to the folder containing images
image_folder = r"images"

# Path to the folder where OCR results will be saved
output_folder = r"ocr_output"




# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop over images "invoice_1.jpg" to "invoice_399.jpg"
for i in range(1, 400):
    image_name = f"invoice_{i}.jpg"
    image_path = os.path.join(image_folder, image_name)

    # Define the output file path
    output_file = os.path.join(output_folder, f"invoice_{i}_output.txt")

    # Check if the image file exists before processing
    if os.path.isfile(image_path):
        try:
            # Load the image
            with Image.open(image_path) as img:
                # Apply Tesseract OCR
                text = pytesseract.image_to_string(img)

            # Save the OCR result to a text file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            # Optional: Print a success message
            print(f"✅ Saved OCR result for {image_name} to {output_file}")

        except Exception as e:
            # Handle exceptions (e.g., unreadable image)
            print(f"❌ Error processing {image_name}: {e}")
    else:
        print(f"⚠️ File not found: {image_path}")
