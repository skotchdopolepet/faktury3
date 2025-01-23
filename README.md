# README

Welcome to this small suite of scripts designed for extracting text from invoice images using OCR (Tesseract), cleaning up that text, applying some regex-based field extraction, and then organizing the results. Below is a friendly overview of what each script does, how to use them, and a bit of extra context around the project.

---

## Overview

- **Goal**: Convert invoice images (`.jpg` files) into structured text with important fields such as Serial No., Date, Sales Tax, and Net Tax Inclusive Value.
- **Tools**:
  - [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) – for turning images into text.
  - Python’s `re` module – for cleaning text and pattern matching.
  - `pandas` – for collating results into CSVs.
  - The usual suspects: `os`, `PIL` (Pillow), etc.

Essentially, we have a few separate scripts that each tackle a piece of this workflow:

1. **ocr.py** → Runs Tesseract OCR on images and saves raw text.
2. **parse\_ocr.py** → Loads the raw text, corrects known OCR mistakes, extracts fields with regex, and writes a “cleaned” summary.
3. **regex.py** → A more advanced version of parsing that compares OCR text against ground-truth labels, organizes results, and saves them to a CSV.
4. **epoch\_losses\_scaled.txt** → A log that shows training/validation loss changes across epochs (not directly related to OCR, but presumably relevant for some training component).
5. **Test\_predictions\_regex.csv** & **Test\_predictions\_scaled.csv** → Example outputs from the regex scripts, showing predicted vs. true fields.

---

## Data Source

This project uses invoice image data sourced from [Roboflow Universe: Invoice Management Dataset](https://universe.roboflow.com/cvip-workspace/invoice-management). Be sure to check their platform for dataset details and licensing information.

---

## What’s Inside Each Script?

### 1) **ocr.py**

- Loops over image files named `invoice_1.jpg` up to `invoice_399.jpg` (or as many exist).
- Uses Tesseract to convert each image into text.
- Outputs a corresponding `.txt` file (e.g., `invoice_1_output.txt`) to the folder `ocr_output/`.
- **Key points**:
  - Make sure Tesseract is installed and either on your PATH or that `pytesseract.pytesseract.tesseract_cmd` is set correctly in `ocr.py`.
  - Adjust `image_folder` and `output_folder` to match where your images and desired output path live.

### 2) **parse\_ocr.py**

- Reads the `.txt` files produced by `ocr.py` from `ocr_output/`.
- Corrects common OCR errors (e.g., “sates tax” → “sales tax”).
- Extracts fields using regular expressions, such as:
  - **Serial No** – looking for patterns like `serial no: 12345`.
  - **Date** – matches `\d{1,2}[-\.]\d{1,2}[-\.]\d{4}`.
  - **Sales Tax** – tries to parse numeric values after “sales tax”.
  - **Net Tax Inclusive Value** – also extracts numeric data if present.
- Appends these fields into a new “summary block” at the end of the cleaned text.
- Writes out the final text to `ocr_output_cleaned/`.
- **Key points**:
  - Great for quickly turning raw OCR into something semi-structured.
  - You can add or remove common OCR mistakes in the `REPLACEMENTS` dictionary.

### 3) **regex.py**

- Similar approach to `parse_ocr.py`, but more advanced handling.
- Also tries to parse ground-truth label files (`label_1.txt` etc.) from a `txt_labels` folder.
- Compares extracted fields to the ground truth, calculates differences, and saves everything in a CSV called `test_predictions_regex.csv`.
- **Key points**:
  - Useful if you have a “label” text file that is known to be correct, so you can compare OCR results vs. truth.
  - Contains extra logic for date parsing and integer rounding of numeric fields like sales tax.

### 4) **epoch\_losses\_scaled.txt**

- A text file that seems to track the training and validation losses over multiple epochs of some model.
- **Key points**:
  - Might be unrelated to the main OCR pipeline, or part of a bigger project that trains a neural net on invoice data.
  - Not required for the OCR scripts to run, but it’s a nice record of how a model performed over time.

### 5) **Test\_predictions\_regex.csv** and **Test\_predictions\_scaled.csv**

- Example CSV outputs from a run of the regex scripts.
- Show how predicted fields compare to the ground truth (e.g., `pred_serial_no` vs. `true_serial_no`), with day/month/year breakdowns, sales tax values, etc.
- Good for debugging and verifying the pipeline.

---

## Getting Started

1. **Install Tesseract**

   - On Windows, you can grab the installer from [Tesseract’s official page](https://github.com/tesseract-ocr/tesseract/wiki).
   - On Linux (Ubuntu/Debian), run:
     ```bash
     sudo apt-get install tesseract-ocr
     ```
   - Adjust `pytesseract.pytesseract.tesseract_cmd` in `ocr.py` if Tesseract isn’t in your PATH.

2. **Install Python Dependencies**

   ```bash
   pip install Pillow pytesseract pandas
   ```

   (or use `conda install` as appropriate).

3. **Prepare Your Folders**

   - Create an `images/` folder with invoice images named `invoice_1.jpg`, `invoice_2.jpg`, etc.
   - Make sure you have a folder named `ocr_output/`. (The scripts automatically create it if it doesn’t exist, but best to confirm.)

4. **Run `ocr.py`**

   - This reads every `invoice_*.jpg` file in `images/`, applies Tesseract, and writes the result to `ocr_output/invoice_*_output.txt`.

5. **Run `parse_ocr.py` or `regex.py`**

   - `parse_ocr.py` is simpler: it cleans and extracts fields, then saves them to `ocr_output_cleaned/`.
   - `regex.py` is more advanced: it looks for matching label files in a `txt_labels/` folder, compares results, and outputs CSV metrics.

---

## Tips & Notes

- **Debugging OCR**: If Tesseract fails to read certain invoices, consider adjusting image clarity, resizing the image, or training Tesseract with custom language packs.
- **Expanding REPLACEMENTS**: If you see consistent OCR mistakes (e.g., “5ales tax” or “SateS tax”), add them to the `REPLACEMENTS` dictionary in `parse_ocr.py` or `regex.py`.
- **Regex Tuning**: If your invoice format changes or you want to capture new fields, update the regex patterns. For instance, if you also need “Vendor Name,” you’d craft a new pattern and add the corresponding logic in `extract_fields()`.
- **Performance**: Tesseract’s speed depends on your system resources and the size/number of images. Bulk processing hundreds of images can take a while; consider running on a server or with GPU acceleration (though standard Tesseract is CPU-based).
- **Data Verification**: The CSV files (like `Test_predictions_regex.csv`) are helpful for verifying your pipeline. Check if you’re consistently missing certain fields or if your data has consistent offset errors.

---

## Possible Next Steps

- **Neural Network Integration**: If you’re training a model to parse fields more accurately, you might use the text from `ocr_output_cleaned/` as input. Integrate these scripts as a data pre-processing step.
- **GUI or Web App**: Wrap these scripts in a simple web interface to let non-technical users upload invoices and see extracted info live.
- **Database Storage**: Instead of text files, push the cleaned data into a database or an API endpoint.
- **Error Handling & Logging**: For robust production usage, you DESPERATELY NEED more detailed logs or error codes when Tesseract or the regex can’t find fields.

---

## Contributing

Feel free to modify the scripts to suit your invoice layout or to add more robust error handling. Any improvements to the regex patterns or replacements for OCR mistakes are especially welcome.

---

That’s it! Hopefully, these scripts get you well on your way to automating invoice text extraction and data organization. If you get stuck or find new ways to optimize the process, have fun experimenting and tweaking. Enjoy your newly automated OCR pipeline!

