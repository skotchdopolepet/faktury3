import os
import re

# 1) Known OCR Mistakes -> Corrections
#    We'll fix repeated text errors. Add more as needed.
REPLACEMENTS = {
    r"sates taxi": "sales tax",
    r"sates tax": "sales tax",
    r"sates tach": "sales tax",
    r"seriat no": "serial no",
    r"sertat no": "serial no",
    r"seat no": "serial no",
    r"erat no": "serial no",
    r"serat no": "serial no",
    r"net tax inclusive valuer?": "net tax inclusive value",
    r"net tax tnclusive value": "net tax inclusive value",
    r"sates tans": "sales tax",
    r"sre?es tax invoice": "sales tax invoice",  # e.g. "SREES" -> "sales"
}

# 2) Regex Patterns for the Four Fields
PAT_SERIAL_NO = re.compile(r"serial\s+no\.?\s*[:\-]?\s*(\d+)", re.IGNORECASE)
PAT_DATE      = re.compile(r"(\d{1,2}[-\.]\d{1,2}[-\.]\d{4})", re.IGNORECASE)
PAT_SALES_TAX = re.compile(r"sales\s+tax[:\s]+\(?([\d\.]+)\)?", re.IGNORECASE)
PAT_NET_TAX   = re.compile(r"net\s+tax\s+inclusive\s+value[:\s]+\(?([\d\.]+)\)?", re.IGNORECASE)


def clean_ocr_text(raw_text: str) -> str:
    """
    Apply known replacements (case-insensitive) to unify frequent OCR mistakes.
    We keep the entire text so the neural network still sees everything.
    """
    cleaned = raw_text
    for wrong, correct in REPLACEMENTS.items():
        cleaned = re.sub(wrong, correct, cleaned, flags=re.IGNORECASE)
    return cleaned

def extract_fields(cleaned_text: str):
    """
    Extract the four fields (Serial No, Date, Sales Tax, Net Tax).
    Returns a dict:
      {
        'serial_no': <str or None>,
        'date': <str or None>,
        'sales_tax': <str or None>,
        'net_tax_inclusive': <str or None>
      }
    """
    fields = {
        "serial_no": None,
        "date": None,
        "sales_tax": None,
        "net_tax_inclusive": None
    }

    m_serial = PAT_SERIAL_NO.search(cleaned_text)
    if m_serial:
        fields["serial_no"] = m_serial.group(1)

    m_date = PAT_DATE.search(cleaned_text)
    if m_date:
        fields["date"] = m_date.group(1)

    m_sales = PAT_SALES_TAX.search(cleaned_text)
    if m_sales:
        fields["sales_tax"] = m_sales.group(1)

    m_net = PAT_NET_TAX.search(cleaned_text)
    if m_net:
        fields["net_tax_inclusive"] = m_net.group(1)

    return fields


def main():
    ocr_folder = "ocr_output"
    output_folder = "ocr_output_cleaned"
    os.makedirs(output_folder, exist_ok=True)

    # Suppose you have invoice_1_output.txt .. invoice_399_output.txt
    for i in range(1, 400):
        filename = f"invoice_{i}_output.txt"
        input_path = os.path.join(ocr_folder, filename)
        if not os.path.isfile(input_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # 1) Clean the text
        cleaned_text = clean_ocr_text(raw_text)

        # 2) Extract the four fields
        fields = extract_fields(cleaned_text)

        # 3) Append them to the end of the text or anywhere you like
        #    so your neural network can still see them. We'll create a short summary block.
        #summary_block = "\n\n--- Extracted Fields ---\n"
        summary_block = f"Serial No: {fields['serial_no']}\n"
        summary_block += f"Date: {fields['date']}\n"
        summary_block += f"Sales Tax: {fields['sales_tax']}\n"
        summary_block += f"Net Tax Inclusive Value: {fields['net_tax_inclusive']}\n"

        final_text = summary_block

        # 4) Write out to new .txt in a cleaned folder
        output_path = os.path.join(output_folder, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"Processed {filename} -> {output_path}")

if __name__ == "__main__":
    main()
