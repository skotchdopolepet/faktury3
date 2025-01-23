import os
import re
import pandas as pd

# 1) Known OCR Mistakes -> Corrections
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
    r"sre?es tax invoice": "sales tax invoice",
}

# 2) Regex Patterns for the Four Fields
PAT_SERIAL_NO = re.compile(r"serial\s+no\.?\s*[:\-]?\s*(\d+)", re.IGNORECASE)
PAT_DATE      = re.compile(r"(\d{1,2}[-\.]\d{1,2}[-\.]\d{4})", re.IGNORECASE)
PAT_SALES_TAX = re.compile(r"sales\s+tax[:\s]+\(?([\d\.]+)\)?", re.IGNORECASE)
PAT_NET_TAX   = re.compile(r"net\s+tax\s+inclusive\s+value[:\s]+\(?([\d\.]+)\)?", re.IGNORECASE)

def clean_ocr_text(raw_text: str) -> str:
    """
    Apply known replacements (case-insensitive) to unify frequent OCR mistakes.
    """
    cleaned = raw_text
    for wrong, correct in REPLACEMENTS.items():
        cleaned = re.sub(wrong, correct, cleaned, flags=re.IGNORECASE)
    return cleaned

def extract_fields(cleaned_text: str):
    """
    Extract the four fields (Serial No, Date, Sales Tax, Net Tax).
    Returns a dict with possible None values if not found.
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

def parse_date(date_str):
    """
    Parse a date string like "12-01-2023" or "12.01.2023" into (day, month, year).
    Returns (None, None, None) if date_str is None or invalid format.
    """
    if not date_str:
        return None, None, None

    parts = re.split(r"[-\.]", date_str)
    if len(parts) != 3:
        return None, None, None
    day, month, year = parts
    return day, month, year

def parse_ground_truth(gt_text):
    """
    Given the raw text from a label file, parse lines like:
      Serial No.: 2568788
      date: 18-12-2020
      Sales Tax: 5490.0
      Net Tax Inclusive Value: 35990.0
    Returns a dict with keys: serial_no, date, sales_tax, net_tax.
    """
    lines = gt_text.strip().splitlines()
    result = {
        "serial_no": None,
        "date": None,
        "sales_tax": None,
        "net_tax": None
    }
    for line in lines:
        # e.g. "Serial No.: 2568788"
        parts = line.split(":", 1)
        if len(parts) < 2:
            continue
        key = parts[0].strip().lower()
        value = parts[1].strip()

        if "serial no" in key:
            result["serial_no"] = value
        elif "date" in key:
            result["date"] = value
        elif "sales tax" in key:
            result["sales_tax"] = value
        elif "net tax inclusive value" in key:
            # The line is often "Net Tax Inclusive Value: X"
            # We'll store that in "net_tax" for consistency
            result["net_tax"] = value
    return result

def main():
    ocr_folder = "ocr_output"
    label_folder = "txt_labels"
    output_folder = "ocr_output_cleaned"
    os.makedirs(output_folder, exist_ok=True)

    final_rows = []

    # Suppose we have invoice_1_output.txt .. invoice_399_output.txt
    # and label_1.txt .. label_399.txt
    for i in range(1, 400):
        invoice_filename = f"invoice_{i}_output.txt"
        label_filename   = f"label_{i}.txt"

        input_path  = os.path.join(ocr_folder, invoice_filename)
        label_path  = os.path.join(label_folder, label_filename)

        # If either invoice text or label file doesn't exist, skip
        if not os.path.isfile(input_path) or not os.path.isfile(label_path):
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        with open(label_path, "r", encoding="utf-8") as f:
            gt_text = f.read()

        # 1) Clean the text
        cleaned_text = clean_ocr_text(raw_text)

        # 2) Extract the four fields from OCR text
        fields = extract_fields(cleaned_text)

        # 3) Optional: Create a short summary block and write out to a cleaned folder
        summary_block = (
            f"Serial No: {fields['serial_no']}\n"
            f"Date: {fields['date']}\n"
            f"Sales Tax: {fields['sales_tax']}\n"
            f"Net Tax Inclusive Value: {fields['net_tax_inclusive']}\n"
        )
        output_path = os.path.join(output_folder, invoice_filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_block)

        # 4) Parse ground truth from the label file
        ground = parse_ground_truth(gt_text)

        # 5) Convert extracted OCR date into day/month/year
        day_pred, month_pred, year_pred = parse_date(fields["date"])

        # 6) Convert the ground truth date into day/month/year
        day_true, month_true, year_true = parse_date(ground["date"])

        # 7) Convert sales tax & net tax to float and round
        def as_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0.0

        st_pred_val = as_float(fields["sales_tax"])
        nt_pred_val = as_float(fields["net_tax_inclusive"])
        st_rounded = int(round(st_pred_val))
        nt_rounded = int(round(nt_pred_val))

        st_true_val = as_float(ground["sales_tax"])
        nt_true_val = as_float(ground["net_tax"])
        t_st_rounded = int(round(st_true_val))
        t_nt_rounded = int(round(nt_true_val))

        # 8) Construct the row
        row = {
            'pred_serial_no': str(fields["serial_no"] or ""),
            'true_serial_no': str(ground["serial_no"] or ""),
            'pred_day': day_pred or "",
            'true_day': day_true or "",
            'pred_month': month_pred or "",
            'true_month': month_true or "",
            'pred_year': year_pred or "",
            'true_year': year_true or "",
            'pred_sales_tax': f"{st_rounded}.0",
            'true_sales_tax': f"{t_st_rounded}.0",
            'pred_net_tax': f"{nt_rounded}.0",
            'true_net_tax': f"{t_nt_rounded}.0"
        }

        final_rows.append(row)
        print(f"Processed invoice_{i}_output.txt with label_{i}.txt")

    # 9) Turn all rows into a dataframe and save
    df_test = pd.DataFrame(final_rows)
    print("\nTest Predictions (sample):")
    print(df_test.head(5))

    df_test.to_csv("test_predictions_regex.csv", index=False)
    print("Saved: test_predictions_regex.csv")

if __name__ == "__main__":
    main()
