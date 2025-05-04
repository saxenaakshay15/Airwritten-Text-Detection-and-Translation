import cv2
import pytesseract
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional: Set this path only if you're on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load airwritten image
image_path = r"C:\Users\User\OneDrive\Desktop\his.png"
image = cv2.imread(image_path)

# Step 1: Preprocess the image for better OCR accuracy
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Invert if the text is white on black background
inverted = cv2.bitwise_not(gray)
# Threshold to make text clearer
thresh = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Step 2: Extract text using Tesseract OCR
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
text = pytesseract.image_to_string(thresh, config=custom_config).strip()
print(f"🧾 Detected Text (Latin): {text}")

# Step 3: Convert to other Indian scripts (not translation)
def convert_to_script(text, target_script):
    return transliterate(text, sanscript.ITRANS, target_script)

print(f"🔤 Devanagari (Hindi): {convert_to_script(text, sanscript.DEVANAGARI)}")
print(f"🔤 Bengali: {convert_to_script(text, sanscript.BENGALI)}")
print(f"🔤 Tamil: {convert_to_script(text, sanscript.TAMIL)}")
print(f"🔤 Gujarati: {convert_to_script(text, sanscript.GUJARATI)}")
