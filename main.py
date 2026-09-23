import cv2
import pytesseract
from sympy import symbols, Eq, solve
import re

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image
img = cv2.imread("test.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OCR
text = pytesseract.image_to_string(gray)
print("Raw OCR:", text)

# 🔥 CLEANING STEP
text = text.strip()
text = text.replace("X", "x")
text = text.replace(" ", "")
text = text.replace("=/", "=")
text = text.replace("//", "/")

# Fix repeated x
text = re.sub(r'x+', 'x', text)

# Add * (2x → 2*x)
text = re.sub(r'(\d)(x)', r'\1*\2', text)

# Remove unwanted characters
text = re.sub(r'[^0-9x=+\-*/]', '', text)

print("Cleaned Text:", text)

# 🔥 SOLVE + EXPLAIN
try:
    if "=" in text:
        left, right = text.split("=")

        x = symbols('x')

        # Step 1
        print("\n--- Step-by-step solution ---")

        print("Original Equation:", left, "=", right)

        # Move constant to right side
        # Assume form ax + b = c
        match = re.match(r'(\d*)\*?x([+-]\d+)?', left)

        if match:
            a = int(match.group(1)) if match.group(1) else 1
            b = int(match.group(2)) if match.group(2) else 0
            c = int(right)

            print(f"Step 1: Move constant → {a}x = {c - b}")

            # Step 2
            print(f"Step 2: Divide by {a} → x = {(c - b)/a}")

        # Final solution using SymPy
        equation = Eq(eval(left), eval(right))
        solution = solve(equation, x)

        print("\nFinal Answer:", solution)

    else:
        print("Not an equation")

except Exception as e:
    print("Error:", e)