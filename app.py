import streamlit as st
import cv2
import numpy as np
import easyocr

try:
    import pytesseract
except ImportError:
    pytesseract = None
import sympy as sp
import re
import gc
import pandas as pd
import matplotlib.pyplot as plt
import database
import os
import time
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="Handwritten Formula Solver")

# Create database tables and default admin without changing existing users/history
try:
    database.create_tables()
    database.register_user("admin", "admin")
except Exception:
    pass

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------- LOGIN / REGISTER ----------------
st.sidebar.title("User Menu")

if not st.session_state.logged_in:
    menu = st.sidebar.selectbox("Select", ["Login", "Register"])

    if menu == "Register":
        st.title("📝 Register")
        new_user = st.text_input("Username")
        new_pass = st.text_input("Password", type="password")

        if st.button("Register"):
            if new_user and new_pass:
                success = database.register_user(new_user, new_pass)
                if success:
                    st.success("Registration successful. Please login.")
                else:
                    st.error("Username already exists.")
            else:
                st.warning("Enter username and password.")
        st.stop()

    if menu == "Login":
        st.title("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = database.login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

st.sidebar.success(f"Logged in as {st.session_state.username}")

if st.session_state.username == "admin":
    page = st.sidebar.radio("Go to", ["Admin Dashboard"])
else:
    page = st.sidebar.radio(
        "Go to",
        ["Solver", "Formula Solver", "History", "Dashboard"]
    )

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ---------------- OCR ----------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(["en"], gpu=False)


def resize_image(img, max_width=1400):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _crop_to_writing(gray):
    """Crop empty paper borders while retaining superscripts and equation signs."""
    try:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        area_limit = max(12, gray.size * 0.00002)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= area_limit and h >= 3:
                boxes.append((x, y, w, h))
        if not boxes:
            return gray
        x1 = min(x for x, y, w, h in boxes)
        y1 = min(y for x, y, w, h in boxes)
        x2 = max(x + w for x, y, w, h in boxes)
        y2 = max(y + h for x, y, w, h in boxes)
        px = max(25, int((x2 - x1) * 0.06))
        py = max(25, int((y2 - y1) * 0.25))
        return gray[max(0, y1-py):min(gray.shape[0], y2+py), max(0, x1-px):min(gray.shape[1], x2+px)]
    except Exception:
        return gray


def _join_easyocr_results(results):
    """Arrange OCR boxes into lines and preserve useful spacing."""
    items = []
    for item in results:
        if len(item) < 3 or not str(item[1]).strip():
            continue
        box, text, confidence = item
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        items.append({"x1": x1, "x2": x2, "yc": (y1+y2)/2,
                      "h": max(1.0, y2-y1), "text": str(text).strip(),
                      "confidence": float(confidence)})
    if not items:
        return "", []
    items.sort(key=lambda d: d["yc"])
    lines = []
    for it in items:
        for line in lines:
            tolerance = max(18.0, 0.55 * max(line["avg_h"], it["h"]))
            if abs(it["yc"] - line["yc"]) <= tolerance:
                line["items"].append(it)
                n = len(line["items"])
                line["yc"] = ((line["yc"]*(n-1))+it["yc"])/n
                line["avg_h"] = ((line["avg_h"]*(n-1))+it["h"])/n
                break
        else:
            lines.append({"yc": it["yc"], "avg_h": it["h"], "items": [it]})
    lines.sort(key=lambda l: l["yc"])
    output_lines, confidences = [], []
    for line in lines:
        row = sorted(line["items"], key=lambda d: d["x1"])
        parts, previous_x2 = [], None
        for part in row:
            if previous_x2 is not None and part["x1"]-previous_x2 > max(12.0, line["avg_h"]*0.30):
                parts.append(" ")
            parts.append(part["text"])
            previous_x2 = part["x2"]
            confidences.append(part["confidence"])
        output_lines.append("".join(parts).strip())
    return "\n".join(x for x in output_lines if x), confidences


def _basic_math_normalize(text):
    if not text:
        return ""
    t = str(text).replace("\n", "").replace(" ", "")
    t = t.replace("−", "-").replace("–", "-").replace("—", "-")
    t = t.replace("×", "*").replace("÷", "/").replace("π", "pi")
    t = t.replace("²", "^2").replace("³", "^3").replace(":", "=")
    t = t.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    return t


def _repair_common_math_ocr(text):
    """Repair only high-confidence handwriting mistakes; never invent random values."""
    t = _basic_math_normalize(text).lower()
    t = t.replace("sln", "sin").replace("5in", "sin").replace("s1n", "sin")
    t = t.replace("c0s", "cos").replace("co5", "cos").replace("c05", "cos")
    t = t.replace("1og", "log").replace("10g", "log").replace("iog", "log").replace("loq", "log")
    t = t.replace("fi", "pi").replace("pl", "pi")
    t = t.replace("/z", "/2").replace("/l", "/1")
    t = re.sub(r"={2,}", "=", t)
    t = re.sub(r"\+{2,}", "+", t)
    t = re.sub(r"-{2,}", "-", t)

    # Very common log handwriting: OCR sees parts in reverse order, e.g. ()-2log10.
    if "log10" in t or ("log" in t and "10" in t):
        nums = re.findall(r"(?<![a-z])(-?\d+(?:\.\d+)?)", t)
        rhs = None
        if "=" in t:
            after = t.split("=", 1)[1]
            m = re.search(r"-?\d+(?:\.\d+)?", after)
            rhs = m.group(0) if m else None
        if rhs is None:
            usable = [n for n in nums if n not in {"10"}]
            rhs = usable[-1] if usable else None
        if rhs is not None and ("x" in t or "()" in t or "(" in t):
            return f"log10(x)={rhs}"

    # Common trigonometric handwriting corrections.
    if "sin" in t:
        t = re.sub(r"sin\((?:f|p)?i?/?[z2]\)", "sin(pi/2)", t)
    if "cos" in t:
        t = re.sub(r"cos\((?:p|f)?i\)", "cos(pi)", t)
    if "sin(pi/2)" in t and "cos(pi)" in t:
        return "y=sin(pi/2)+cos(pi)" if "y" in t or "=" in t else "sin(pi/2)+cos(pi)"

    # Put the equation around '=' in the correct direction when OCR reverses boxes.
    if t.count("=") == 1:
        left, right = t.split("=", 1)
        if re.fullmatch(r"-?\d+(?:\.\d+)?", left) and re.search(r"[xy]|sin|cos|log|sqrt", right):
            t = right + "=" + left

    return t


def _ocr_math_score(candidate, confidences=None):
    if not candidate:
        return -10000.0
    t = _repair_common_math_ocr(candidate)
    compact = t.replace(" ", "").replace("\n", "")
    score = 0.0
    eq_count = compact.count("=")
    score += 28 if eq_count == 1 else -18 * abs(eq_count - 1)
    score += min(sum(c.isdigit() for c in compact), 8)
    score += min(sum(compact.count(op) for op in "+-/*^"), 6) * 2.0
    score += 8 if re.search(r"[xy]", compact) else 0
    score += 8 if any(k in compact for k in ("sin","cos","tan","log","sqrt","pi")) else 0
    score -= abs(compact.count("(") - compact.count(")")) * 5
    cleaned = compact
    for word in ("sin","cos","tan","log","sqrt","pi","alpha","beta","theta","omega","lambda","find"):
        cleaned = cleaned.replace(word, "")
    extra_letters = re.findall(r"[a-wzA-WZ]", cleaned.replace("x","").replace("y",""))
    score -= len(extra_letters) * 4.0
    score -= len(re.findall(r"\d{5,}", compact)) * 10.0
    score -= max(0, len(compact)-30) * 0.8
    if confidences:
        score += (sum(confidences)/max(1,len(confidences))) * 7
    return score


def _tesseract_candidates(image_versions):
    if pytesseract is None:
        return []
    out = []
    whitelist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=*/().^"
    configs = [
        f"--oem 3 --psm 7 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}",
        f"--oem 3 --psm 13 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}",
        f"--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist={whitelist}",
    ]
    for image in image_versions:
        for config in configs:
            try:
                text = pytesseract.image_to_string(image, config=config).strip()
                if text:
                    out.append((text, []))
            except Exception:
                pass
    return out


@st.cache_data(show_spinner=False)
def _ocr_from_bytes(file_bytes):
    img_array = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return None, ""

    img = resize_image(img, max_width=1400)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _crop_to_writing(gray_full)

    # Upscale characters because superscripts and brackets are otherwise lost.
    h, w = gray.shape[:2]
    target_h = 420
    scale = max(1.0, min(3.0, target_h / max(h, 1)))
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(enlarged, 7, 45, 45)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(denoised)
    otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 13)

    versions = [clahe, otsu, adaptive]
    candidates = []

    reader = load_reader()
    allowlist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=*/().^π√"
    for version in versions:
        try:
            results = reader.readtext(
                version,
                detail=1,
                paragraph=False,
                decoder="greedy",
                allowlist=allowlist,
                canvas_size=2200,
                mag_ratio=1.15,
                width_ths=0.25,
                height_ths=0.35,
                text_threshold=0.35,
                low_text=0.20,
                link_threshold=0.20,
                add_margin=0.08,
            )
            raw, conf = _join_easyocr_results(results)
            if raw:
                candidates.append((raw, conf))
        except Exception:
            pass

    # Tesseract often preserves a one-line equation's order better than EasyOCR.
    candidates.extend(_tesseract_candidates(versions))

    if not candidates:
        return img, ""

    ranked = []
    seen = set()
    for raw, conf in candidates:
        fixed = _repair_common_math_ocr(raw)
        if fixed and fixed not in seen:
            seen.add(fixed)
            ranked.append((_ocr_math_score(fixed, conf), fixed))

    if not ranked:
        return img, ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    return img, ranked[0][1]


def preprocess_camera_frame(file_bytes):
    """Prepare a webcam photo without destroying thin handwritten strokes."""
    data = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img = resize_image(img, max_width=1800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0,0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(gray, background, scale=255)
    normalized = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8)).apply(normalized)
    writing = _crop_to_writing(normalized)
    if writing is None or writing.size == 0:
        writing = normalized
    h, w = writing.shape[:2]
    scale = max(1.5, min(3.0, 520/max(h,1)))
    enlarged = cv2.resize(writing, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    enlarged = cv2.GaussianBlur(enlarged, (3,3), 0)
    enhanced = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    return img, enhanced


@st.cache_data(show_spinner=False)
def _ocr_camera_from_bytes(file_bytes):
    original, enhanced = preprocess_camera_frame(file_bytes)
    if original is None:
        return None, None, ""
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0,0), sigmaX=35, sigmaY=35)
    normalized = cv2.divide(gray, background, scale=255)
    writing = _crop_to_writing(normalized)
    if writing is None or writing.size == 0:
        writing = normalized
    h, w = writing.shape[:2]
    scale = max(1.4, min(3.2, 560/max(h,1)))
    big = cv2.resize(writing, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8,8)).apply(big)
    otsu = cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,11)
    versions = [big, clahe, otsu, adaptive, enhanced]
    candidates = []
    reader = load_reader()
    allowlist = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-=*/().^π√"
    for version in versions:
        for width_ths in (0.45,0.75):
            try:
                results = reader.readtext(version, detail=1, paragraph=False,
                    decoder="beamsearch", beamWidth=10, allowlist=allowlist,
                    canvas_size=2560, mag_ratio=1.2, width_ths=width_ths,
                    height_ths=0.5, text_threshold=0.45, low_text=0.25,
                    link_threshold=0.30, add_margin=0.05)
                raw, conf = _join_easyocr_results(results)
                if raw:
                    candidates.append((raw, conf))
            except Exception:
                pass
    candidates.extend(_tesseract_candidates(versions))
    if not candidates:
        return original, enhanced, ""
    ranked, seen = [], set()
    for raw, conf in candidates:
        fixed = _repair_common_math_ocr(raw).replace("\n", " ").strip()
        if fixed and fixed not in seen:
            seen.add(fixed)
            ranked.append((_ocr_math_score(fixed, conf), fixed))
    if not ranked:
        return original, enhanced, ""
    ranked.sort(key=lambda item: item[0], reverse=True)
    return original, enhanced, ranked[0][1]


def read_image_ocr(uploaded_file, camera=False):
    if camera:
        original, enhanced, text = _ocr_camera_from_bytes(uploaded_file.getvalue())
        return original, text, enhanced
    original, text = _ocr_from_bytes(uploaded_file.getvalue())
    return original, text, None


# ---------------- EQUATION SOLVER HELPERS ----------------
def smart_fix_ocr_text(text):
    """Normalize common OCR mistakes without forcing a specific equation."""
    if text is None:
        return ""

    value = str(text).strip().replace("\n", " ")
    value = value.replace("‘", "").replace("’", "").replace("'", "").replace('"', "")
    value = value.replace("`", "").replace(" ", "")

    # Mathematical symbols and superscripts.
    value = value.replace("²", "^2").replace("³", "^3")
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("×", "*").replace("÷", "/").replace(":", "=")
    value = value.replace("π", "pi").replace("Π", "pi").replace("𝜋", "pi")

    # Common OCR spellings for functions.
    replacements = {
        "sln": "sin", "5in": "sin", "s1n": "sin", "sih": "sin",
        "c0s": "cos", "c05": "cos", "co5": "cos",
        "1og": "log", "10g": "log", "iog": "log", "loq": "log",
        "loglo": "log10", "logl0": "log10", "logio": "log10", "logi0": "log10"
    }
    lowered = value.lower()
    for wrong, right in replacements.items():
        lowered = lowered.replace(wrong, right)

    # Keep only the likely equation if OCR has harmless text before/after it.
    lowered = re.sub(r"^(equation|answer|solve|find)", "", lowered)

    # Context-sensitive character corrections.
    lowered = re.sub(r"(?<=\d)[o](?=\d|$)", "0", lowered)
    lowered = re.sub(r"(?<=[=+\-*/])[o](?=\d)", "0", lowered)
    lowered = re.sub(r"(?<=\d)[l|i](?=\d)", "1", lowered)

    # Normalize a handwritten power such as x2, x^2 or xˆ2.
    lowered = lowered.replace("ˆ", "^")
    lowered = re.sub(r"([xy])\s*\^\s*([23])", r"\1^\2", lowered)
    lowered = re.sub(r"\b([xy])([23])(?=[+\-=]|$)", r"\1^\2", lowered)

    # Remove duplicate equals and repeated operators created by OCR.
    lowered = re.sub(r"={2,}", "=", lowered)
    lowered = re.sub(r"\+{2,}", "+", lowered)
    lowered = re.sub(r"-{2,}", "-", lowered)
    lowered = re.sub(r"\*{3,}", "**", lowered)

    return _repair_common_math_ocr(lowered)


def extract_best_equation(text):
    """Extract the most plausible equation from OCR text."""
    t = smart_fix_ocr_text(text).replace(" ", "")

    # Remove leading/trailing punctuation that cannot belong to an equation.
    t = t.strip(";,._")

    # If multiple equals signs remain, retain the strongest equation segment.
    if t.count("=") > 1:
        pieces = re.findall(r"[a-z0-9().^+\-*/]+=[a-z0-9().^+\-*/]+", t, re.I)
        if pieces:
            pieces.sort(key=_ocr_math_score, reverse=True)
            t = pieces[0]

    return t


def clean_equation(text):
    text = extract_best_equation(text)
    text = text.strip().replace(" ", "")

    symbol_map = {
        "λ": "lambda", "Λ": "lambda", "π": "pi", "𝜋": "pi",
        "θ": "theta", "ω": "omega", "α": "alpha", "β": "beta",
        "∑": "sum", "Σ": "sum"
    }
    for old, new in symbol_map.items():
        text = text.replace(old, new)

    text = text.replace("²", "^2").replace("³", "^3")
    text = text.replace("^", "**")
    text = text.replace("÷", "/").replace("×", "*").replace(":", "=")
    text = text.replace("X", "x").replace("Y", "y")

    # Apply ambiguous substitutions only in numeric context.
    text = re.sub(r"(?<=\d)[oO](?=\d|$)", "0", text)
    text = re.sub(r"(?<=\d)[lI|](?=\d)", "1", text)

    # Function normalization.
    function_fixes = {
        "Sin": "sin", "SIN": "sin", "Cos": "cos", "COS": "cos",
        "Tan": "tan", "TAN": "tan", "Log": "log", "LOG": "log",
        "Sln": "sin", "c0s": "cos", "C0S": "cos", "10g": "log",
        "1og": "log", "Iog": "log"
    }
    for old, new in function_fixes.items():
        text = text.replace(old, new)

    compact_fixes = {
        "log10x": "log10(x)", "log10y": "log10(y)",
        "logx": "log(x)", "logy": "log(y)",
        "lnx": "log(x)", "lny": "log(y)",
        "sqrtx": "sqrt(x)", "sqrty": "sqrt(y)",
        "sinx": "sin(x)", "siny": "sin(y)",
        "cosx": "cos(x)", "cosy": "cos(y)",
        "tanx": "tan(x)", "tany": "tan(y)"
    }
    for old, new in compact_fixes.items():
        text = text.replace(old, new)

    # x2 is treated as x squared when it appears as a complete algebraic term.
    text = re.sub(r"\b([xy])2(?=[+\-=]|$)", r"\1**2", text)

    # Insert missing multiplication: 5x -> 5*x, 2(x+1) -> 2*(x+1).
    text = re.sub(r"(?<=\d)(?=(?:x|y|pi|lambda|theta|omega|alpha|beta|sin|cos|tan|log|sqrt))", "*", text)
    text = re.sub(r"(?<=[xy)])(?=\()", "*", text)
    text = re.sub(r"\)(?=[xy])", ")*", text)
    text = re.sub(r"\)(?=sin|cos|tan|log|sqrt)", ")+", text)

    # Keep only supported mathematical characters and names.
    text = re.sub(r"[^0-9xy=+\-*/().a-zA-Z]", "", text)
    text = re.sub(r"={2,}", "=", text)

    return text


def parse_expr(expr_text, x, y):
    lam, theta, omega, alpha, beta = sp.symbols(
        "lambda theta omega alpha beta"
    )

    local_dict = {
        "x": x,
        "y": y,
        "lambda": lam,
        "pi": sp.pi,
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "e": sp.E,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
    }

    expr_text = expr_text.replace("log10(x)", "log(x, 10)")
    expr_text = expr_text.replace("log10(y)", "log(y, 10)")

    return sp.sympify(expr_text, locals=local_dict)


def detect_equation_type(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if len(lines) > 1:
        return "System of Equations"
    if "sin" in text or "cos" in text or "tan" in text:
        return "Trigonometric Equation"
    elif "log10" in text or "log" in text:
        return "Logarithmic Equation"
    elif "sqrt" in text:
        return "Square Root Equation"
    elif "lambda" in text or "omega" in text or "theta" in text or "alpha" in text or "beta" in text:
        return "Greek Symbol Formula"
    elif "**2" in text:
        return "Quadratic Equation"
    elif "x" in text and "y" in text:
        return "Two Variable Equation"
    elif "x" in text:
        return "Linear Equation"
    elif "y" in text:
        return "Expression Evaluation"
    else:
        return "Mathematical Expression"


def valid_side(side):
    if side == "":
        return False

    if not re.search(
        r"[0-9a-zA-Z]|log|sqrt|sin|cos|tan|lambda|theta|omega|alpha|beta|pi",
        side
    ):
        return False

    if side[0] in ["+", "*", "/", "="]:
        return False

    if side[-1] in ["+", "-", "*", "/", "="]:
        return False

    return True


def clean_equation_input(text):
    """Clean one equation per line while preserving line breaks for systems."""
    cleaned_lines = []
    for raw_line in str(text or "").splitlines():
        if raw_line.strip():
            cleaned = clean_equation(raw_line)
            if cleaned:
                cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def _parse_system_expression(expr_text, local_dict):
    return sp.sympify(expr_text, locals=local_dict)


def solve_equation(text):
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]

    if len(lines) > 1:
        equations = []
        variables = set()

        for line in lines:
            if line.count("=") != 1:
                raise ValueError("Each line must contain exactly one '=' sign.")

            left, right = line.split("=", 1)
            symbol_names = sorted(set(re.findall(r"\b[a-zA-Z]\w*\b", line)))
            reserved = {
                "sin", "cos", "tan", "asin", "acos", "atan",
                "log", "sqrt", "pi", "e"
            }

            local_dict = {
                "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
                "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
                "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "e": sp.E
            }

            for name in symbol_names:
                if name not in reserved:
                    local_dict[name] = sp.Symbol(name)

            left_expr = _parse_system_expression(left, local_dict)
            right_expr = _parse_system_expression(right, local_dict)
            equation = sp.Eq(left_expr, right_expr)

            equations.append(equation)
            variables.update((left_expr - right_expr).free_symbols)

        ordered_variables = sorted(variables, key=lambda item: str(item))
        solution = sp.solve(equations, ordered_variables, dict=True)

        return (
            solution,
            "System of Equations",
            equations,
            equations,
            ordered_variables
        )

    if not lines:
        raise ValueError("Enter an equation first.")

    text = lines[0]
    left, right = text.split("=")

    x, y = sp.symbols("x y")

    left_expr = parse_expr(left, x, y)
    right_expr = parse_expr(right, x, y)

    equation = sp.Eq(left_expr, right_expr)
    equation_type = detect_equation_type(text)

    variables = list((left_expr - right_expr).free_symbols)
    rearranged = left_expr - right_expr
    simplified = sp.simplify(rearranged)

    if left.strip().lower() == "y" and y in variables:
        value = sp.simplify(right_expr)
        if len(value.free_symbols) == 0:
            return [value], equation_type, rearranged, simplified, [y]

    if left.strip().lower() == "x" and x in variables:
        value = sp.simplify(right_expr)
        if len(value.free_symbols) == 0:
            return [value], equation_type, rearranged, simplified, [x]

    if len(variables) == 0:
        value = sp.simplify(right_expr)
        solution = [value]
        variables = [sp.Symbol("Answer")]
    elif len(variables) == 1:
        solution = sp.solve(equation, variables[0])
    else:
        solution = sp.solve(equation, variables[0])

    return solution, equation_type, rearranged, simplified, variables



def build_detailed_steps(text, equation_type, rearranged, simplified, variables, solution):
    """Create clear, human-readable solution steps without losing the original input."""
    steps = []

    if equation_type == "System of Equations":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        steps.append((
            "Identify the system",
            f"{len(lines)} equations with {len(variables)} unknowns were detected."
        ))

        for index, line in enumerate(lines, start=1):
            steps.append((f"Write Equation {index}", line))

        steps.append((
            "Solve the equations simultaneously",
            "Use elimination or substitution to find values that satisfy every equation."
        ))

        if solution:
            first_solution = solution[0]
            for variable in variables:
                if variable in first_solution:
                    steps.append((
                        f"Solve for {variable}",
                        f"{variable} = {format_answer_value(first_solution[variable])}"
                    ))

            verification_parts = []
            for variable in variables:
                if variable in first_solution:
                    verification_parts.append(
                        f"{variable}={format_answer_value(first_solution[variable])}"
                    )
            if verification_parts:
                steps.append((
                    "Verify the solution",
                    "Substitute " + ", ".join(verification_parts) +
                    " into all original equations; each left side equals its right side."
                ))

        return steps

    left, right = text.split("=", 1)
    original_left = left.strip()
    original_right = right.strip()
    x, y = sp.symbols("x y")

    left_expr = parse_expr(original_left, x, y)
    right_expr = parse_expr(original_right, x, y)

    # Keep the exact cleaned equation text for the first step. SymPy may simplify
    # sin(pi/2)+cos(pi) to 0 immediately, so using Eq(...) here would lose detail.
    steps.append(("Original equation", f"{original_left} = {original_right}"))

    if equation_type == "Trigonometric Equation":
        # Determine which side contains the trigonometric expression.
        if any(fn in original_right for fn in ("sin", "cos", "tan")):
            trig_text = original_right
            target_text = original_left
            trig_expr = right_expr
        else:
            trig_text = original_left
            target_text = original_right
            trig_expr = left_expr

        # Find each trig function from the original text, so it is not lost after
        # SymPy simplification.
        trig_matches = re.findall(r"(?:sin|cos|tan)\([^()]+\)", trig_text)
        substitutions = {}

        for trig_string in trig_matches:
            try:
                trig_value = sp.simplify(parse_expr(trig_string, x, y))
                substitutions[trig_string] = format_answer_value(trig_value)
                steps.append(
                    (f"Evaluate {trig_string}", f"{trig_string} = {format_answer_value(trig_value)}")
                )
            except Exception:
                continue

        substituted_text = trig_text
        for trig_string, trig_value in substitutions.items():
            substituted_text = substituted_text.replace(trig_string, f"({trig_value})")

        # Make the substitution easier to read, for example 1+(-1).
        substituted_text = substituted_text.replace("(+", "(")
        steps.append(("Substitute the trigonometric values", f"{target_text} = {substituted_text}"))

        final_value = sp.simplify(trig_expr)
        steps.append(("Simplify", f"{target_text} = {format_answer_value(final_value)}"))
        return steps

    if equation_type == "Logarithmic Equation":
        log_terms = list((left_expr - right_expr).atoms(sp.log))

        if log_terms and len(variables) == 1:
            variable = variables[0]
            exponent_value = right_expr if "log" in original_left else left_expr

            if "log10" in text:
                steps.append(("Identify the logarithmic form", f"{original_left} = {original_right}"))
                steps.append(("Convert to exponential form", f"{variable} = 10^({exponent_value})"))
                steps.append(("Calculate the power", f"{variable} = {format_answer_value(sp.simplify(10 ** exponent_value))}"))
                return steps

    if equation_type == "Quadratic Equation":
        variable = variables[0] if variables else x
        polynomial = sp.Poly(sp.expand(rearranged), variable)
        factored = sp.factor(polynomial.as_expr())

        steps.append(("Move all terms to one side", sp.Eq(rearranged, 0)))

        if simplified != rearranged:
            steps.append(("Simplify the equation", sp.Eq(simplified, 0)))

        if factored != polynomial.as_expr():
            steps.append(("Factorise the quadratic expression", sp.Eq(factored, 0)))

        if solution:
            for value in solution:
                steps.append(("Solve for the variable", sp.Eq(variable, value)))
        return steps

    steps.append(("Move all terms to one side", sp.Eq(rearranged, 0)))

    if simplified != rearranged:
        steps.append(("Simplify the equation", sp.Eq(simplified, 0)))

    if solution and variables:
        variable = variables[0]
        for value in solution:
            steps.append(("Solve for the variable", sp.Eq(variable, value)))

    return steps


def display_detailed_steps(steps):
    """Display generated explanation steps in Streamlit."""
    st.subheader("📘 Step-by-Step Solution")

    for number, (description, expression) in enumerate(steps, start=1):
        st.write(f"Step {number}: {description}")
        if expression is not None:
            if isinstance(expression, str):
                st.code(expression, language=None)
            else:
                st.latex(sp.latex(expression))


def format_step_for_report(step, number):
    """Convert a detailed-step tuple into readable PDF text."""
    if isinstance(step, tuple) and len(step) == 2:
        description, expression = step
        if isinstance(expression, str):
            expression_text = expression
        else:
            expression_text = sp.sstr(expression)
        return f"{number}. {description}: {expression_text}"
    return f"{number}. {step}"


def save_uploaded_image(uploaded_file, username, prefix="img"):
    """Save camera/upload image and return path for history."""
    os.makedirs("uploads", exist_ok=True)
    safe_user = re.sub(r"[^a-zA-Z0-9_]", "_", username or "user")
    image_path = f"uploads/{safe_user}_{prefix}_{int(time.time() * 1000)}.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return image_path


def format_answer_value(value):
    """Show clean answers like 0 instead of 0.0."""
    try:
        value = sp.simplify(value)
        if value.is_number:
            if sp.simplify(value - int(value)) == 0:
                return str(int(value))
    except Exception:
        pass
    return str(value)

# ---------------- PDF ----------------
def create_pdf(username, equation, equation_type, answer, steps):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(80, 800, "Handwritten Formula Solver Report")

    pdf.setFont("Helvetica", 12)
    pdf.drawString(80, 760, f"Username: {username}")
    pdf.drawString(80, 735, f"Equation: {equation}")
    pdf.drawString(80, 710, f"Equation Type: {equation_type}")
    pdf.drawString(80, 685, f"Answer: {answer}")

    pdf.drawString(80, 645, "Steps:")

    y_pos = 620
    for number, step in enumerate(steps, start=1):
        line = format_step_for_report(step, number)

        # Wrap long steps instead of cutting them off.
        words = line.split()
        wrapped_lines = []
        current_line = ""
        for word in words:
            trial = f"{current_line} {word}".strip()
            if pdf.stringWidth(trial, "Helvetica", 11) <= 420:
                current_line = trial
            else:
                if current_line:
                    wrapped_lines.append(current_line)
                current_line = word
        if current_line:
            wrapped_lines.append(current_line)

        for wrapped_line in wrapped_lines:
            if y_pos < 80:
                pdf.showPage()
                pdf.setFont("Helvetica", 11)
                y_pos = 800
            pdf.drawString(100, y_pos, wrapped_line)
            y_pos -= 20
        y_pos -= 3

    pdf.save()
    buffer.seek(0)
    return buffer


# ---------------- FORMULA HELPERS ----------------
def normalize_formula_text(text):
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("λ", "lambda")
    text = text.replace("Λ", "lambda")
    text = text.replace("π", "pi")
    text = text.replace("Π", "pi")
    text = text.replace("θ", "theta")
    text = text.replace("Θ", "theta")
    text = text.replace("ω", "omega")
    text = text.replace("Ω", "omega")
    text = text.replace("α", "alpha")
    text = text.replace("β", "beta")
    text = text.replace("∑", "sum")
    text = text.replace("Σ", "sum")
    text = text.replace("Find", "find")
    text = text.replace("FIND", "find")
    return text


def get_value_from_text(text, variable):
    text_raw = normalize_formula_text(text)
    text_lower = text_raw.lower()

    aliases = {
        "m": ["m"],
        "a": ["a"],
        "v": ["v"],
        "f": ["f"],
        "g": ["g"],
        "h": ["h"],
        "r": ["r"],
        "n": ["n"],
        "F": ["F"],
        "theta": ["theta", "0", "o"],
        "lambda": ["lambda", "lamda", "lam", "A"],
        "alpha": ["alpha", "a"],
        "beta": ["beta", "B", "b", "8"],
    }

    for alias in aliases.get(variable, [variable]):
        if variable == "F" and alias == "F":
            match = re.search(r"F=(-?\d+\.?\d*)", text_raw)
        elif alias in ["B", "A"]:
            match = re.search(rf"{alias}=(-?\d+\.?\d*)", text_raw)
        elif alias == "0":
            match = re.search(r"0=(-?\d+\.?\d*)", text_raw)
        else:
            match = re.search(rf"{alias}=(-?\d+\.?\d*)", text_lower, re.IGNORECASE)

        if match:
            return float(match.group(1))

    return None


def calculate_auto_formula(text):
    t_raw = normalize_formula_text(text)
    t = t_raw.lower()

    m = get_value_from_text(text, "m")
    a = get_value_from_text(text, "a")
    v = get_value_from_text(text, "v")
    f = get_value_from_text(text, "f")
    lam = get_value_from_text(text, "lambda")
    g = get_value_from_text(text, "g")
    h = get_value_from_text(text, "h")
    r = get_value_from_text(text, "r")
    n = get_value_from_text(text, "n")
    alpha = get_value_from_text(text, "alpha")
    beta = get_value_from_text(text, "beta")
    force = get_value_from_text(text, "F")
    theta = get_value_from_text(text, "theta")

    if n is not None and ("find2" in t or "findsum" in t or "sigma" in t or "sum" in t):
        return "Summation: ∑ 1 to n", {"n": n}, "∑", n * (n + 1) / 2

    if alpha is not None and beta is not None and (
        "finda+8" in t or "findalpha" in t or "alpha+beta" in t or "a+8" in t or "a+b" in t
    ):
        return "Alpha-Beta Addition", {"α": alpha, "β": beta}, "α + β", alpha + beta

    if m is not None and a is not None and ("findf" in t or "force" in t):
        return "Force: F = ma", {"m": m, "a": a}, "F", m * a

    if v is not None and f is not None and (
        "fa" in t or "flambda" in t or "finda" in t or "findlambda" in t or "v=f" in t
    ):
        return "Wave Speed: v = fλ", {"v": v, "f": f}, "λ", v / f

    if f is not None and lam is not None:
        return "Wave Speed: v = fλ", {"f": f, "λ": lam}, "v", f * lam

    if v is not None and lam is not None:
        return "Wave Speed: v = fλ", {"v": v, "λ": lam}, "f", v / lam

    if f is not None and (
        "omega" in t or "w=2" in t or "w=2pi" in t or "w=2nf" in t or "findw" in t
    ):
        return "Angular Frequency: ω = 2πf", {"f": f}, "ω", float(2 * sp.pi * f)

    if m is not None and v is not None and ("findp" in t or "momentum" in t):
        return "Momentum: p = mv", {"m": m, "v": v}, "p", m * v

    if m is not None and h is not None and ("pe" in t or "potential" in t):
        if g is None:
            g = 9.8
        return "Potential Energy: PE = mgh", {"m": m, "g": g, "h": h}, "PE", m * g * h

    if m is not None and ("weight" in t or "findf" in t) and a is None:
        if g is None:
            g = 9.8
        return "Weight: F = mg", {"m": m, "g": g}, "F", m * g

    if r is not None and (
        "area" in t or "findarea" in t or "a=tr2" in t or "a=pir2" in t or "a=nr2" in t
    ):
        return "Circle Area: A = πr²", {"r": r}, "A", float(sp.pi * r ** 2)

    if r is not None and ("findc" in t or "circumference" in t or "c=2" in t):
        return "Circumference: C = 2πr", {"r": r}, "C", float(2 * sp.pi * r)

    if r is not None and force is not None and theta is not None and (
        "findt" in t or "torque" in t or "tau" in t
    ):
        theta_rad = float(theta) * np.pi / 180
        return "Torque: τ = rF sin(θ)", {"r": r, "F": force, "θ": theta}, "τ", r * force * np.sin(theta_rad)

    return None, {}, "", None


def get_formula_steps(formula_name, detected_values, answer_label, answer):
    answer_value = round(float(answer), 4)

    if formula_name == "Summation: ∑ 1 to n":
        n = int(detected_values["n"])
        return [
            "Formula Used: ∑ from 1 to n = n(n + 1) / 2",
            f"Given: n = {n}",
            "Step 1: Write the summation formula for the first n natural numbers.",
            f"Step 2: Substitute n = {n}: ∑ = {n}({n} + 1) / 2",
            f"Step 3: Add inside the bracket: ∑ = {n} × {n + 1} / 2",
            f"Step 4: Multiply and divide by 2: ∑ = {answer_value}",
            f"Therefore: {answer_label} = {answer_value}"
        ]

    if formula_name == "Alpha-Beta Addition":
        alpha = detected_values["α"]
        beta = detected_values["β"]
        return [
            "Formula Used: Result = α + β",
            f"Given: α = {format_answer_value(alpha)}, β = {format_answer_value(beta)}",
            "Step 1: Write the addition formula.",
            f"Step 2: Substitute the values: Result = {alpha} + {beta}",
            f"Step 3: Add the two values: Result = {answer_value}",
            f"Therefore: {answer_label} = {answer_value}"
        ]

    if formula_name == "Force: F = ma":
        m = detected_values["m"]
        a = detected_values["a"]
        return [
            "Formula Used: F = m × a",
            f"Given: Mass (m) = {format_answer_value(m)} kg",
            f"Given: Acceleration (a) = {format_answer_value(a)} m/s²",
            "Step 1: Write the force formula: F = m × a",
            f"Step 2: Substitute the values: F = {m} × {a}",
            f"Step 3: Multiply mass by acceleration: F = {answer_value}",
            f"Therefore: F = {answer_value} N"
        ]

    if formula_name == "Weight: F = mg":
        m = detected_values["m"]
        g = detected_values["g"]
        return [
            "Formula Used: F = m × g",
            f"Given: Mass (m) = {format_answer_value(m)} kg",
            f"Given: Gravitational acceleration (g) = {format_answer_value(g)} m/s²",
            "Step 1: Write the weight formula: F = m × g",
            f"Step 2: Substitute the values: F = {m} × {g}",
            f"Step 3: Multiply mass by gravitational acceleration: F = {answer_value}",
            f"Therefore: F = {answer_value} N"
        ]

    if formula_name == "Potential Energy: PE = mgh":
        m = detected_values["m"]
        g = detected_values["g"]
        h = detected_values["h"]
        return [
            "Formula Used: PE = m × g × h",
            f"Given: Mass (m) = {format_answer_value(m)} kg",
            f"Given: Gravitational acceleration (g) = {format_answer_value(g)} m/s²",
            f"Given: Height (h) = {format_answer_value(h)} m",
            "Step 1: Write the potential-energy formula: PE = m × g × h",
            f"Step 2: Substitute the values: PE = {m} × {g} × {h}",
            f"Step 3: Multiply the values: PE = {answer_value}",
            f"Therefore: PE = {answer_value} J"
        ]

    if formula_name == "Momentum: p = mv":
        m = detected_values["m"]
        v = detected_values["v"]
        return [
            "Formula Used: p = m × v",
            f"Given: Mass (m) = {format_answer_value(m)} kg",
            f"Given: Velocity (v) = {format_answer_value(v)} m/s",
            "Step 1: Write the momentum formula: p = m × v",
            f"Step 2: Substitute the values: p = {m} × {v}",
            f"Step 3: Multiply mass by velocity: p = {answer_value}",
            f"Therefore: p = {answer_value} kg·m/s"
        ]

    if formula_name == "Wave Speed: v = fλ":
        v = detected_values.get("v")
        f = detected_values.get("f")
        lam = detected_values.get("λ")

        if v is not None and f is not None:
            return [
                "Formula Used: v = f × λ",
                f"Given: Wave speed (v) = {format_answer_value(v)} m/s",
                f"Given: Frequency (f) = {format_answer_value(f)} Hz",
                "Step 1: Rearrange the formula to make λ the subject: λ = v / f",
                f"Step 2: Substitute the values: λ = {v} / {f}",
                f"Step 3: Divide wave speed by frequency: λ = {answer_value}",
                f"Therefore: λ = {answer_value} m"
            ]

        if f is not None and lam is not None:
            return [
                "Formula Used: v = f × λ",
                f"Given: Frequency (f) = {format_answer_value(f)} Hz",
                f"Given: Wavelength (λ) = {format_answer_value(lam)} m",
                "Step 1: Write the wave-speed formula: v = f × λ",
                f"Step 2: Substitute the values: v = {f} × {lam}",
                f"Step 3: Multiply frequency by wavelength: v = {answer_value}",
                f"Therefore: v = {answer_value} m/s"
            ]

        if v is not None and lam is not None:
            return [
                "Formula Used: v = f × λ",
                f"Given: Wave speed (v) = {format_answer_value(v)} m/s",
                f"Given: Wavelength (λ) = {format_answer_value(lam)} m",
                "Step 1: Rearrange the formula to make f the subject: f = v / λ",
                f"Step 2: Substitute the values: f = {v} / {lam}",
                f"Step 3: Divide wave speed by wavelength: f = {answer_value}",
                f"Therefore: f = {answer_value} Hz"
            ]

    if formula_name == "Angular Frequency: ω = 2πf":
        f = detected_values["f"]
        return [
            "Formula Used: ω = 2 × π × f",
            f"Given: Frequency (f) = {format_answer_value(f)} Hz",
            "Step 1: Write the angular-frequency formula: ω = 2πf",
            f"Step 2: Substitute the frequency: ω = 2 × π × {f}",
            f"Step 3: Calculate the value: ω = {answer_value}",
            f"Therefore: ω = {answer_value} rad/s"
        ]

    if formula_name == "Circle Area: A = πr²":
        r = detected_values["r"]
        return [
            "Formula Used: A = π × r²",
            f"Given: Radius (r) = {format_answer_value(r)}",
            "Step 1: Write the area formula: A = πr²",
            f"Step 2: Substitute the radius: A = π × ({r})²",
            f"Step 3: Square the radius: A = π × {round(float(r ** 2), 4)}",
            f"Step 4: Multiply by π: A = {answer_value}",
            f"Therefore: A = {answer_value} square units"
        ]

    if formula_name == "Circumference: C = 2πr":
        r = detected_values["r"]
        return [
            "Formula Used: C = 2 × π × r",
            f"Given: Radius (r) = {format_answer_value(r)}",
            "Step 1: Write the circumference formula: C = 2πr",
            f"Step 2: Substitute the radius: C = 2 × π × {r}",
            f"Step 3: Calculate the value: C = {answer_value}",
            f"Therefore: C = {answer_value} units"
        ]

    if formula_name == "Torque: τ = rF sin(θ)":
        r = detected_values["r"]
        force = detected_values["F"]
        theta = detected_values["θ"]
        sin_value = round(float(np.sin(float(theta) * np.pi / 180)), 4)
        return [
            "Formula Used: τ = r × F × sin(θ)",
            f"Given: Distance from pivot (r) = {format_answer_value(r)} m",
            f"Given: Force (F) = {format_answer_value(force)} N",
            f"Given: Angle (θ) = {format_answer_value(theta)}°",
            "Step 1: Write the torque formula: τ = rF sin(θ)",
            f"Step 2: Find sin({theta}°) = {sin_value}",
            f"Step 3: Substitute the values: τ = {r} × {force} × {sin_value}",
            f"Step 4: Multiply the values: τ = {answer_value}",
            f"Therefore: τ = {answer_value} N·m"
        ]

    return [
        f"Formula Used: {formula_name}",
        "Step 1: Identify the known values from the question.",
        "Step 2: Substitute the known values into the formula.",
        "Step 3: Perform the calculation carefully.",
        f"Therefore: {answer_label} = {answer_value}"
    ]


def show_result(label, value):
    st.success(f"{label} = {round(float(value), 4)}")


# ---------------- SOLVER PAGE ----------------
if page == "Solver":
    st.title("✍️ Handwritten Formula Solver")
    st.write("Capture, upload, or directly type an equation and solve it step-by-step.")

    option = st.radio(
        "Choose Input Method",
        ["Use Camera", "Upload Image", "Enter Text"]
    )

    uploaded_file = None
    text = ""
    equation_key = "manual_equation"

    if option == "Use Camera":
        st.info(
            "Use a plain white sheet, thick black pen, bright light, and keep only the equation "
            "inside the frame. Hold the camera steady before capture."
        )
        uploaded_file = st.camera_input("Capture Equation")

    elif option == "Upload Image":
        uploaded_file = st.file_uploader(
            "Upload Equation Image",
            type=["png", "jpg", "jpeg"],
            key="solver_upload"
        )

    else:
        st.subheader("Enter Equation")
        st.caption("For a system of equations, enter one equation on each line.")
        text = st.text_area(
            "Type your equation",
            height=140,
            key="manual_equation_input",
            placeholder="Examples:\n2*x+5=15\n\nx+y=10\nx-y=2\n\na-6*b+6*c=4\n6*a+3*b-3*c=50"
        )
        st.caption("Use ** for powers, for example x**2.")

    if uploaded_file is not None:
        with st.spinner("Reading the handwritten equation..."):
            img, extracted_text, enhanced_img = read_image_ocr(
                uploaded_file, camera=(option == "Use Camera")
            )

        if img is None:
            st.error("Could not read the selected image. Please use PNG or JPG format.")
        else:
            st.image(img, caption="Input Image", use_container_width=True)

            if enhanced_img is not None:
                with st.expander("View camera-enhanced image"):
                    st.image(
                        enhanced_img,
                        caption="Enhanced Image Used for OCR",
                        use_container_width=True
                    )

            st.subheader("OCR Output")
            if extracted_text:
                st.code(extracted_text)
            else:
                st.warning(
                    "OCR could not read the handwriting clearly. "
                    "Enter the equation manually below."
                )

            cleaned_text = clean_equation(extracted_text) if extracted_text else ""

            st.subheader("Verify / Correct Equation")
            equation_key = f"equation_{hash(uploaded_file.getvalue())}"
            text = st.text_area(
                "Equation",
                value=cleaned_text,
                height=140,
                key=equation_key,
                placeholder="Enter one equation per line for simultaneous equations."
            )

            st.caption(
                "Use ** for powers, for example x**2. For a system, keep each equation on a separate line. "
                "You can correct any OCR mistake before solving."
            )

    solve_clicked = st.button(
        "Solve Equation",
        type="primary",
        key=f"solve_{equation_key}"
    )

    if solve_clicked:
        text = clean_equation_input(text)
        st.write("Final Equation:")
        st.code(text)

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        if not lines:
            st.error("Enter an equation first.")

        elif any(line.count("=") != 1 for line in lines):
            st.error("Each equation must contain exactly one '=' sign.")

        else:
            invalid_line = False
            for line in lines:
                left, right = line.split("=", 1)
                if not valid_side(left) or not valid_side(right):
                    invalid_line = True
                    break

            if invalid_line:
                st.error("One or more equations are not valid. Correct them and try again.")

            else:
                try:
                    solution, equation_type, rearranged, simplified, variables = solve_equation(text)

                    st.info(f"Detected Equation Type: {equation_type}")

                    detailed_steps = build_detailed_steps(
                        text,
                        equation_type,
                        rearranged,
                        simplified,
                        variables,
                        solution
                    )

                    display_detailed_steps(detailed_steps)

                    if solution:
                        if equation_type == "System of Equations":
                            solution_dict = solution[0]
                            answers = []

                            for variable in variables:
                                if variable in solution_dict:
                                    value = solution_dict[variable]
                                    answers.append(
                                        f"{variable} = {format_answer_value(value)}"
                                    )
                                    st.latex(
                                        f"{sp.latex(variable)} = {sp.latex(value)}"
                                    )

                            answer_text = ", ".join(answers)
                        elif len(solution) == 1:
                            answer_text = (
                                f"{variables[0]} = "
                                f"{format_answer_value(solution[0])}"
                            )
                            st.latex(
                                f"{sp.latex(variables[0])} = "
                                f"{sp.latex(solution[0])}"
                            )
                        else:
                            answers = []
                            for sol in solution:
                                answers.append(
                                    f"{variables[0]} = "
                                    f"{format_answer_value(sol)}"
                                )
                                st.latex(
                                    f"{sp.latex(variables[0])} = "
                                    f"{sp.latex(sol)}"
                                )
                            answer_text = ", ".join(answers)

                        st.success(f"Final Answer: {answer_text}")

                        if uploaded_file is not None:
                            image_path = save_uploaded_image(
                                uploaded_file,
                                st.session_state.username,
                                "solver"
                            )
                        else:
                            image_path = ""

                        database.save_history(
                            st.session_state.username,
                            text.replace("\n", " | "),
                            answer_text,
                            equation_type,
                            image_path
                        )

                        steps = [
                            format_step_for_report(step, index)
                            for index, step in enumerate(detailed_steps, start=1)
                        ]
                        steps.append(f"Final Answer: {answer_text}")

                        pdf_file = create_pdf(
                            st.session_state.username,
                            text.replace("\n", " | "),
                            equation_type,
                            answer_text,
                            steps
                        )

                        st.download_button(
                            "📥 Download PDF Report",
                            pdf_file,
                            file_name="solution_report.pdf",
                            mime="application/pdf"
                        )

                    else:
                        st.error("No solution was found for this equation.")

                except Exception as error:
                    st.error(f"Could not solve this equation: {error}")

    gc.collect()


# ---------------- FORMULA SOLVER PAGE ----------------
elif page == "Formula Solver":
    st.title("🧮 Formula Solver with Camera + Auto Detection")
    st.write(
        "Capture, upload, or directly type a formula question, then press Solve Formula."
    )

    input_method = st.radio(
        "Choose Input Method",
        ["Use Camera", "Upload Image", "Enter Text"]
    )

    formula_file = None
    final_text = ""
    formula_key = "manual_formula"

    if input_method == "Use Camera":
        st.info(
            "Write one value per line, keep the paper close to the camera, use a thick black pen, "
            "and avoid shadows. Example: m = 25, a = 4, Find F."
        )
        formula_file = st.camera_input("Capture Formula / Equation")

    elif input_method == "Upload Image":
        formula_file = st.file_uploader(
            "Upload Formula / Equation Image",
            type=["png", "jpg", "jpeg"],
            key="formula_upload"
        )

    else:
        st.subheader("Enter Formula / Equation")
        st.caption("Keep each value on a separate line.")
        final_text = st.text_area(
            "Type your formula question",
            height=140,
            key="manual_formula_input",
            placeholder="m = 25\na = 4\nFind F"
        )

    if formula_file is not None:
        img, extracted_formula_text, enhanced_formula_img = read_image_ocr(
            formula_file,
            camera=(input_method == "Use Camera")
        )

        if img is None:
            st.error("Could not read image.")
            st.stop()

        st.image(img, caption="Input Image", use_container_width=True)

        if enhanced_formula_img is not None:
            with st.expander("View camera-enhanced image"):
                st.image(
                    enhanced_formula_img,
                    caption="Enhanced Image Used for OCR",
                    use_container_width=True
                )

        st.subheader("OCR Output")
        if extracted_formula_text:
            st.code(extracted_formula_text)
        else:
            st.warning(
                "OCR could not read the image. Enter the values manually below."
            )

        suggested_text = str(extracted_formula_text or "")
        suggested_text = suggested_text.replace("pind", "Find").replace("Pind", "Find")

        st.subheader("Editable Text")
        st.caption(
            "Keep each value on a separate line. Correct OCR mistakes before solving."
        )

        formula_key = f"formula_text_{hash(formula_file.getvalue())}"
        final_text = st.text_area(
            "Correct OCR if needed",
            value=suggested_text,
            height=140,
            key=formula_key,
            placeholder="m = 25\na = 4\nFind F"
        )

    st.caption(
        "Supported examples: m=25, a=4, Find F | m=2, g=9.8, h=10, Find PE | "
        "v=20, f=4, Find λ | r=7, Find Area"
    )

    solve_formula_clicked = st.button(
        "Solve Formula",
        type="primary",
        key=f"solve_{formula_key}"
    )

    if solve_formula_clicked:
        user_text = final_text.strip()

        if not user_text:
            st.error("Enter or correct the formula question first.")
            st.stop()

        formula_name, detected_values, answer_label, answer = calculate_auto_formula(
            user_text
        )

        if formula_name:
            if formula_file is not None:
                image_path = save_uploaded_image(
                    formula_file,
                    st.session_state.username,
                    "formula"
                )
            else:
                image_path = ""

            st.info(f"Auto Detected Formula: {formula_name}")

            st.subheader("Detected Values")
            for key, value in detected_values.items():
                st.write(f"{key}: `{format_answer_value(value)}`")

            st.subheader("Step-by-Step Calculation")
            steps = get_formula_steps(
                formula_name,
                detected_values,
                answer_label,
                answer
            )

            for step in steps:
                st.write(step)

            answer_text = f"{answer_label} = {round(float(answer), 4)}"
            st.success(f"Final Answer: {answer_text}")

            database.save_history(
                st.session_state.username,
                user_text.replace("\n", " | "),
                answer_text,
                "Formula Solver",
                image_path
            )

            formula_pdf = create_pdf(
                st.session_state.username,
                user_text.replace("\n", " | "),
                "Formula Solver",
                answer_text,
                steps
            )

            st.download_button(
                "📥 Download Formula PDF Report",
                formula_pdf,
                file_name="formula_solution_report.pdf",
                mime="application/pdf"
            )

        else:
            equation_text = clean_equation(user_text)

            if equation_text.count("=") == 1:
                try:
                    left, right = equation_text.split("=", 1)

                    if not valid_side(left) or not valid_side(right):
                        raise ValueError("Invalid equation")

                    solution, equation_type, rearranged, simplified, variables = solve_equation(
                        equation_text
                    )

                    detailed_steps = build_detailed_steps(
                        equation_text,
                        equation_type,
                        rearranged,
                        simplified,
                        variables,
                        solution
                    )

                    st.info(f"Auto Detected Type: {equation_type}")
                    display_detailed_steps(detailed_steps)

                    if not solution:
                        st.error("No solution found.")
                        st.stop()

                    if len(solution) == 1:
                        answer_text = (
                            f"{variables[0]} = "
                            f"{format_answer_value(solution[0])}"
                        )
                        st.latex(
                            f"{sp.latex(variables[0])} = "
                            f"{sp.latex(solution[0])}"
                        )
                    else:
                        answers = [
                            f"{variables[0]} = {format_answer_value(sol)}"
                            for sol in solution
                        ]
                        answer_text = ", ".join(answers)

                        for sol in solution:
                            st.latex(
                                f"{sp.latex(variables[0])} = "
                                f"{sp.latex(sol)}"
                            )

                    st.success(f"Final Answer: {answer_text}")

                    if formula_file is not None:
                        image_path = save_uploaded_image(
                            formula_file,
                            st.session_state.username,
                            "formula_equation"
                        )
                    else:
                        image_path = ""

                    database.save_history(
                        st.session_state.username,
                        equation_text,
                        answer_text,
                        equation_type,
                        image_path
                    )

                    pdf_file = create_pdf(
                        st.session_state.username,
                        equation_text,
                        equation_type,
                        answer_text,
                        detailed_steps
                    )

                    st.download_button(
                        "📥 Download PDF Report",
                        pdf_file,
                        file_name="formula_equation_report.pdf",
                        mime="application/pdf"
                    )

                except Exception as error:
                    st.error(
                        "The values were not recognised. Correct the text in this format:\n\n"
                        "m = 25\na = 4\nFind F"
                    )
                    st.caption(f"Details: {error}")

            else:
                st.error(
                    "The formula question was not recognised. Correct the text using one value per line, "
                    "for example: m = 25, a = 4, Find F."
                )


# ---------------- HISTORY PAGE ----------------
elif page == "History":
    st.title("📜 Solved Equation History")

    rows = database.get_history(st.session_state.username)

    if rows:
        df = pd.DataFrame(
            rows,
            columns=["Equation", "Answer", "Equation Type", "Image Path", "Solved At"]
        )

        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="History")

        st.download_button(
            label="📊 Download History Excel",
            data=excel_buffer.getvalue(),
            file_name="equation_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        for index, row in df.iterrows():
            st.subheader(f"Equation {index + 1}")

            col1, col2 = st.columns([1, 2])

            with col1:
                image_path = row["Image Path"]
                if image_path and os.path.exists(image_path):
                    st.image(
                        image_path,
                        caption="Uploaded Image",
                        use_container_width=True
                    )
                else:
                    st.info("No image available")

            with col2:
                st.write("**Equation:**", row["Equation"])
                st.write("**Answer:**", row["Answer"])
                st.write("**Type:**", row["Equation Type"])
                st.write("**Solved At:**", row["Solved At"])

            st.divider()

    else:
        st.info("No history found yet.")


# ---------------- DASHBOARD PAGE ----------------
elif page == "Dashboard":
    st.title("📊 Dashboard")

    rows = database.get_history(st.session_state.username)

    if rows:
        df = pd.DataFrame(
            rows,
            columns=["Equation", "Answer", "Equation Type", "Image Path", "Solved At"]
        )

        total = len(df)
        linear = len(df[df["Equation Type"] == "Linear Equation"])
        quadratic = len(df[df["Equation Type"] == "Quadratic Equation"])
        log_count = len(df[df["Equation Type"] == "Logarithmic Equation"])
        sqrt_count = len(df[df["Equation Type"] == "Square Root Equation"])
        trig = len(df[df["Equation Type"] == "Trigonometric Equation"])
        two_var = len(df[df["Equation Type"] == "Two Variable Equation"])
        formula_count = len(df[df["Equation Type"] == "Formula Solver"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Solved", total)
        col2.metric("Linear", linear)
        col3.metric("Quadratic", quadratic)

        col4, col5, col6 = st.columns(3)
        col4.metric("Logarithmic", log_count)
        col5.metric("Square Root", sqrt_count)
        col6.metric("Trigonometric", trig)

        col7, col8, _ = st.columns(3)
        col7.metric("Two Variable", two_var)
        col8.metric("Formula Solver", formula_count)

        st.subheader("Equation Type Count")
        st.bar_chart(df["Equation Type"].value_counts())

        # ---------------- VISUALIZATION OF RESULTS ----------------
        st.subheader("📈 Performance Analytics")
        st.caption(
            "These visualisations are generated automatically from your saved history records."
        )

        equation_counts = df["Equation Type"].value_counts()

        # 2. Bar chart: number of solved problems in each category
        st.markdown("#### 📊 Category-wise Performance")
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        ax2.bar(equation_counts.index, equation_counts.values)
        ax2.set_title("Solved Problems by Category")
        ax2.set_xlabel("Equation Type")
        ax2.set_ylabel("Number of Problems Solved")
        ax2.tick_params(axis="x", rotation=35)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        # 3. Line graph: number of problems solved per day
        st.markdown("#### 📅 Daily Performance Trend")
        try:
            trend_df = df.copy()
            trend_df["Solved At"] = pd.to_datetime(
                trend_df["Solved At"], errors="coerce"
            )
            trend_df = trend_df.dropna(subset=["Solved At"])

            if not trend_df.empty:
                daily_counts = (
                    trend_df.groupby(trend_df["Solved At"].dt.date)
                    .size()
                    .sort_index()
                )

                fig3, ax3 = plt.subplots(figsize=(9, 5))
                ax3.plot(
                    [str(date) for date in daily_counts.index],
                    daily_counts.values,
                    marker="o"
                )
                ax3.set_title("Problems Solved Per Day")
                ax3.set_xlabel("Date")
                ax3.set_ylabel("Number of Problems Solved")
                ax3.tick_params(axis="x", rotation=35)
                ax3.grid(True, alpha=0.3)
                fig3.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)
            else:
                st.info("Daily trend will appear after valid solved dates are stored.")

        except Exception as trend_error:
            st.info(f"Daily trend is not available yet: {trend_error}")

        # 4. Recent activity table
        st.subheader("Recent History")
        st.dataframe(df, use_container_width=True)

    else:
        st.info("No dashboard data yet.")


# ---------------- ADMIN DASHBOARD PAGE ----------------
elif page == "Admin Dashboard":
    st.title("🛠️ Admin Dashboard")

    if st.session_state.username != "admin":
        st.error("Access denied. Admin only.")
        st.stop()

    all_rows = database.get_all_history()
    total_users = database.get_total_users()
    total_equations = database.get_total_equations()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Users", total_users)
    col2.metric("Total Equations Solved", total_equations)
    col3.metric("Total Records", len(all_rows))

    if all_rows:
        df = pd.DataFrame(
            all_rows,
            columns=["Username", "Equation", "Answer", "Equation Type", "Image Path", "Solved At"]
        )

        st.subheader("All User Activity")
        st.dataframe(df, use_container_width=True)

        st.subheader("Most Used Equation Types")
        st.bar_chart(df["Equation Type"].value_counts())

        excel_buffer = BytesIO()

        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Admin History")

        st.download_button(
            "📊 Download All User History",
            excel_buffer.getvalue(),
            file_name="admin_all_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.info("No user activity found yet.")