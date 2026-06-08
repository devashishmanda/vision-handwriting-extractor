import streamlit as st
import os, json, time, re, base64
from io import BytesIO
from PIL import Image
import pypdfium2 as pdfium
from docx import Document
import pandas as pd
from groq import Groq

try:
    import easyocr
    import numpy as np
    OCR_AVAILABLE = True
    reader = easyocr.Reader(['en'], gpu=False)
except Exception:
    OCR_AVAILABLE = False

st.set_page_config(page_title="Text Extractor Hub", page_icon="🔐", layout="wide")

api_key_available = "GROQ_API_KEY" in st.secrets

default_blueprint = {
    "FOR OFFICE USE": ["Receipt Date"],
    "PART 1 - PROPOSER": [
        "Name",
        "Date of Birth",
        "Aadhaar Card No",
        "PAN Card / Unique Identification No",
        "Proposer Permanent Residential Address including Pincode"
    ]
}

blueprint_text = st.sidebar.text_area(
    "Modify Sections & Fields:",
    value=json.dumps(default_blueprint, indent=4),
    height=250
)

try:
    SECTION_STRUCTURE = json.loads(blueprint_text)
except Exception:
    SECTION_STRUCTURE = default_blueprint

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

def extract_ocr_text(img):
    if not OCR_AVAILABLE:
        return ""
    try:
        return "\\n".join(reader.readtext(np.array(img), detail=0, paragraph=True))
    except Exception:
        return ""

def convert_to_images(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    images = []

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".tiff"]:
        images.append(Image.open(uploaded_file).convert("RGB"))

    elif ext == ".pdf":
        pdf = pdfium.PdfDocument(uploaded_file.read())
        for i in range(len(pdf)):
            page = pdf[i]
            bitmap = page.render(scale=5)
            images.append(bitmap.to_pil().convert("RGB"))

    elif ext == ".docx":
        doc = Document(uploaded_file)
        text = "\\n".join([p.text for p in doc.paragraphs])
        from PIL import ImageDraw
        img = Image.new("RGB", (1000, 1400), "white")
        ImageDraw.Draw(img).text((20,20), text[:3000], fill="black")
        images.append(img)

    return images

def merge_results(master, new):
    for k, v in new.items():
        if isinstance(v, dict):
            if k not in master:
                master[k] = {}
            merge_results(master[k], v)
        elif v not in [None, "", "null"]:
            master[k] = v

def extract_hierarchical_data(img, structure):
    schema = {
        section: {field: "value or null" for field in fields}
        for section, fields in structure.items()
    }

    ocr_text = extract_ocr_text(img)

    prompt = f"""
Extract fields from this insurance proposal form.

Rules:
- Copy values exactly as written.
- Aadhaar must be 12 digits.
- PAN must be 10 characters.
- Return null if unreadable.
- Return ONLY valid JSON.

OCR TEXT:
{ocr_text}

Schema:
{json.dumps(schema, indent=2)}
"""

    img.thumbnail((1800, 1800))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    img_str = base64.b64encode(buf.getvalue()).decode()

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": [
                {"type":"text","text":prompt},
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{img_str}"}}
            ]
        }],
        temperature=0
    )

    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"RAW": content}

st.title("🔐 Document Extractor")

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf","docx","png","jpg","jpeg","webp"],
    accept_multiple_files=True
)

if uploaded_files and st.button("Extract", disabled=not api_key_available):

    results = []

    for file in uploaded_files:

        pages = convert_to_images(file)
        merged = {}

        for page in pages:
            page_result = extract_hierarchical_data(page, SECTION_STRUCTURE)
            if isinstance(page_result, dict):
                merge_results(merged, page_result)

        row = {"File Name": file.name}

        for section, fields in SECTION_STRUCTURE.items():
            for field in fields:
                row[field] = merged.get(section, {}).get(field, "Not Found")

        results.append(row)

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode(),
        "extracted_data.csv",
        "text/csv"
    )
