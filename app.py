import streamlit as st
import os
import json
import time
import re
import base64
from io import BytesIO
from PIL import Image, ImageEnhance
import pypdfium2 as pdfium
from docx import Document
import pandas as pd
from groq import Groq
import easyocr
import numpy as np

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

def extract_ocr_text(img):

    results = reader.readtext(
        np.array(img),
        detail=0,
        paragraph=True
    )

    return "\n".join(results)

# ==============================================================================
# 🎨 BRAND THEME LAYOUT (CSS Injection)
# ==============================================================================
extractor_theme_css = """
<style>
.stApp { background-color: #FAFAFA; color: #2F3E46; }
h1, h2, h3, .stSubheader { color: #005F73 !important; font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 600; }
div.stButton > button:first-child { background-color: #005F73 !important; color: white !important; border-radius: 4px; border: none; font-weight: bold; padding: 0.5rem 2rem; transition: all 0.3s ease; }
div.stButton > button:first-child:hover { background-color: #0A9396 !important; box-shadow: 0px 4px 10px rgba(0, 95, 115, 0.2); }
section[data-testid="stSidebar"] { background-color: #E0F2F1 !important; border-right: 1px solid #B2DFDB; }
div[data-testid="stMetricValue"] { color: #005F73 !important; }
.status-box { padding: 15px; border-radius: 5px; background-color: white; border-left: 5px solid #005F73; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
"""

st.set_page_config(page_title="Text Extractor Hub", page_icon="🔐", layout="wide")
st.markdown(extractor_theme_css, unsafe_allow_html=True)

# Safely check for API key (Fixes the disappearing upload button)
api_key_available = "GROQ_API_KEY" in st.secrets

# ==============================================================================
# ⚙️ SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.markdown("<h2 style='margin-top:0;'>⚙️ Section Configurator</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Define your sections and parameters below using clean JSON format:")

default_blueprint = {
    "FOR OFFICE USE": ["Receipt Date"],
    "PART 1 - PROPOSER": [
        "Name", "Date of Birth", "Aadhaar Card No", 
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
    if not isinstance(SECTION_STRUCTURE, dict):
        SECTION_STRUCTURE = default_blueprint
except Exception:
    SECTION_STRUCTURE = default_blueprint

MODEL_SELECTION = st.sidebar.selectbox("Choose AI Vision Engine:", ["Llama 4 Scout Vision"])
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

ALL_FLAT_COLUMNS = ["File Name", "Validation Report"]
for section, items in SECTION_STRUCTURE.items():
    if isinstance(items, list):
        ALL_FLAT_COLUMNS.extend(items)

st.markdown("<h1>🔐 Document Text Extractor Intelligence Hub</h1>", unsafe_allow_html=True)
st.caption("Securely parse unstructured forms into custom databases using Llama 4 Vision on Groq.")

if not api_key_available:
    st.warning("⚠️ Missing GROQ_API_KEY. Please add it to your Streamlit Advanced Settings -> Secrets to unlock processing.")

# ==============================================================================
# 🛠️ PROCESSING PIPELINE ENGINE
# ==============================================================================
def enhance_image_for_ai(img):
    """Boosts contrast and heavily increases sharpness in full color."""
    # Ensure image is in RGB format (Grayscale conversion removed)
    img = img.convert('RGB')
    return img

def convert_to_images(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    images = []
    try:
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.tiff']:
            raw_img = Image.open(uploaded_file).convert('RGB')
            images.append(enhance_image_for_ai(raw_img))
        elif ext == '.pdf':
            pdf = pdfium.PdfDocument(uploaded_file.read())
            for i in range(len(pdf)):
    page = pdf[i]

    bitmap = page.render(scale=5)

    raw_img = bitmap.to_pil().convert("RGB")

    images.append(
        enhance_image_for_ai(raw_img)
    )
            bitmap = page.render(scale=5)
            raw_img = bitmap.to_pil().convert('RGB')
            images.append(enhance_image_for_ai(raw_img))
        elif ext == '.docx':
            doc = Document(uploaded_file)
            full_text = "\n".join([para.text for para in doc.paragraphs])
            from PIL import ImageDraw
            img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
            ImageDraw.Draw(img).text((20, 20), full_text[:2000], fill=(0, 0, 0))
            images.append(img)
    except Exception as e:
        st.error(f"Error converting file {uploaded_file.name}: {e}")
    return images

def normalize_extracted_values(row_data):
    for key, value in list(row_data.items()):
        if str(value).lower() in ["not found", "null", "none"] or not value: 
            continue
        val_str = str(value).strip()
        if "aadhaar" in key.lower():
            clean_digits = ''.join(filter(str.isdigit, val_str))
            row_data[key] = clean_digits if clean_digits else "Not Found"
        elif "pan" in key.lower() or "identification" in key.lower():
            row_data[key] = re.sub(r'[\s\-]', '', val_str).upper()
        elif "date" in key.lower() or "receipt" in key.lower():
            clean_date = re.sub(r'[\s\.\-\/]', '', val_str)
            if len(clean_date) == 8 and clean_date.isdigit():
                row_data[key] = f"{clean_date[:2]}/{clean_date[2:4]}/{clean_date[4:]}"
    return row_data

def run_regex_validation(extracted_row):
    issues = []
    for k, v in extracted_row.items():
        if str(v).lower() in ["not found", "null", "none"] or not v:
            continue
            
        val_str = re.sub(r'[\s\-]', '', str(v))
        
        # 1. Aadhaar Check (Exactly 12 Digits)
        if "aadhaar" in k.lower():
            if not val_str.isdigit() or len(val_str) != 12:
                issues.append(f"Aadhaar error ({len(val_str)} digits)")
                
        # 2. PAN Card Check (5 Letters, 4 Digits, 1 Letter)
        elif "pan" in k.lower() or "identification" in k.lower():
            if len(val_str) != 10:
                issues.append(f"PAN length error ({len(val_str)} chars)")
            elif not re.match(r'^[A-Z]{5}[0-9]{4}[A-Z]$', val_str.upper()):
                issues.append("PAN format error")
                
    return "🟢 Validated" if not issues else f"⚠️ Review: {', '.join(issues)}"

from difflib import SequenceMatcher

def find_best_match(
    target_field,
    ai_response_dict
):

    best_score = 0
    best_value = None

    def search(node):

        nonlocal best_score
        nonlocal best_value

        if isinstance(node,dict):

            for k,v in node.items():

                score = SequenceMatcher(
                    None,
                    target_field.lower(),
                    k.lower()
                ).ratio()

                if (
                    score > best_score
                    and not isinstance(
                        v,
                        (dict,list)
                    )
                ):
                    best_score = score
                    best_value = v

                search(v)

        elif isinstance(node,list):

            for item in node:
                search(item)

    search(ai_response_dict)

    if best_score > 0.75:
        return best_value

    return None
        for v in ai_response_dict.values():
            result = find_best_match(target_field, v)
            if result:
                return result
    elif isinstance(ai_response_dict, list):
        for item in ai_response_dict:
            result = find_best_match(target_field, item)
            if result:
                return result
    return None

def extract_hierarchical_data(img, structure):
    schema_instruction = {}
    for section, parameters in structure.items():
        if isinstance(parameters, list):
            schema_instruction[section] = {param: "Extracted value or null" for param in parameters}

ocr_text = extract_ocr_text(img)
    
    prompt = f"""
Analyze this proposal form.

Extract the requested fields exactly as written.

Rules:
- Aadhaar Card No must contain exactly 12 digits.
- PAN Card No must contain exactly 10 characters.
- Do not guess values.
- Preserve spelling exactly as written.
- If a field cannot be read, return null.

Return ONLY valid JSON.

OCR TEXT:

{ocr_text}

Use OCR text when possible.
Use image only for unclear fields.

Schema:

{json.dumps(schema_instruction, indent=2)}
"""

    img.thumbnail((1800, 1800)) 
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=98)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    client = Groq(api_key=st.secrets["GROQ_API_KEY"] if api_key_available else "")

    try:
        response = client.chat.completions.create(
    model=MODEL_NAME,
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {
                     "url": f"data:image/jpeg;base64,{img_str}"
                 }}
            ]
        }
    ],
    temperature=0,
    max_tokens=1024
)
        
        return json.loads(
    response.choices[0].message.content
)
        if json_match:
            return json.loads(json_match.group(0))
        return {"Error": "AI did not return JSON format", "RawResponse": raw_output}
    except Exception as e:
        return {"Error": str(e)}

# ==============================================================================
# 📥 FILE UPLOADER & WORKSPACE
# ==============================================================================
uploaded_files = st.file_uploader(
    "Upload claim documents to parse (PDF, DOCX, Images):", 
    type=["pdf", "docx", "png", "jpg", "jpeg", "webp"], 
    accept_multiple_files=True
)

# Initialize a session bucket so data doesn't vanish when we edit the table
if "saved_extracted_data" not in st.session_state:
    st.session_state.saved_extracted_data = []

if uploaded_files:
    # Disable extraction if API key is missing, but don't hide the uploader
    if st.button("🚀 Start Dynamic Section Extraction", disabled=not api_key_available):
        flat_results_for_csv = []
        progress_bar = st.progress(0)
        status_update = st.empty()
        
        left_preview, right_database = st.columns(2)
        start_time = time.time()

        for idx, file in enumerate(uploaded_files):
            status_update.markdown(f"<div class='status-box'>⏳ <b>Processing [{idx+1}/{len(uploaded_files)}]:</b> {file.name}</div>", unsafe_allow_html=True)
            pages = convert_to_images(file)
            if not pages: 
                continue


            def merge_results(master,new):

    for key,val in new.items():

        if isinstance(val,dict):

            if key not in master:
                master[key]={}

            merge_results(
                master[key],
                val
            )

        else:

            if (
                val
                and str(val).lower()
                not in ["null","none",""]
            ):
                master[key]=val
            
            nested_json = {}

for page in pages:

    page_result = extract_hierarchical_data(
        page,
        SECTION_STRUCTURE
    )

    merge_results(
        nested_json,
        page_result
    )
            
            with left_preview:
                st.markdown(f"### 📄 Visual Preview: {file.name}")
                st.image(pages[0], use_container_width=True)
                with st.expander("🛠️ Debug: See Raw AI Output"):
                    st.json(nested_json)
                
            row = {"File Name": file.name}
            for section, parameters in SECTION_STRUCTURE.items():
                if isinstance(parameters, list):
                    for field in parameters:
                        val = nested_json.get(section, {}).get(field)
                        if val is None or str(val).lower() in ["null", "none"]:
                            val = find_best_match(field, nested_json)
                        row[field] = val if val and str(val).strip() != "" else "Not Found"
                        
            row = normalize_extracted_values(row)
            row["Validation Report"] = run_regex_validation(row)
            flat_results_for_csv.append(row)
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        # Save results to session state so we can edit them below
        st.session_state.saved_extracted_data = flat_results_for_csv
        status_update.success(f"🎉 Processed all documents in {time.time() - start_time:.2f} seconds!")

# ==============================================================================
# 📊 EDITABLE SPREADSHEET & EXPORT
# ==============================================================================
if st.session_state.saved_extracted_data:
    st.markdown("---")
    st.markdown("### 📊 Extracted Records Database (Editable)")
    st.caption("Double-click any cell to manually fix AI mistakes before downloading!")
    
    # Render the interactive editor
    edited_data = st.data_editor(st.session_state.saved_extracted_data, use_container_width=True)
    
    # Update validation report dynamically in case the user fixes a field
    for row in edited_data:
        row["Validation Report"] = run_regex_validation(row)
        
    df_export = pd.DataFrame(edited_data).reindex(columns=ALL_FLAT_COLUMNS, fill_value="Not Found")
    csv_bytes = df_export.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Extracted Data as CSV",
        data=csv_bytes,
        file_name="extracted_claims_data.csv",
        mime="text/csv",
    )
