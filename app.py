import streamlit as st
import os
import json
import time
import re
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from docx import Document
import ollama
import pandas as pd

# ==============================================================================
# 🎨 BRAND THEME LAYOUT (CSS Injection)
# ==============================================================================
extractor_theme_css = """
<style>
    .stApp { background-color: #FAFAFA; color: #2F3E46; }
    h1, h2, h3, .stSubheader { color: #005F73 !important; font-family: 'Helvetica Neue', Arial, sans-serif; font-weight: 600; }
    div.stButton > button:first-child {
        background-color: #005F73 !important; color: white !important; border-radius: 4px; border: none; font-weight: bold; padding: 0.5rem 2rem; transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover { background-color: #0A9396 !important; box-shadow: 0px 4px 10px rgba(0, 95, 115, 0.2); }
    section[data-testid="stSidebar"] { background-color: #E0F2F1 !important; border-right: 1px solid #B2DFDB; }
    div[data-testid="stMetricValue"] { color: #005F73 !important; }
    .status-box { padding: 15px; border-radius: 5px; background-color: white; border-left: 5px solid #005F73; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
</style>
"""

st.set_page_config(page_title="Text Extractor Hub", page_icon="🔒", layout="wide")
st.markdown(extractor_theme_css, unsafe_allow_html=True)

# ==============================================================================
# ⚙️ SIDEBAR CONFIGURATION
# ==============================================================================
st.sidebar.markdown("<h2 style='margin-top:0;'>⚙️ Section Configurator</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Define your sections and parameters below using clean JSON format:")

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
    if not isinstance(SECTION_STRUCTURE, dict):
        SECTION_STRUCTURE = default_blueprint
except Exception:
    SECTION_STRUCTURE = default_blueprint

MODEL_SELECTION = st.sidebar.selectbox("Choose AI Vision Engine:", ["Qwen 2.5 VL (Optimized)", "Llama 3.2 Vision"])
if MODEL_SELECTION == "Qwen 2.5 VL (Optimized)":
    MODEL_NAME = "qwen2.5vl"
else:
    MODEL_NAME = "llama3.2-vision"

ALL_FLAT_COLUMNS = ["File Name", "Validation Report"]
for section, items in SECTION_STRUCTURE.items():
    if isinstance(items, list):
        ALL_FLAT_COLUMNS.extend(items)

st.markdown("<h1>🔒 Document Text Extractor Intelligence Hub</h1>", unsafe_allow_html=True)
st.caption("Securely parse unstructured forms into custom segmented databases completely offline.")

# ==============================================================================
# 🛠️ PROCESSING PIPELINE ENGINE
# ==============================================================================
def convert_to_images(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    images = []
    try:
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.tiff']:
            images.append(Image.open(uploaded_file).convert('RGB'))
        elif ext == '.pdf':
            pdf = pdfium.PdfDocument(uploaded_file.read())
            page = pdf[0]
            bitmap = page.render(scale=3)  
            images.append(bitmap.to_pil().convert('RGB'))
        elif ext == '.docx':
            doc = Document(uploaded_file)
            full_text = "\n".join([para.text for para in doc.paragraphs])
            from PIL import ImageDraw
            img = Image.new('RGB', (800, 1000), color=(255, 255, 255))
            ImageDraw.Draw(img).text((20, 20), full_text[:2000], fill=(0, 0, 0))
            images.append(img)
    except Exception as e:
        st.error(f"Error converting file: {e}")
    return images

def normalize_extracted_values(row_data):
    for key, value in list(row_data.items()):
        if value == "Not Found" or not value:
            continue
        val_str = str(value).strip()
        if "aadhaar" in key.lower():
            clean_digits = re.sub(r'[\s\.\-]', '', val_str)
            clean_digits = clean_digits.replace('O', '0').replace('o', '0')
            row_data[key] = clean_digits
        elif "pan" in key.lower() or "identification" in key.lower():
            clean_pan = re.sub(r'[\s\-]', '', val_str).upper()
            row_data[key] = clean_pan
        elif "date" in key.lower() or "receipt" in key.lower():
            clean_date = re.sub(r'[\s\.\-\/]', '', val_str)
            if len(clean_date) == 8 and clean_date.isdigit():
                row_data[key] = f"{clean_date[:2]}/{clean_date[2:4]}/{clean_date[4:]}"
    return row_data

def run_regex_validation(extracted_row):
    issues = []
    for k, v in extracted_row.items():
        val_str = re.sub(r'[\s\-]', '', str(v))
        if "aadhaar" in k.lower() and val_str != "not found":
            if not val_str.isdigit() or len(val_str) != 12:
                issues.append(f"Aadhaar length error ({len(val_str)} digits)")
    if len(issues) == 0:
        return "🟢 Validated"
    else:
        return f"⚠️ Review: {', '.join(issues)}"

def extract_hierarchical_data(img, structure):
    schema_instruction = {}
    for section, parameters in structure.items():
        if isinstance(parameters, list):
            schema_instruction[section] = {param: "Extracted value or null" for param in parameters}

    prompt = (
        f"Analyze this document canvas. Extract handwriting text fields and group them into sections.\n"
        f"CRITICAL STRUCTURAL INSTRUCTIONS:\n"
        f"- Look carefully at boxed digits. Aadhaar Card No MUST contain exactly 12 numerical digits.\n"
        f"- PAN Card Numbers follow an exact 10-character alphanumeric sequence (5 letters, 4 digits, 1 letter).\n"
        f"- Transcribe character-by-character from the form fields.\n\n"
        f"You must strictly output a valid JSON object matching this exact structural hierarchy:\n"
        f"{json.dumps(schema_instruction, indent=2)}\n"
        f"Do not include code blocks, markdown wrappers, or conversational text. Return raw JSON string only."
    )
    
    img.thumbnail((1600, 1600))
    temp_path = "_dynamic_section_temp.jpg"
    img.save(temp_path, "JPEG", quality=95)
    
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            format='json',
            messages=[{'role': 'user', 'content': prompt, 'images': [temp_path]}],
            options={'temperature': 0.0, 'num_predict': 1024}
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return json.loads(response['message']['content'])
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return {}

# ==============================================================================
# 📥 FILE UPLOADER & WORKSPACE
# ==============================================================================
uploaded_files = st.file_uploader(
    "Upload claim documents to parse (PDF, DOCX, Images):", 
    type=["pdf", "docx", "png", "jpg", "jpeg", "webp"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.button("🚀 Start Dynamic Section Extraction"):
        flat_results_for_csv = []
        progress_bar = st.progress(0)
        status_update = st.empty()
        
        left_preview, right_database = st.columns(2)
        start_time = time.time()
        
        for idx, file in enumerate(uploaded_files):
            status_update.markdown(f"<div class='status-box'>⏳ <b>Processing [{idx+1}/{len(uploaded_files)}]:</b> {file.name}</div>", unsafe_allow_html=True)
            
            pages = convert_to_images(file)
            if len(pages) == 0:
                continue
            
            nested_json = extract_hierarchical_data(pages[0], SECTION_STRUCTURE)
            
            with left_preview:
                st.markdown(f"### 📄 Visual Preview: {file.name}")
                st.image(pages[0], use_container_width=True)
            
            row = {"File Name": file.name}
            for section, parameters in SECTION_STRUCTURE.items():
                if isinstance(parameters, list):
                    section_data = nested_json.get(section, {})
                    for field in parameters:
                        val = section_data.get(field, "Not Found")
                        if val == "Not Found":
                            for raw_k, raw_v in section_data.items():
                                if field.lower() in raw_k.lower() or raw_k.lower() in field.lower():
                                    val = raw_v
                        row[field] = val
            
            row = normalize_extracted_values(row)
            row["Validation Report"] = run_regex_validation(row)
            flat_results_for_csv.append(row)
            
            with right_database:
                st.markdown("### 📊 Extracted Records Database")
                st.dataframe(flat_results_for_csv)
                
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_update.success(f"🎉 Processed all documents completely offline in {time.time() - start_time:.2f} seconds!")
        
        # Completely Flat Export Logic (Zero conditional indent blocks)
        df_export = pd.DataFrame(flat_results_for_csv).reindex(columns=ALL_FLAT_COLUMNS, fill_value="Not Found")
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.sidebar.markdown("---")
