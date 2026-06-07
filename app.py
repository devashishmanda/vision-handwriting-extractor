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

# Require API Key to run securely in the cloud
if "GROQ_API_KEY" not in st.secrets:
    st.error("⚠️ Missing GROQ_API_KEY. Please add it to your Streamlit Advanced Settings -> Secrets.")
    st.stop()

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

# ==============================================================================
# 🛠️ PROCESSING PIPELINE ENGINE
# ==============================================================================
def enhance_image_for_ai(img):
    """Converts to grayscale and boosts contrast to simulate a clean 2D scan."""
    # 1. Convert to Grayscale to remove color noise
    img = img.convert('L')
    
    # 2. Boost Contrast significantly
    enhancer_contrast = ImageEnhance.Contrast(img)
    img = enhancer_contrast.enhance(2.0)
    
    # 3. Light Sharpness (Dialed down so box lines don't look like text)
    enhancer_sharpness = ImageEnhance.Sharpness(img)
    img = enhancer_sharpness.enhance(1.2)
    
    # API requires RGB format
    return img.convert('RGB')

def convert_to_images(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    images = []
    try:
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.tiff']:
            raw_img = Image.open(uploaded_file).convert('RGB')
            images.append(enhance_image_for_ai(raw_img))
            
        elif ext == '.pdf':
            pdf = pdfium.PdfDocument(uploaded_file.read())
            page = pdf[0]
            bitmap = page.render(scale=3)
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
            # Forcefully extract ONLY digits, ignoring any letters/symbols the AI added
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
        if "aadhaar" in k.lower():
            if not val_str.isdigit() or len(val_str) != 12:
                issues.append(f"Aadhaar length error ({len(val_str)} digits)")
                
    return "🟢 Validated" if not issues else f"⚠️ Review: {', '.join(issues)}"

def find_best_match(target_field, ai_response_dict):
    """Aggressively hunts for the target field anywhere in the AI's response."""
    if isinstance(ai_response_dict, dict):
        for k, v in ai_response_dict.items():
            if target_field.lower() in k.lower():
                if not isinstance(v, (dict, list)):
                    return str(v)
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

    # Tuned Prompt utilizing Visual Anchoring and a Two-Pass self-verification step
    prompt = (
        f"You are an expert forensic document parser running a zero-error data extraction task.\n\n"
        f"STEP 1: VISUAL SCAN ANCHORS\n"
        f"- Locate '8. Aadhaar Card No.:'. Focus entirely on the sequence of individual printed boxes to the right of this text.\n"
        f"- Locate '7. Unique Identification No.:' or 'PAN:'. Focus entirely on the alphanumeric sequence inside the boxes.\n\n"
        f"STEP 2: RUTHLESS TRANSCRIPTION RULES\n"
        f"- DO NOT assume or correct spelling. Write exactly what is written (e.g., if it reads 'PULICE', output 'PULICE').\n"
        f"- BOX BOUNDARY GUARD: The printed vertical lines separating the boxes are NOT characters. Do not mistake a vertical box border for the number '1', number '6', or letter 'I'.\n"
        f"- AADHAAR DOUBLE-COUNT: The Aadhaar field must contain EXACTLY 12 digits. Count them from left to right. Then count them from right to left. If your count equals 13, you have mistakenly captured a vertical box border line at the beginning or end. Drop that artifact line.\n"
        f"- PAN CHARACTER RECOGNITION: Pay meticulous attention to box strokes. Differentiate clearly between 'W' vs 'U', 'L' vs 'C', and '0' vs 'D'.\n\n"
        f"STEP 3: EXECUTION\n"
        f"Verify your character counts internally, then populate this exact JSON structural hierarchy strictly:\n"
        f"{json.dumps(schema_instruction, indent=2)}\n\n"
        f"OUTPUT REQUIREMENT:\n"
        f"Return ONLY the raw JSON object. Do not wrap it in markdown code blocks, do not include introductory text, and do not append a conversational sign-off."
    )

    # Maintain crisp resolution for small box detection
    img.thumbnail((1800, 1800)) 
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=98)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=1024
        )
        
        raw_output = response.choices[0].message.content
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        
        if json_match:
            return json.loads(json_match.group(0))
        else:
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
            
            if not pages: 
                continue
            
            nested_json = extract_hierarchical_data(pages[0], SECTION_STRUCTURE)
            
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
            
            with right_database:
                st.markdown("### 📊 Extracted Records Database")
                st.dataframe(flat_results_for_csv)
                
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        status_update.success(f"🎉 Processed all documents in {time.time() - start_time:.2f} seconds!")

        st.markdown("---")
        df_export = pd.DataFrame(flat_results_for_csv).reindex(columns=ALL_FLAT_COLUMNS, fill_value="Not Found")
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Extracted Data as CSV",
            data=csv_bytes,
            file_name="extracted_claims_data.csv",
            mime="text/csv",
        )
