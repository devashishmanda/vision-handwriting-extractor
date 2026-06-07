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
    img = enhancer_
