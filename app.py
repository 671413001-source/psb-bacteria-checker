"""
PSB Bottle Inspector & Classifier (แอปพลิเคชันตรวจสอบและจำแนกขวดจุลินทรีย์สังเคราะห์แสง)
-----------------------------------------------------------------------------------------
A modern, clean Streamlit web application to inspect and classify Photosynthetic Bacteria (PSB)
liquid bottles based on uploaded images using OpenCV / PIL HSV color space analysis and
integrated with ThaiLLM (Pathumma 3.0) for expert agricultural recommendations.
"""

import os
import re
import json
import time
import requests
import numpy as np
from PIL import Image
import streamlit as st

# fpdf2 import with fallback safety
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# OpenCV import with fallback safety
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


# ==============================================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM STYLES
# ==============================================================================
st.set_page_config(
    page_title="PSB Bottle Inspector | ตรวจสอบจุลินทรีย์สังเคราะห์แสง",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS matching the warm bio-agricultural theme
st.markdown("""
<style>
    /* Main container refinements */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    
    /* Header card */
    .header-banner {
        background: linear-gradient(135deg, #FFF5EB 0%, #FEE8D6 100%);
        border: 1px solid #FCD2B3;
        border-radius: 14px;
        padding: 22px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(232, 89, 12, 0.06);
    }
    .header-title {
        color: #C04205;
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        color: #614B3D;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    
    /* Metric / Status badges */
    .status-badge {
        display: inline-block;
        padding: 8px 18px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .badge-ready {
        background-color: #D3F9D8;
        color: #2B8A3E;
        border: 2px solid #69DB7C;
    }
    .badge-not-ready {
        background-color: #FFF3BF;
        color: #D9480F;
        border: 2px solid #FFD43B;
    }
    
    /* Metric Card */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E9ECEF;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .metric-title {
        font-size: 13px;
        color: #868E96;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #212529;
    }
    .metric-sub {
        font-size: 12px;
        color: #ADB5BD;
        margin-top: 4px;
    }

    /* Color Swatch Pill */
    .color-swatch-box {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 10px;
        padding: 8px 12px;
    }
    .swatch-dot {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        border: 2px solid #FFFFFF;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Advice box */
    .advice-box {
        background-color: #F8F9FA;
        border-left: 5px solid #E8590C;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 15px;
    }
    
    /* Thinking block */
    .thinking-box {
        background-color: #F1F3F5;
        border-left: 4px solid #868E96;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 13px;
        color: #495057;
        font-family: monospace;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. COMPUTER VISION & HSV ANALYSIS FUNCTIONS
# ==============================================================================
def rgb_to_hsv_numpy(rgb_img: np.ndarray) -> np.ndarray:
    rgb = rgb_img.astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c = np.max(rgb, axis=2)
    min_c = np.min(rgb, axis=2)
    diff = max_c - min_c

    h = np.zeros_like(max_c)
    mask = diff > 1e-6

    r_mask = mask & (max_c == r)
    h[r_mask] = (60.0 * ((g[r_mask] - b[r_mask]) / diff[r_mask]) + 360.0) % 360.0

    g_mask = mask & (max_c == g)
    h[g_mask] = 60.0 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120.0

    b_mask = mask & (max_c == b)
    h[b_mask] = 60.0 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240.0

    h_cv = (h / 2.0).astype(np.uint8)

    s = np.zeros_like(max_c)
    s[max_c > 1e-6] = (diff[max_c > 1e-6] / max_c[max_c > 1e-6]) * 255.0
    s_cv = np.clip(s, 0, 255).astype(np.uint8)

    v_cv = np.clip(max_c * 255.0, 0, 255).astype(np.uint8)

    return np.dstack((h_cv, s_cv, v_cv))


def analyze_psb_bottle(
    pil_image: Image.Image,
    red_threshold_pct: float = 25.0,
    roi_mode: str = "liquid_center",
    custom_roi: tuple = None,
    hue_tolerance: int = 14,
    min_saturation: int = 45,
    min_value: int = 30
) -> dict:
    rgb_full = np.array(pil_image.convert("RGB"))
    height, width = rgb_full.shape[:2]

    if roi_mode == "custom" and custom_roi is not None:
        x1 = int(np.clip(custom_roi[0], 0, 1) * width)
        y1 = int(np.clip(custom_roi[1], 0, 1) * height)
        x2 = int(np.clip(custom_roi[2], 0, 1) * width)
        y2 = int(np.clip(custom_roi[3], 0, 1) * height)
        if x2 <= x1:
            x2 = min(width, x1 + 10)
        if y2 <= y1:
            y2 = min(height, y1 + 10)
    elif roi_mode == "liquid_center":
        y1, y2 = int(height * 0.20), int(height * 0.80)
        x1, x2 = int(width * 0.20), int(width * 0.80)
    else:
        y1, y2 = 0, height
        x1, x2 = 0, width

    rgb_roi = rgb_full[y1:y2, x1:x2]

    if OPENCV_AVAILABLE:
        hsv_roi = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2HSV)
    else:
        hsv_roi = rgb_to_hsv_numpy(rgb_roi)

    h_channel = hsv_roi[:, :, 0]
    s_channel = hsv_roi[:, :, 1]
    v_channel = hsv_roi[:, :, 2]

    lower_red_mask = (h_channel <= hue_tolerance) & (s_channel >= min_saturation) & (v_channel >= min_value)
    upper_red_mask = (h_channel >= 150) & (s_channel >= min_saturation) & (v_channel >= min_value)
    red_mask = lower_red_mask | upper_red_mask

    roi_pixel_count = rgb_roi.shape[0] * rgb_roi.shape[1]
    red_pixel_count = int(np.count_nonzero(red_mask))
    red_ratio_pct = (red_pixel_count / roi_pixel_count) * 100.0 if roi_pixel_count > 0 else 0.0

    mean_rgb = np.mean(rgb_roi, axis=(0, 1)).astype(int)
    hex_color = f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"
    mean_hsv = np.mean(hsv_roi, axis=(0, 1)).astype(int)

    is_ready = red_ratio_pct >= red_threshold_pct

    if red_ratio_pct < 5.0:
        stage_name_th = "ระยะเริ่มต้น (น้ำ + หัวเชื้อ)"
        stage_name_en = "Stage 0: Starter / Inoculation"
        stage_desc_th = "น้ำขุ่นขาว/ใส เม็ดสีสังเคราะห์แสงยังไม่เพิ่มจำนวน"
        est_days_left = "21 - 28 วัน"
    elif red_ratio_pct < red_threshold_pct * 0.6:
        stage_name_th = "ระยะเริ่มมีสี (ระยะแรก)"
        stage_name_en = "Stage 1: Early Fermentation"
        stage_desc_th = "เริ่มมีสีชมพูอ่อน ส้มจาง หรือเขียวอมน้ำตาล แบคทีเรียเริ่มสังเคราะห์แสง"
        est_days_left = "14 - 21 วัน"
    elif not is_ready:
        stage_name_th = "ระยะสีเข้มขึ้น (ระยะกลาง)"
        stage_name_en = "Stage 2: Mid Fermentation"
        stage_desc_th = "สีแดงส้มหรือแดงกระจายตัว เซลล์กำลังขยายตัวอย่างรวดเร็ว"
        est_days_left = "7 - 10 วัน"
    else:
        stage_name_th = "ระยะสมบูรณ์ (พร้อมใช้งาน)"
        stage_name_en = "Stage 3: Mature & Ready to Use"
        stage_desc_th = "สีแดงเข้ม ไวน์แดง หรือแดงเลือดนก เม็ดสีและสารต้านอนุมูลอิสระสูงสุด"
        est_days_left = "0 วัน (พร้อมนำไปใช้ทันที)"

    overlay_roi = rgb_roi.copy()
    overlay_roi[red_mask] = (
        0.35 * overlay_roi[red_mask] + 0.65 * np.array([30, 230, 80], dtype=np.float32)
    ).astype(np.uint8)

    full_preview = rgb_full.copy()
    if roi_mode != "full":
        border_thickness = max(2, int(min(height, width) * 0.005))
        full_preview[max(0, y1-border_thickness):y1+border_thickness, x1:x2] = [232, 89, 12]
        full_preview[max(0, y2-border_thickness):y2+border_thickness, x1:x2] = [232, 89, 12]
        full_preview[y1:y2, max(0, x1-border_thickness):x1+border_thickness] = [232, 89, 12]
        full_preview[y1:y2, max(0, x2-border_thickness):x2+border_thickness] = [232, 89, 12]

    mask_visual = (red_mask.astype(np.uint8) * 255)

    return {
        "is_ready": is_ready,
        "status_label_th": "พร้อมใช้งาน (Ready to Use)" if is_ready else "ยังไม่พร้อมใช้ / อยู่ระหว่างการหมัก (Not Ready)",
        "status_label_en": "Ready to Use" if is_ready else "Not Ready / Still Fermenting",
        "red_percentage": round(red_ratio_pct, 2),
        "red_threshold": red_threshold_pct,
        "stage_th": stage_name_th,
        "stage_en": stage_name_en,
        "stage_desc": stage_desc_th,
        "days_left": est_days_left,
        "dominant_hex": hex_color,
        "mean_rgb": mean_rgb.tolist(),
        "mean_hsv": mean_hsv.tolist(),
        "roi_box": (x1, y1, x2, y2),
        "overlay_roi": Image.fromarray(overlay_roi),
        "mask_image": Image.fromarray(mask_visual),
        "full_preview": Image.fromarray(full_preview)
    }


# ==============================================================================
# 3. PDF REPORT GENERATOR
# ==============================================================================
def generate_pdf_report(report_data: dict) -> bytes:
    """Generates a structured PDF report using fpdf2 with safe Latin-1 text."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "PSB Inspection & Quality Report", new_x="LMARGIN", new_y="NEXT", align="C")
    
    # Meta Info
    pdf.set_font("Helvetica", size=9)
    pdf.cell(0, 6, f"Report ID: {report_data['report_id']}  |  Timestamp: {report_data['timestamp']}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, f"Analyzed File: {report_data['file_name']}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)
    
    # Divider
    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    # Classification Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "1. Inspection Summary", new_x="LMARGIN", new_y="NEXT")
    
    c = report_data["classification"]
    status_text = "READY TO USE" if c["is_ready"] else "NOT READY (FERMENTING)"
    
    pdf.set_font("Helvetica", size=10)
    pdf.cell(60, 7, "Classification Result:")
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"{status_text} ({c['status_en']})", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", size=10)
    pdf.cell(60, 7, "Red Pigment Density:")
    pdf.cell(0, 7, f"{c['red_pigment_percentage']}% (Threshold: {c['threshold_applied']}%)", new_x="LMARGIN", new_y="NEXT")
    
    # ใช้ stage_en แทน stage_th เพื่อป้องกันปัญหา Font ภาษาไทย
    stage_english = c.get("stage_en", "Fermentation Stage")
    pdf.cell(60, 7, "Fermentation Stage:")
    pdf.cell(0, 7, f"{stage_english}", new_x="LMARGIN", new_y="NEXT")
    
    # แปลงข้อความวันให้เป็นภาษาอังกฤษ (ตัดคำว่า วัน ออก)
    days_clean = str(c['estimated_days_left']).replace("วัน", "days").replace("พร้อมนำไปใช้ทันที", "Ready now")
    pdf.cell(60, 7, "Estimated Days Remaining:")
    pdf.cell(0, 7, f"{days_clean}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(5)
    
    # Color Profile Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "2. Color & Pigment Profile", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    
    col = report_data["color_metrics"]
    pdf.cell(60, 7, "Dominant Color (HEX):")
    pdf.cell(0, 7, f"{col['hex']}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(60, 7, "Average RGB Values:")
    pdf.cell(0, 7, f"R: {col['rgb'][0]} | G: {col['rgb'][1]} | B: {col['rgb'][2]}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.cell(60, 7, "Average HSV Values:")
    pdf.cell(0, 7, f"H: {col['hsv'][0]} | S: {col['hsv'][1]} | V: {col['hsv'][2]}", new_x="LMARGIN", new_y="NEXT")
    
    pdf.ln(10)
    pdf.set_draw_color(220, 220, 220)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(0, 6, "Generated automatically by PSB Bottle Inspector & Quality Classifier", new_x="LMARGIN", new_y="NEXT", align="C")
    
    return bytes(pdf.output())


# ==============================================================================
# 4. THAILLM API INTEGRATION (PATHUMMA 3.0) & OFFLINE ADVISOR
# ==============================================================================
THAILLM_DEFAULT_ENDPOINT = "https://thaillm.or.th/api/v1/chat/completions"
THAILLM_DEFAULT_MODEL = "pathumma-thaillm-qwen3-8b-think-3.0.0"

def parse_llm_thinking(raw_text: str) -> tuple:
    thought_match = re.search(r'<thought>(.*?)</thought>', raw_text, re.DOTALL)
    if thought_match:
        thought_content = thought_match.group(1).strip()
        final_answer = re.sub(r'<thought>.*?</thought>', '', raw_text, flags=re.DOTALL).strip()
        return thought_content, final_answer
    return None, raw_text.strip()


def query_thaillm_api(
    endpoint: str,
    api_token: str,
    model: str,
    inspection_result: dict,
    custom_user_query: str = ""
) -> tuple:
    status_th = inspection_result["status_label_th"]
    red_pct = inspection_result["red_percentage"]
    stage = inspection_result["stage_th"]
    days_left = inspection_result["days_left"]
    hex_color = inspection_result["dominant_hex"]

    system_instruction = (
        "คุณคือ 'ผู้เชี่ยวชาญด้านเกษตรอินทรีย์และชีวภัณฑ์' ที่มีความรู้ลึกซึ้งเกี่ยวกับการผลิตและใช้งาน "
        "จุลินทรีย์สังเคราะห์แสง (Photosynthetic Bacteria: PSB สายพันธุ์ Rhodopseudomonas palustris) "
        "หน้าที่ของคุณคือให้คำแนะนำที่เป็นรูปธรรม ชัดเจน ปฏิบัติได้จริง แก่เกษตรกรหรือผู้ปลูกพืช"
    )

    if inspection_result["is_ready"]:
        user_prompt = f"""
ตรวจวิเคราะห์ขวดจุลินทรีย์สังเคราะห์แสง (PSB) จากรูปภาพด้วยระบบ Computer Vision:
- สถานะขวด: {status_th}
- ปริมาณเม็ดสีแดง-ม่วง (Red Pigment Density): {red_pct}%
- ระดับความพร้อม: {stage}
- รหัสสีของน้ำ: {hex_color}

กรุณาให้คำแนะนำแก่เกษตรกรดังต่อไปนี้:
1. อัตราส่วนการผสมน้ำและการใช้งาน (ฉีดพ่นทางใบ, ราดรดทางดิน, แช่เมล็ดพันธุ์, นาข้าว, พืชผักสวนครัว)
2. ประโยชน์ที่พืชจะได้รับเมื่อนำไปใช้ในระยะนี้
3. วิธีการแบ่งเก็บรักษาหัวเชื้อ และเทคนิคการนำไป 'ต่อเชื้อ' รอบใหม่ให้ได้ผลดีที่สุด
"""
    else:
        user_prompt = f"""
ตรวจวิเคราะห์ขวดจุลินทรีย์สังเคราะห์แสง (PSB) จากรูปภาพด้วยระบบ Computer Vision:
- สถานะขวด: {status_th}
- ปริมาณเม็ดสีแดง-ม่วง: {red_pct}% (ยังต่ำกว่าเกณฑ์ความพร้อม)
- ระดับการหมักปัจจุบัน: {stage}
- ระยะเวลาที่คาดว่าต้องหมักต่อ: {days_left}
- รหัสสีของน้ำ: {hex_color}

กรุณาให้คำแนะนำแก่เกษตรกรดังต่อไปนี้:
1. การดูแลรักษาขวดในระยะนี้ (การจัดวางตำแหน่งรับแสงแดด, อุณหภูมิที่เหมาะสม)
2. ข้อปฏิบัติการเขย่าขวดและการระบายแก๊ส (Burping) เพื่อไม่ให้ขวดบวมระเบิด
3. สัญญาณเตือนขวดเสียหรือปนเปื้อน (เช่น น้ำเปลี่ยนเป็นสีเขียวจากสาหร่าย หรือมีกลิ่นเน่าเหม็นเปรี้ยว) และแนวทางแก้ไข
4. ปัจจัยที่ทำให้เชื้อติดสีแดงช้าและวิธีแก้ไข
"""

    if custom_user_query.strip():
        user_prompt += f"\nคำถามเพิ่มเติมจากเกษตรกร:\n{custom_user_query.strip()}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token.strip()}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            endpoint.strip(),
            json=payload,
            headers=headers,
            timeout=25
        )

        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                raw_content = data["choices"][0]["message"]["content"]
                thought, clean_text = parse_llm_thinking(raw_content)
                return True, clean_text, thought
            else:
                return False, f"รูปแบบข้อมูลจากเซิร์ฟเวอร์ไม่ถูกต้อง: {response.text}", None
        elif response.status_code == 401:
            return False, "401 Unauthorized: Bearer Token ของ ThaiLLM ไม่ถูกต้องหรือหมดอายุ กรุณาตรวจสอบ API Key ของท่าน", None
        elif response.status_code == 404:
            return False, f"404 Not Found: ไม่พบ Endpoint หรือโมเดล {model}", None
        else:
            return False, f"HTTP Error {response.status_code}: {response.text[:300]}", None

    except requests.exceptions.Timeout:
        return False, "การเชื่อมต่อหมดเวลา (Timeout) โปรดตรวจสอบอินเทอร์เน็ตหรือลองใหม่อีกครั้ง", None
    except requests.exceptions.ConnectionError:
        return False, "ไม่สามารถเชื่อมต่อไปยังเซิร์ฟเวอร์ ThaiLLM ได้ โปรดตรวจสอบ Endpoint หรือเครือข่าย", None
    except Exception as e:
        return False, f"เกิดข้อผิดพลาดในการเรียก API: {str(e)}", None


def get_offline_expert_advice(inspection_result: dict) -> str:
    is_ready = inspection_result["is_ready"]
    red_pct = inspection_result["red_percentage"]
    stage = inspection_result["stage_th"]

    if is_ready:
        return f"""
### 🌿 คำแนะนำการใช้งานจุลินทรีย์สังเคราะห์แสง (ระยะสมบูรณ์: {red_pct:.1f}%)

ขวดของท่านผ่านเกณฑ์การตรวจสอบ **พร้อมใช้งาน (Ready to Use)** มีความหนาแน่นของเซลล์จุลินทรีย์ *Rhodopseudomonas palustris* และสารรงควัตถุสีแดง (Carotenoids & Bacteriochlorophyll) สูง

#### 1. อัตราส่วนการผสมน้ำสำหรับพืชแต่ละชนิด
* **ฉีดพ่นทางใบ (พืชผักสวนครัว/ไม้ดอก):**
  * อัตราส่วน **1 : 1,000** (จุลินทรีย์ 20 ซีซี หรือ 2 ช้อนโต๊ะ ต่อน้ำ 20 ลิตร)
  * ฉีดพ่นช่วงเช้าแดดอ่อน (06:00 - 09:00 น.) ทุกๆ 7 - 10 วัน ช่วยเร่งการสังเคราะห์แสงและเสริมภูมิต้านทานโรคพืช
* **ราดรดทางดิน / ไม้ผล / นาข้าว:**
  * อัตราส่วน **1 : 500** (จุลินทรีย์ 40-50 ซีซี ต่อน้ำ 20 ลิตร)
  * ราดโคนต้นช่วยปลดปล่อยธาตุฟอสฟอรัสในดิน และตรึงไนโตรเจนจากอากาศ
* **การปรับปรุงบ่อน้ำ / บ่อปลาบ่อกุ้ง:**
  * อัตราส่วน **1 ลิตร ต่อน้ำ 10,000 ลิตร (10 ลูกบาศก์เมตร)** สาดกระจายทั่วผิวน้ำ ช่วยย่อยสลายก๊าซไข่เน่า (H₂S) และแอมโมเนีย

#### 2. เทคนิคการต่อหัวเชื้อรอบใหม่ (ขยายเชื้อ)
* แบ่งหัวเชื้อขวดนี้ไว้ **1 ส่วน (เช่น 200 - 300 มล.)** ผสมกับน้ำสะอาด 1.5 ลิตร
* เติมอาหารเลี้ยงเชื้อ: ไข่ไก่ 1 ฟอง ตีผสมกับน้ำปลา 1 ช้อนโต๊ะ และผงชูรส 1 ช้อนชา (ใส่เพียง 1 ช้อนโต๊ะต่อขวด 1.5 ลิตร)
* ปิดฝาตากแดดจัด จะได้ขวดใหม่แดงเร็วขึ้นภายใน 10-14 วัน

#### 3. การเก็บรักษา
* สามารถเก็บในร่มที่อุณหภูมิห้อง ไม่จำเป็นต้องแช่เย็น ปิดฝาให้สนิท มีอายุใช้งานได้นาน 6 - 12 เดือน
"""
    else:
        return f"""
### ⏳ คำแนะนำการดูแลการหมักจุลินทรีย์สังเคราะห์แสง (ระยะ: {stage})

ปัจจุบันตรวจพบเม็ดสีแดงที่ **{red_pct:.1f}%** ซึ่งยังอยู่ในระยะบ่มหมัก **ยังไม่แนะนำให้นำไปใช้** เนื่องจากจำนวนเซลล์ยังไม่เข้มข้นพอและอาหารโปรตีนในขวดยังย่อยสลายไม่หมด

#### 1. การจัดวางตำแหน่งรับแสงแดด (สำคัญที่สุด)
* จุลินทรีย์สังเคราะห์แสงต้องการพลังงานแสงในการสร้างรงควัตถุ
* วางขวดในจุดที่ **ได้รับแสงแดดจัดอย่างน้อย 6 - 8 ชั่วโมงต่อวัน** (แดดช่วงเที่ยงถึงบ่ายจะช่วยเร่งสีแดงได้ดีที่สุด)
* วางขวดบนพื้นปูนหรือพื้นดินที่มีความร้อนสะท้อน (อุณหภูมิที่เชื้อเจริญได้ดีคือ 30 - 38 °C)

#### 2. การเขย่าขวดและการระบายแก๊ส (Burping)
* **เขย่าขวดวันละ 1 - 2 ครั้ง** เพื่อให้อาหารและหัวเชื้อคลุกเคล้า ไม่ตกตะกอนนอนก้น
* หากพบว่าขวดตึงบวมมาก ให้ค่อยๆ คลายเกลียวฝาเล็กน้อยเพื่อระบายก๊าซ แล้วหมุนปิดให้สนิททันที (ไม่เปิดฝาทิ้งไว้ เพื่อรักษาภาวะไม่ใช้ออกซิเจนแบบไม่สมบูรณ์)

#### 3. วิธีสังเกตอาการผิดปกติและแนวทางแก้ไข
* **ถ้าขวดกลายเป็นสีเขียวเข้ม:** เกิดจากสาหร่ายเซลล์เดียวปนเปื้อน (มักใช้น้ำที่ไม่สะอาดหรือตากแดดน้อยเกินไป) *แนวทาง:* นำไปตากแดดจัดข้ามวัน หากไม่แดงขึ้นให้ทิ้งและเริ่มใหม่ด้วยน้ำต้มสุก
* **ถ้าขวดมีกลิ่นเหม็นเน่าฉุนจัด แต่ไม่มีสีแดงเลยนานเกิน 3 สัปดาห์:** อาหารโปรตีนอาจใส่มากเกินไปจนเน่าเสีย ให้เติมหัวเชื้อแดงเข้มเพิ่มเข้าไป 1 ถ้วยเพื่อช่วยดึงเชื้อ
"""


# ==============================================================================
# 5. SAMPLE IMAGES HELPER
# ==============================================================================
SAMPLE_IMAGES_DIR = "samples"

def get_sample_images():
    samples = []
    if os.path.exists(SAMPLE_IMAGES_DIR):
        for fname in os.listdir(SAMPLE_IMAGES_DIR):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append(os.path.join(SAMPLE_IMAGES_DIR, fname))
    return sorted(samples)


# ==============================================================================
# 6. SIDEBAR: SETTINGS & THAILLM CREDENTIALS
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ การตั้งค่าระบบ (Configuration)")
    
    st.markdown("#### 🔬 การตรวจจับเม็ดสี (CV Thresholds)")
    red_threshold = st.slider(
        "เกณฑ์เม็ดสีแดงสำหรับ 'พร้อมใช้' (%):",
        min_value=5.0,
        max_value=60.0,
        value=25.0,
        step=1.0,
        help="หากสัดส่วนเม็ดสีแดง-ม่วงในของเหลวสูงกว่าเกณฑ์นี้ ระบบจะจัดให้เป็น 'พร้อมใช้งาน'"
    )
    
    roi_option = st.selectbox(
        "บริเวณที่ต้องการตรวจวิเคราะห์ (ROI):",
        options=["liquid_center", "full", "custom"],
        format_func=lambda x: {
            "liquid_center": "🎯 แกนกลางขวดของเหลว (แนะนำสำหรับขวดเดี่ยว)",
            "full": "🖼️ ทั้งภาพ (Full Image)",
            "custom": "📐 กำหนดกรอบพิกัดเอง (Custom Crop)"
        }[x]
    )

    custom_roi_coords = None
    if roi_option == "custom":
        st.markdown("##### กำหนดตำแหน่งกรอบวิเคราะห์ (%)")
        col_x1, col_x2 = st.columns(2)
        with col_x1:
            cx1 = st.slider("X เริ่มต้น", 0.0, 0.8, 0.1, 0.05)
            cy1 = st.slider("Y เริ่มต้น", 0.0, 0.8, 0.1, 0.05)
        with col_x2:
            cx2 = st.slider("X สิ้นสุด", 0.2, 1.0, 0.9, 0.05)
            cy2 = st.slider("Y สิ้นสุด", 0.2, 1.0, 0.9, 0.05)
        custom_roi_coords = (cx1, cy1, cx2, cy2)

    with st.expander("🛠️ ปรับความไวสี HSV ขั้นสูง"):
        hue_tol = st.slider("Hue Tolerance (สีแดงล่าง)", 5, 25, 14)
        min_sat = st.slider("Min Saturation (ตัดสีขาว/ใส)", 10, 100, 45)
        min_val = st.slider("Min Value (ตัดเงามืด)", 10, 100, 30)

    st.markdown("---")
    
    st.markdown("#### 🤖 เชื่อมต่อ ThaiLLM API")
    thaillm_endpoint = st.text_input(
        "API Endpoint:",
        value=THAILLM_DEFAULT_ENDPOINT,
        help="URL สำหรับเรียกใช้งาน API ThaiLLM Chat Completions"
    )
    thaillm_model = st.text_input(
        "Model Name:",
        value=THAILLM_DEFAULT_MODEL,
        help="ชื่อโมเดลภาษาของ ThaiLLM เช่น pathumma-thaillm-qwen3-8b-think-3.0.0"
    )

    default_secret_key = ""
    try:
        if "THAILLM_API_KEY" in st.secrets:
            default_secret_key = st.secrets["THAILLM_API_KEY"]
        elif "AIFORTHAI_KEY" in st.secrets:
            default_secret_key = st.secrets["AIFORTHAI_KEY"]
    except Exception:
        pass

    demo_token = "8esTwrak4niLwPCtPM69FLbFbeuZ3huu"

    thaillm_token = st.text_input(
        "Bearer Token / API Key:",
        value=default_secret_key if default_secret_key else demo_token,
        type="password",
        help="ใส่ Bearer Token ของท่านจากบัญชี ThaiLLM / AI For Thai"
    )

    if st.button("🔌 ทดสอบการเชื่อมต่อ ThaiLLM"):
        with st.spinner("กำลังตรวจสอบสถานะการเชื่อมต่อ..."):
            try:
                test_res = requests.post(
                    thaillm_endpoint.strip(),
                    headers={"Authorization": f"Bearer {thaillm_token.strip()}"},
                    json={"model": thaillm_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5},
                    timeout=8
                )
                if test_res.status_code == 200:
                    st.success("✅ เชื่อมต่อ ThaiLLM สำเร็จ!")
                elif test_res.status_code == 401:
                    st.warning("⚠️ 401 Unauthorized: Bearer Token หมดอายุหรือไม่ถูกต้อง (ระบบจะใช้ผู้เชี่ยวชาญออฟไลน์แทนอัตโนมัติ)")
                else:
                    st.warning(f"⚠️ ตอบกลับด้วยรหัส: {test_res.status_code}")
            except Exception as e:
                st.error(f"❌ เชื่อมต่อไม่สำเร็จ: {e}")

    st.markdown("---")
    st.caption("Developed with Streamlit & OpenCV • Open Source Bio-Agriculture")


# ==============================================================================
# 7. MAIN CONTENT LAYOUT
# ==============================================================================

# Header Banner
st.markdown("""
<div class="header-banner">
    <div class="header-title">
        <span>🧪</span>
        <span>ระบบตรวจสอบและจำแนกขวดจุลินทรีย์สังเคราะห์แสง (PSB Inspector)</span>
    </div>
    <p class="header-subtitle">
        Photosynthetic Bacteria Liquid Bottle Inspection & Quality Classifier powered by Computer Vision & ThaiLLM Pathumma
    </p>
</div>
""", unsafe_allow_html=True)


tab_main, tab_gallery, tab_guide = st.tabs([
    "🔍 ตรวจสอบภาพขวด (Inspection)",
    "🖼️ คลังภาพตัวอย่าง (Sample Gallery)",
    "📖 คู่มือและเกณฑ์มาตรฐาน (Reference & Guide)"
])

with tab_main:
    st.markdown("### 1. เลือกภาพขวดจุลินทรีย์เพื่อตรวจวิเคราะห์")
    
    if "gallery_selected" in st.session_state:
        default_sample_path = st.session_state["gallery_selected"]
    else:
        default_sample_path = None

    col_input_type, col_quick_pick = st.columns([1, 1])

    with col_input_type:
        input_source = st.radio(
            "แหล่งที่มาของภาพ:",
            options=["🗂️ เลือกจากตัวอย่างการทดสอบ (Preset Samples)", "📤 อัปโหลดภาพของท่าน (Upload Image)", "📸 ถ่ายภาพจากกล้อง (Camera)"],
            index=0,
            horizontal=True
        )

    active_image = None
    image_title = ""

    if input_source == "📤 อัปโหลดภาพของท่าน (Upload Image)":
        uploaded_file = st.file_uploader(
            "เลือกไฟล์รูปภาพขวดจุลินทรีย์ (.jpg, .jpeg, .png):",
            type=["jpg", "jpeg", "png"],
            help="รองรับภาพถ่ายขวดจุลินทรีย์ทั้งขวดเดี่ยวหรือแถวขวดตากแดด"
        )
        if uploaded_file is not None:
            active_image = Image.open(uploaded_file)
            image_title = uploaded_file.name

    elif input_source == "📸 ถ่ายภาพจากกล้อง (Camera)":
        camera_pic = st.camera_input("ถ่ายรูปขวดจุลินทรีย์ของคุณทันที:")
        if camera_pic is not None:
            active_image = Image.open(camera_pic)
            image_title = "Camera Capture"

    else:
        sample_files = get_sample_images()
        if sample_files:
            sample_labels = {
                "diagram_stages.jpg": "📊 แผนผังเปรียบเทียบระยะ (Diagram: Stage 0 vs Ready Burgundy)",
                "garden_and_farm_bottles.jpg": "🧴 ขวดน้ำใสเทียบขวดแดงสมบูรณ์ (Garden & Farm Bottles)",
                "fermentation_series.jpg": "☀️ ซีรีส์ 5 ระยะการหมักกลางแจ้ง (5-Stage Fermentation Series)",
                "outdoor_batch_row.jpg": "🌾 แถวขวดหัวเชื้อสีแดงเข้มกลางแจ้ง (Outdoor Mature Batch)"
            }
            
            initial_idx = 0
            if default_sample_path and default_sample_path in sample_files:
                initial_idx = sample_files.index(default_sample_path)
                
            selected_sample_path = st.selectbox(
                "เลือกภาพตัวอย่างสำหรับการทดสอบระบบ:",
                options=sample_files,
                index=initial_idx,
                format_func=lambda p: sample_labels.get(os.path.basename(p), os.path.basename(p))
            )
            if selected_sample_path and os.path.exists(selected_sample_path):
                active_image = Image.open(selected_sample_path)
                image_title = os.path.basename(selected_sample_path)
        else:
            st.info("ไม่พบไฟล์ในโฟลเดอร์ samples/ กรุณาอัปโหลดรูปภาพของท่าน")

    if active_image is None:
        st.info("👆 กรุณาอัปโหลดรูปภาพ หรือเลือกภาพตัวอย่างจากเมนูด้านบนเพื่อเริ่มต้นการตรวจสอบ")
    else:
        st.markdown("---")
        with st.spinner("🔬 กำลังประมวลผลเม็ดสีในพื้นที่สี HSV..."):
            result = analyze_psb_bottle(
                pil_image=active_image,
                red_threshold_pct=red_threshold,
                roi_mode=roi_option,
                custom_roi=custom_roi_coords,
                hue_tolerance=hue_tol,
                min_saturation=min_sat,
                min_value=min_val
            )

        st.markdown("### 2. ผลการวิเคราะห์ด้วย Computer Vision (HSV Color Space)")

        if result["is_ready"]:
            st.markdown(f"""
            <div style="background-color: #EBFBEE; border: 2px solid #51CF66; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="status-badge badge-ready">✅ {result['status_label_th']}</span>
                        <div style="color: #2B8A3E; font-size: 15px; margin-top: 8px; font-weight: 500;">
                            ✨ จุลินทรีย์ขยายตัวเต็มที่ รงควัตถุสีแดงเข้มมีความเข้มข้นสูง พร้อมสำหรับการฉีดพ่นและใช้งานทางการเกษตร
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 32px; font-weight: 800; color: #2B8A3E;">{result['red_percentage']}%</span>
                        <div style="font-size: 12px; color: #51CF66;">(เกณฑ์ขั้นต่ำ: {result['red_threshold']}%)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #FFF9DB; border: 2px solid #FCC419; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span class="status-badge badge-not-ready">⏳ {result['status_label_th']}</span>
                        <div style="color: #C25E00; font-size: 15px; margin-top: 8px; font-weight: 500;">
                            ⚠️ ปริมาณเม็ดสีแดง-ม่วงยังไม่ถึงเกณฑ์ที่กำหนด ({result['red_percentage']}% < {result['red_threshold']}%) ควรหมักและตากแดดต่อไป
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 32px; font-weight: 800; color: #D9480F;">{result['red_percentage']}%</span>
                        <div style="font-size: 12px; color: #F59F00;">(เกณฑ์ขั้นต่ำ: {result['red_threshold']}%)</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ความหนาแน่นเม็ดสีแดง</div>
                <div class="metric-value" style="color: {'#2B8A3E' if result['is_ready'] else '#D9480F'};">
                    {result['red_percentage']}%
                </div>
                <div class="metric-sub">เทียบกับเกณฑ์ {result['red_threshold']}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ระดับการหมัก</div>
                <div class="metric-value" style="font-size: 18px; color: #343A40; margin-top: 6px;">
                    {result['stage_th'].split('(')[0]}
                </div>
                <div class="metric-sub">{result['stage_desc']}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ระยะเวลาคาดการณ์</div>
                <div class="metric-value" style="font-size: 20px; color: #495057;">
                    {result['days_left']}
                </div>
                <div class="metric-sub">จนกว่าจะสมบูรณ์</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">โทนสีเฉลี่ยของของเหลว</div>
                <div style="display: flex; justify-content: center; align-items: center; gap: 8px; margin: 6px 0;">
                    <div style="width: 26px; height: 26px; border-radius: 50%; background-color: {result['dominant_hex']}; border: 2px solid #CED4DA;"></div>
                    <span style="font-size: 18px; font-weight: 700; color: #212529;">{result['dominant_hex'].upper()}</span>
                </div>
                <div class="metric-sub">RGB: {result['mean_rgb'][0]}, {result['mean_rgb'][1]}, {result['mean_rgb'][2]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.image(result["full_preview"], caption=f"1. ภาพต้นฉบับ & กรอบ ROI ({image_title})", use_container_width=True)
        with col_img2:
            st.image(result["mask_image"], caption="2. แมสก์เม็ดสีแดงใน HSV (White = Red Pigment)", use_container_width=True)
        with col_img3:
            st.image(result["overlay_roi"], caption="3. ไฮไลต์เม็ดสีสังเคราะห์แสง (Green Highlight)", use_container_width=True)

        st.markdown("---")

        st.markdown("### 3. คำแนะนำทางการเกษตรโดยปัญญาประดิษฐ์ (ThaiLLM Pathumma)")
        st.caption("ระบบจะส่งผลการตรวจสอบไปยังโมเดลภาษา **pathumma-thaillm-qwen3-8b-think-3.0.0** เพื่อขอคำแนะนำการดูแลหรืออัตราส่วนผสม")

        col_llm_action, col_llm_options = st.columns([1, 1])
        with col_llm_action:
            generate_advice_btn = st.button("🌱 ขอคำแนะนำจาก ThaiLLM ทันที", type="primary", use_container_width=True)
        with col_llm_options:
            user_question = st.text_input("💬 มีคำถามเพิ่มเติมถึงผู้เชี่ยวชาญ (ไม่บังคับ):", placeholder="เช่น ต้องการใช้กับมะเขือเทศและนาข้าว...")

        if generate_advice_btn or "last_advice" in st.session_state:
            if generate_advice_btn:
                with st.spinner("🤖 กำลังเชื่อมต่อไปยัง ThaiLLM (Pathumma 3.0)..."):
                    success, content, thought = query_thaillm_api(
                        endpoint=thaillm_endpoint,
                        api_token=thaillm_token,
                        model=thaillm_model,
                        inspection_result=result,
                        custom_user_query=user_question
                    )
                    st.session_state["last_advice"] = {
                        "success": success,
                        "content": content,
                        "thought": thought,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }

            advice_data = st.session_state.get("last_advice", {})
            
            if advice_data.get("success"):
                st.success("✅ ได้รับคำแนะนำจากโมเดล ThaiLLM เรียบร้อยแล้ว")
                
                if advice_data.get("thought"):
                    with st.expander("💭 ขั้นตอนการคิดของโมเดล (ThaiLLM Reasoning Process)"):
                        st.markdown(f'<div class="thinking-box">{advice_data["thought"]}</div>', unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="advice-box">
                    {advice_data["content"]}
                </div>
                """, unsafe_allow_html=True)

            else:
                err_msg = advice_data.get("content", "Unknown Error")
                st.warning(f"⚠️ **การแจ้งเตือนจาก ThaiLLM API:** {err_msg}")
                st.info("💡 **ระบบเปิดใช้งาน 'ผู้เชี่ยวชาญเกษตรอินทรีย์ออฟไลน์ (Offline Bio-Expert)' อัตโนมัติ เพื่อให้คำแนะนำที่สมบูรณ์ทันที:**")
                
                offline_advice = get_offline_expert_advice(result)
                st.markdown(f"""
                <div class="advice-box">
                    {offline_advice}
                </div>
                """, unsafe_allow_html=True)

        # ----------------------------------------------------------------------
        # Step 4: Export Report & Summary (JSON & PDF)
        # ----------------------------------------------------------------------
        st.markdown("---")
        with st.expander("📥 ดาวน์โหลดรายงานผลการตรวจสอบ (Export Report)", expanded=True):
            report_dict = {
                "report_id": f"PSB-{int(time.time())}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_name": image_title,
                "classification": {
                    "is_ready": result["is_ready"],
                    "status_th": result["status_label_th"],
                    "status_en": result["status_label_en"],
                    "red_pigment_percentage": result["red_percentage"],
                    "threshold_applied": result["red_threshold"],
                    "fermentation_stage": result["stage_th"],
                    "stage_en": result["stage_en"],  # <-- เพิ่มบรรทัดนี้
                    "estimated_days_left": result["days_left"]
                },
                "color_metrics": {
                    "hex": result["dominant_hex"],
                    "rgb": result["mean_rgb"],
                    "hsv": result["mean_hsv"]
                }
            }

            col_pdf, col_json = st.columns(2)

            with col_pdf:
                if FPDF_AVAILABLE:
                    pdf_data = generate_pdf_report(report_dict)
                    st.download_button(
                        label="📑 ดาวน์โหลดรายงาน PDF",
                        data=pdf_data,
                        file_name=f"PSB_Report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.warning("⚠️ กรุณาติดตั้ง fpdf2 ด้วยคำสั่ง: pip install fpdf2 เพื่อเปิดใช้งานการดาวน์โหลด PDF")

            with col_json:
                report_json = json.dumps(report_dict, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📄 ดาวน์โหลดรายงาน JSON",
                    data=report_json,
                    file_name=f"PSB_Inspection_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )


# ==============================================================================
# TAB 2: SAMPLE GALLERY
# ==============================================================================
with tab_gallery:
    st.markdown("### 🖼️ คลังภาพตัวอย่างสำหรับการเปรียบเทียบระยะการหมัก")
    st.markdown("ภาพตัวอย่างจริงที่ใช้ในการทดสอบระบบ ประกอบด้วยขวดในระยะต่างๆ:")

    samples_list = get_sample_images()
    if samples_list:
        g_cols = st.columns(len(samples_list))
        for idx, s_path in enumerate(samples_list):
            with g_cols[idx]:
                s_name = os.path.basename(s_path)
                s_img = Image.open(s_path)
                st.image(s_img, caption=s_name, use_container_width=True)
                if st.button(f"🔍 ตรวจสอบภาพนี้", key=f"btn_gallery_{idx}"):
                    st.session_state["gallery_selected"] = s_path
                    st.rerun()
    else:
        st.write("ไม่พบภาพใน samples/")


# ==============================================================================
# TAB 3: REFERENCE & USAGE GUIDE
# ==============================================================================
with tab_guide:
    st.markdown("""
    ### 📖 คู่มือมาตรฐานการตรวจและใช้งานจุลินทรีย์สังเคราะห์แสง (PSB)

    **จุลินทรีย์สังเคราะห์แสง (Photosynthetic Bacteria: PSB)** โดยทั่วไปคือกลุ่มแบคทีเรีย *Rhodopseudomonas palustris* 
    ที่มีความสามารถพิเศษในการใช้พลังงานแสงอาทิตย์ในการย่อยสลายสารอินทรีย์ ตรึงไนโตรเจน และปลดปล่อยธาตุอาหารให้แก่พืช

    ---

    #### 🧪 4 ระยะการหมักของขวดจุลินทรีย์สังเคราะห์แสง
    | ระยะ | ลักษณะสีของน้ำ | ปริมาณเม็ดสีแดง | สถานะ | คำแนะนำ |
    | :--- | :--- | :---: | :---: | :--- |
    | **Stage 0: เริ่มต้น** | ขาวขุ่น / น้ำใสอมเหลือง | < 5% | ยังไม่พร้อม | ตากแดดจัด 6-8 ชม./วัน เขย่าขวดทุกวัน |
    | **Stage 1: เริ่มมีสี** | ชมพูอ่อน / ส้มจาง | 5% - 15% | บ่มหมัก | เริ่มสร้างเม็ดสี ระบายแก๊สหากขวดบวม |
    | **Stage 2: สีเข้มขึ้น** | แดงส้ม / แดงสด | 15% - 25% | บ่มหมัก | จุลินทรีย์ทวีคูณอย่างรวดเร็ว ตากแดดต่ออีก 7-10 วัน |
    | **Stage 3: สมบูรณ์** | แดงเข้ม / ไวน์แดง / เลือดนก | > 25% | **พร้อมใช้งาน** | นำไปผสมน้ำฉีดพ่น หรือเก็บเป็นหัวเชื้อต่อขวดใหม่ |

    ---

    #### 🌾 ตารางอัตราส่วนการผสมน้ำตามชนิดพืช
    * **พืชผักใบ / พืชผักสวนครัว:** 1 : 1,000 (20 ซีซี ต่อน้ำ 20 ลิตร) ฉีดพ่นทางใบทุก 7 วัน
    * **ไม้ผล / ไม้ยืนต้น:** 1 : 500 (40 ซีซี ต่อน้ำ 20 ลิตร) ราดโคนต้นเดือนละ 2 ครั้ง
    * **นาข้าว:** สาดหัวเชื้อเข้มข้น 1-2 ลิตร ต่อไร่ หลังหว่านข้าว 15 วัน ช่วยย่อยตอซังและตรึงไนโตรเจน
    * **การบำบัดน้ำเสีย / บ่อปลา:** 1 ลิตร ต่อน้ำ 10,000 ลิตร สาดให้ทั่วผิวน้ำทุก 15 วัน

    ---

    #### 💡 เทคนิคการแก้ปัญหาขวดไม่ยอมแดง
    1. **น้ำที่ใช้:** ควรใช้น้ำจากแหล่งธรรมชาติ น้ำบาดาล หรือน้ำประปาที่พักคลอรีนแล้ว 2-3 วัน
    2. **อาหารเลี้ยงเชื้อ:** ไข่ไก่ 1 ฟอง + น้ำปลา 1 ช้อนโต๊ะ + ชูรส 1 ช้อนชา (ใส่เพียง 1 ช้อนโต๊ะ ต่อขวด 1.5 ลิตร อย่าใส่มากเกินไปจะเน่า)
    3. **ความร้อนและแสงแดด:** ต้องวางบนพื้นปูนหรือจุดที่แดดส่องถึงทั้งวัน เชื้อชอบอุณหภูมิ 30-38°C
    """)