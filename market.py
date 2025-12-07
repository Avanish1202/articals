import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION -----------------
# Impact Table: Maps Pattern Name -> (Score, Strength Label)
IMPACT_TABLE = {
    # UP Patterns
    "Hammer":           (2.0, "Strong Up"),
    "Inverted Hammer":  (1.5, "Medium Up"),
    "Bullish Marubozu": (3.0, "Very Strong Up"),
    "Bullish Engulfing":(3.0, "Very Strong Up"),
    "Morning Star":     (3.0, "Very Strong Up"),
    "3 White Soldiers": (3.0, "Very Strong Up"),
    "Piercing Pattern": (2.0, "Strong Up"),
    "Bullish Kick":     (2.5, "Strong Up"),
    "Dragonfly Doji":   (2.0, "Strong Reversal Up"),
    
    # DOWN Patterns
    "Hanging Man":      (-2.0, "Strong Down"),
    "Shooting Star":    (-2.0, "Strong Down"),
    "Bearish Marubozu": (-3.0, "Very Strong Down"),
    "Bearish Engulfing":(-3.0, "Very Strong Down"),
    "Evening Star":     (-3.0, "Very Strong Down"),
    "3 Black Crows":    (-3.0, "Very Strong Down"),
    "Dark Cloud Cover": (-2.0, "Strong Down"),
    "Gravestone Doji":  (-2.0, "Strong Reversal Down"),

    # NEUTRAL / WEAK
    "Doji":             (0.0, "Neutral"),
    "Spinning Top":     (0.5, "Weak Indecision"),
    "High Wave":        (0.5, "Weak Indecision")
}

# ---------------- STREAMLIT UI SETUP -----------------
st.set_page_config(layout="wide", page_title="Candle Pattern AI")

st.title("📈 Candlestick Pattern AI")
st.markdown("Upload a clear screenshot of a candlestick chart. The AI will detect candles, identify patterns, and predict the next move based on the last 5 candles.")

# File Uploader
uploaded_file = st.file_uploader("Upload Chart Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # ---------------- 1. IMAGE PROCESSING ----------------
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, 1)
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ---------------- 2. DETECT CANDLES ----------------
    candles_detected = []

    # Pass 1: Get average height
    temp_heights = []
    for cnt in contours:
        _, _, w, h = cv2.boundingRect(cnt)
        if w > 2 and h > 5 and w < 50: temp_heights.append(h)
    avg_height = np.mean(temp_heights) if temp_heights else 10

    # Pass 2: Identification
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 5 or w > 50: continue 

        roi = gray[y:y+h, x:x+w]
        candle_crop = img_rgb[y:y+h, x:x+w]
        avg_color = np.mean(candle_crop, axis=(0, 1))
        is_green = avg_color[1] > avg_color[0] + 5
        
        col_data = roi[:, w//2]
        candle_pix = np.where(col_data < 240)[0]
        
        if len(candle_pix) > 0:
            y_top, y_bot = candle_pix[0], candle_pix[-1]
            total_len = max(1, y_bot - y_top)
            
            row_widths = [np.count_nonzero(roi[r, :] < 240) for r in range(h)]
            max_w = max(row_widths) if row_widths else 1
            body_rows = np.where(np.array(row_widths) > max_w * 0.6)[0]
            
            if len(body_rows) > 0:
                body_len = body_rows[-1] - body_rows[0]
                body_top = body_rows[0]
                body_bot = body_rows[-1]
            else: 
                body_len = 1
                body_top = y_top + total_len//2
                body_bot = body_top + 1

            upper_wick = body_top - y_top
            lower_wick = y_bot - body_bot

            # NAMING LOGIC
            body_pct = body_len / total_len
            upper_pct = upper_wick / total_len
            lower_pct = lower_wick / total_len
            name = "Unknown"

            if body_pct < 0.10:
                if lower_pct > 0.6: name = "Dragonfly Doji"
                elif upper_pct > 0.6: name = "Gravestone Doji"
                else: name = "Doji"
            elif body_pct > 0.80:
                name = "Bullish Marubozu" if is_green else "Bearish Marubozu"
            elif body_pct < 0.35:
                if lower_wick > body_len * 2 and upper_wick < body_len:
                    name = "Hammer" if is_green else "Hanging Man"
                elif upper_wick > body_len * 2 and lower_wick < body_len:
                    name = "Inverted Hammer" if is_green else "Shooting Star"
                else:
                    name = "Spinning Top"
            else:
                direction = "Bullish" if is_green else "Bearish"
                if total_len > avg_height * 1.2:
                    name = f"{direction} Marubozu" 
                else:
                    name = "Spinning Top"

            candles_detected.append({
                'rect': (x, y, w, h),
                'name': name,
                'is_green': is_green,
                'x': x
            })

    candles_detected.sort(key=lambda c: c['x'])

    # ---------------- 3. PREDICT WITH PERCENTAGE ----------------
    if len(candles_detected) > 0:
        recent_candles = candles_detected[-5:] 
        total_score = 0
        reasons = []

        for c in recent_candles:
            score_data = IMPACT_TABLE.get(c['name'], (0, "Neutral"))
            if score_data[0] == 0 and c['name'] != "Doji":
                score = 0.5 if c['is_green'] else -0.5 
            else:
                score, impact = score_data

            total_score += score
            color_icon = "🟢" if c['is_green'] else "🔴"
            reasons.append(f"{color_icon} {c['name']} (Score: {score})")

        # --- CALCULATION LOGIC ---
        # We assume a score of +/- 10 is 100% certainty (max score)
        # 50% is Neutral. 
        # Formula: Base 50% + (Score * 5)
        raw_percentage = 50 + (total_score * 6) 
        
        # Clamp between 0 and 100
        probability = max(0, min(100, raw_percentage))

        if total_score >= 1.0:
            direction = "UP ⬆️"
            prediction_color = "green"
            display_prob = probability
        elif total_score <= -1.0:
            direction = "DOWN ⬇️"
            prediction_color = "red"
            display_prob = 100 - probability # Invert for display (e.g., 80% Down)
        else:
            direction = "SIDEWAYS ➡️"
            prediction_color = "gray"
            display_prob = 50.0

        # ---------------- 4. VISUALIZATION ----------------
        col1, col2 = st.columns([1, 3])

        with col1:
            st.subheader("Analysis Results")
            
            # BIG METRICS
            st.metric(label="Predicted Direction", value=direction)
            st.metric(label="Confidence Level", value=f"{display_prob:.1f}%")
            
            st.divider()
            st.write("**Signal Strength (Net Score):**")
            st.info(f"{total_score:.1f}")
            
            st.write("**Pattern Breakdown (Last 5):**")
            for r in reasons:
                st.write(r)

        with col2:
            st.subheader("Chart Visualization")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(img_rgb)
            
            for i, c in enumerate(candles_detected):
                x, y, w, h = c['rect']
                if i >= len(candles_detected) - 5:
                    edge_color = 'blue'
                    line_width = 2
                else:
                    base_score = IMPACT_TABLE.get(c['name'], (0,0))[0]
                    if base_score > 0: edge_color = '#00FF00'
                    elif base_score < 0: edge_color = '#FF0000'
                    else: edge_color = '#FFFF00'
                    line_width = 1

                rect = plt.Rectangle((x, y), w, h, linewidth=line_width, edgecolor=edge_color, facecolor='none')
                ax.add_patch(rect)
                
                clean_name = c['name'].replace("Bullish ", "").replace("Bearish ", "")
                ax.text(x+w/2, y+h+20, clean_name, 
                         rotation=90, fontsize=8, color='red', ha='center', va='top', fontweight='bold')

            ax.axis('off')
            st.pyplot(fig)
    else:
        st.warning("No candles detected. Please upload a clearer image.")
