
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- ページ設定 ---
st.set_page_config(page_title="信号機判別アシスタント", page_icon="🚦")

# スタイル調整（視認性向上）
st.markdown("""
    <style>
    .big-font { font-size:30px !important; font-weight: bold; }
    .stAlert { border: 2px solid; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 信号機判別支援アプリ")
st.write("YOLO11とOpenCVを使用して、カメラ画像から信号機の色を判定します。")

# モデルの読み込み（Streamlit Cloud用に軽量なnモデルを使用）
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

def get_color_name(img_bgr):
    """HSV空間を利用した日本の信号機の色判定"""
    # 画像が空の場合は「不明」を返す
    if img_bgr is None or img_bgr.size == 0:
        return "判定不能"
       
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
   
    # 日本の信号機特性に合わせたHSV範囲設定
    color_ranges = {
        "青色（進めます）": [((35, 70, 50), (95, 255, 255))],
        "黄色（注意）": [((15, 70, 50), (35, 255, 255))],
        "赤色（止まれ）": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))]
    }

    counts = {}
    for color_name, ranges in color_ranges.items():
        mask = None
        for (lower, upper) in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        counts[color_name] = cv2.countNonZero(mask)
   
    max_color = max(counts, key=counts.get)
    # 面積が小さすぎる場合は誤検知として無視
    if counts[max_color] < 50:
        return "判定不能"
    return max_color

# --- メイン機能 ---
# 手動アップロードも可能にしておくとデバッグしやすいです
img_file = st.camera_input("信号機を撮影してください")

if img_file:
    # 画像の読み込み
    image = Image.open(img_file)
    frame_rgb = np.array(image)
    # OpenCV形式(BGR)に変換
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # YOLO11で物体検出 (class 9 = traffic light)
    # Streamlit CloudのCPU環境を考慮し、confを少し調整
    results = model.predict(frame_bgr, classes=[9], conf=0.3, verbose=False)
   
    found = False
    for r in results:
        for box in r.boxes:
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
           
            # 信号部分を切り抜いて色判定
            crop = frame_bgr[y1:y2, x1:x2]
            color_res = get_color_name(crop)
           
            # UIへの結果表示
            if "青色" in color_res:
                st.success(f"✅ 【判別結果】 {color_res}")
            elif "赤色" in color_res:
                st.error(f"🛑 【判別結果】 {color_res}")
            elif "黄色" in color_res:
                st.warning(f"⚠️ 【判別結果】 {color_res}")
           
            # 解析後の画像をリサイズして表示（スマホで見やすくするため）
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame_rgb, color_res, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    if not found:
        st.info("信号機が見つかりませんでした。正面から大きく写してください。")

    # 最終的な画像表示
    st.image(frame_rgb, caption="解析プレビュー", use_container_width=True)

st.divider()
st.caption("⚠️ 本アプリは補助的なツールです。必ず周囲の音や誘導鈴、歩行者用信号の音を確認して安全を確保してください。")
