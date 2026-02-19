
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

# モデルの読み込み
@st.cache_resource
def load_model():
    # 初回実行時にYOLO11モデルをダウンロード/ロード
    return YOLO("yolo11n.pt")

model = load_model()

def get_color_name(img_bgr):
    """HSV空間を利用した日本の信号機の色判定"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
   
    # 日本の信号機特性に合わせたHSV範囲設定
    # 青信号（実際は緑〜青緑）
    lower_blue = np.array([35, 70, 50])
    upper_blue = np.array([95, 255, 255])
   
    # 黄信号
    lower_yellow = np.array([15, 70, 50])
    upper_yellow = np.array([35, 255, 255])
   
    # 赤信号（0-10付近と170-180付近の2か所）
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                             cv2.inRange(hsv, lower_red2, upper_red2))

    counts = {
        "青色（進めます）": cv2.countNonZero(mask_blue),
        "黄色（注意）": cv2.countNonZero(mask_yellow),
        "赤色（止まれ）": cv2.countNonZero(mask_red)
    }
   
    # 最も面積が大きい色を返す
    max_color = max(counts, key=counts.get)
    if counts[max_color] < 30: # 閾値以下なら判定不能
        return "判定不能"
    return max_color

# --- メイン機能 ---
img_file = st.camera_input("カメラで信号機を撮影してください")

if img_file:
    image = Image.open(img_file)
    frame = np.array(image)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # YOLO11で物体検出 (class 9 = traffic light)
    results = model(frame_bgr, classes=[9], conf=0.45)
   
    found = False
    for r in results:
        for box in r.boxes:
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
           
            # 信号部分を切り抜いて色判定
            crop = frame_bgr[y1:y2, x1:x2]
            color_res = get_color_name(crop)
           
            # 結果表示
            if "青色" in color_res:
                st.success(f"✅ {color_res}")
            elif "赤色" in color_res:
                st.error(f"🛑 {color_res}")
            else:
                st.warning(f"⚠️ {color_res}")
           
            # 描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, color_res, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    if not found:
        st.info("信号機が検出されませんでした。正面から大きく写してください。")

    st.image(frame, channels="RGB", caption="解析中...")

st.divider()
st.caption("⚠️ 注意: 本アプリは補助ツールです。必ず周囲の音や状況を自身の感覚で確認してください。")
