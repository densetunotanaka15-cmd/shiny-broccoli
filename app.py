import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import streamlit.components.v1 as components

def speak(text):
    """Web Speech APIで日本語音声読み上げ"""
    components.html(f"""
        <script>
        (function() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance({text!r});
            msg.lang = 'ja-JP';
            msg.rate = 0.9;
            msg.pitch = 1.0;
            msg.volume = 1.0;
            window.speechSynthesis.speak(msg);
        }})();
        </script>
    """, height=0)

# --- ページ設定 ---
st.set_page_config(page_title="信号機判別アシスタント", page_icon="🚦", layout="wide")

# スタイル調整（視覚障害者向け：大きなフォント・高コントラスト・シンプルなレイアウト）
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Noto Sans JP', sans-serif;
    }

    /* 全体の背景を暗くして見やすく */
    .stApp {
        background-color: #111111;
        color: #FFFFFF;
    }

    /* タイトル */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        text-align: center;
        padding: 0.5em 0;
        letter-spacing: 0.05em;
    }

    /* 説明文 */
    p, .stMarkdown p {
        font-size: 1.4rem !important;
        color: #DDDDDD !important;
        text-align: center;
        line-height: 1.8;
    }

    /* カメラ入力ラベル */
    label {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
    }

    /* 結果ボックス共通 */
    .result-box {
        border-radius: 20px;
        padding: 40px 30px;
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        margin: 20px 0;
        letter-spacing: 0.05em;
        line-height: 1.4;
        border: 6px solid;
    }

    /* 青（進め） */
    .result-blue {
        background-color: #003399;
        color: #FFFFFF;
        border-color: #4488FF;
    }

    /* 赤（止まれ） */
    .result-red {
        background-color: #990000;
        color: #FFFFFF;
        border-color: #FF4444;
    }

    /* 黄（注意） */
    .result-yellow {
        background-color: #886600;
        color: #FFFFFF;
        border-color: #FFCC00;
    }

    /* 不明 */
    .result-unknown {
        background-color: #333333;
        color: #AAAAAA;
        border-color: #666666;
    }

    /* 情報メッセージ */
    .stInfo {
        font-size: 1.5rem !important;
    }

    /* 注意書き */
    .stCaption {
        font-size: 1.2rem !important;
        color: #AAAAAA !important;
        text-align: center;
    }

    /* カメラウィジェット */
    .stCameraInput > div {
        border: 4px dashed #555555;
        border-radius: 16px;
        padding: 10px;
    }

    /* 画像表示 */
    .stImage img {
        border-radius: 16px;
        border: 4px solid #444444;
    }

    /* セパレーター */
    hr {
        border-color: #444444 !important;
        margin: 2em 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚦 信号機判別支援アプリ")
st.write("YOLO11とOpenCVを使用して、カメラ画像から信号機の色を判定します。")
st.write("📷 下のカメラボタンで信号機を撮影してください。")

# モデルの読み込み（Streamlit Cloud用に軽量なnモデルを使用）
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

def get_color_name(img_bgr):
    """HSV空間を利用した日本の信号機の色判定"""
    if img_bgr is None or img_bgr.size == 0:
        return "判定不能"

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # 日本の信号機特性に合わせたHSV範囲設定
    color_ranges = {
        "青色（進めます）": [((35, 70, 50), (95, 255, 255))],
        "黄色（注意）":     [((15, 70, 50), (35, 255, 255))],
        "赤色（止まれ）":   [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (180, 255, 255))]
    }

    counts = {}
    for color_name, ranges in color_ranges.items():
        mask = None
        for (lower, upper) in ranges:
            m = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        counts[color_name] = cv2.countNonZero(mask)

    max_color = max(counts, key=counts.get)
    if counts[max_color] < 50:
        return "判定不能"
    return max_color

# --- メイン機能 ---
img_file = st.camera_input("📸 信号機を撮影してください")

if img_file:
    image = Image.open(img_file)
    frame_rgb = np.array(image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # YOLO11で物体検出 (class 9 = traffic light)
    results = model.predict(frame_bgr, classes=[9], conf=0.3, verbose=False)

    found = False
    for r in results:
        for box in r.boxes:
            found = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame_bgr[y1:y2, x1:x2]
            color_res = get_color_name(crop)

            # 大きく・わかりやすく結果を表示
            if "青色" in color_res:
                st.markdown(f"""
                    <div class="result-box result-blue">
                        ✅ 進めます<br>
                        <span style="font-size:1.6rem">（青信号）</span>
                    </div>
                """, unsafe_allow_html=True)
                speak("青信号です。進めます。")
            elif "赤色" in color_res:
                st.markdown(f"""
                    <div class="result-box result-red">
                        🛑 止まれ<br>
                        <span style="font-size:1.6rem">（赤信号）</span>
                    </div>
                """, unsafe_allow_html=True)
                speak("赤信号です。止まってください。")
            elif "黄色" in color_res:
                st.markdown(f"""
                    <div class="result-box result-yellow">
                        ⚠️ 注意してください<br>
                        <span style="font-size:1.6rem">（黄信号）</span>
                    </div>
                """, unsafe_allow_html=True)
                speak("黄信号です。注意してください。")
            else:
                st.markdown(f"""
                    <div class="result-box result-unknown">
                        ❓ 判定できませんでした<br>
                        <span style="font-size:1.6rem">もう一度撮影してください</span>
                    </div>
                """, unsafe_allow_html=True)
                speak("判定できませんでした。もう一度撮影してください。")

            # バウンディングボックスと結果テキストを描画
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame_rgb, color_res, (x1, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if not found:
        st.markdown("""
            <div class="result-box result-unknown">
                🔍 信号機が見つかりませんでした<br>
                <span style="font-size:1.6rem">正面から大きく写してください</span>
            </div>
        """, unsafe_allow_html=True)
        speak("信号機が見つかりませんでした。正面から大きく写してください。")

    st.image(frame_rgb, caption="解析プレビュー", use_container_width=True)

st.divider()
st.caption("⚠️ 本アプリは補助的なツールです。必ず周囲の音・誘導鈴・歩行者用信号の音を確認して安全を確保してください。")

