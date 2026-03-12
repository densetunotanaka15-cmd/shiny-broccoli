"""
視覚障害者向け信号機判別ウェブアプリ
Traffic Signal Detection App for Visually Impaired
Using YOLOv11 + OpenCV + Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import base64
from pathlib import Path
import tempfile
import os

# ページ設定
st.set_page_config(
    page_title="信号機判別アプリ | Traffic Signal Detector",
    page_icon="🚦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── PWA対応 (スマートフォンでホーム画面に追加可能) ─────────
st.markdown("""
<link rel="manifest" href="/app/static/manifest.json">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="信号判別">
<meta name="theme-color" content="#0A0A0F">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<link rel="apple-touch-icon" href="/app/static/icon-192.png">
<script>
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/app/static/sw.js');
  }
</script>
""", unsafe_allow_html=True)

# ─── カスタムCSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700;900&family=Space+Grotesk:wght@400;600;700&display=swap');

:root {
    --red:    #FF3B3B;
    --yellow: #FFD600;
    --green:  #00E676;
    --bg:     #0A0A0F;
    --card:   #12121A;
    --border: #2A2A3A;
    --text:   #E8E8F0;
    --muted:  #6B6B85;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Noto Sans JP', sans-serif;
}

[data-testid="stHeader"] { background: transparent !important; }

.hero {
    text-align: center;
    padding: 2rem 0 1rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin: 0;
    background: linear-gradient(135deg, #FF3B3B 0%, #FFD600 50%, #00E676 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

.signal-box {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    font-size: 5rem;
    margin: 1rem 0;
    border: 2px solid var(--border);
    background: var(--card);
    transition: all 0.3s ease;
}
.signal-red    { border-color: var(--red);    box-shadow: 0 0 40px rgba(255,59,59,0.3); }
.signal-yellow { border-color: var(--yellow); box-shadow: 0 0 40px rgba(255,214,0,0.3); }
.signal-green  { border-color: var(--green);  box-shadow: 0 0 40px rgba(0,230,118,0.3); }
.signal-none   { border-color: var(--border); }

.label-red    { color: var(--red);    font-weight: 900; font-size: 2rem; }
.label-yellow { color: var(--yellow); font-weight: 900; font-size: 2rem; }
.label-green  { color: var(--green);  font-weight: 900; font-size: 2rem; }
.label-none   { color: var(--muted);  font-weight: 700; font-size: 1.4rem; }

.conf-bar-wrap { margin-top: 1rem; }
.conf-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
.conf-bar {
    height: 8px;
    border-radius: 4px;
    background: var(--border);
    overflow: hidden;
}
.conf-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }

.info-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: var(--muted);
}
.info-card strong { color: var(--text); }

.stButton > button {
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Noto Sans JP', sans-serif !important;
    font-weight: 700 !important;
    padding: 0.6rem 1.4rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
}

[data-testid="stFileUploader"] {
    background: var(--card);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 1rem;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: var(--card) !important;
    border-radius: 8px !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
}
.stTabs [aria-selected="true"] {
    background: #1E1E2E !important;
    color: var(--text) !important;
    border-color: var(--green) !important;
}

div[data-testid="stMarkdownContainer"] p { color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ─── ユーティリティ関数 ───────────────────────────────────
def load_yolo_model():
    """YOLOv11モデルを読み込む"""
    try:
        from ultralytics import YOLO
        model_path = Path("models/yolo11_traffic.pt")
        if model_path.exists():
            model = YOLO(str(model_path))
        else:
            # 事前学習済みyolo11n をベースに使用
            model = YOLO("yolo11n.pt")
            st.warning("⚠️ カスタムモデルが見つかりません。デモ用にyolo11n.ptを使用しています。")
        return model
    except ImportError:
        st.error("ultralytics がインストールされていません。requirements.txt を確認してください。")
        return None
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None


def analyze_traffic_light_color(roi: np.ndarray) -> tuple[str, float]:
    """
    OpenCVでROI内の信号色を判別する
    Returns: (color_label, confidence)
    """
    if roi is None or roi.size == 0:
        return "unknown", 0.0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # HSV範囲定義
    red_lower1  = np.array([0,   120, 70]);  red_upper1  = np.array([10,  255, 255])
    red_lower2  = np.array([160, 120, 70]);  red_upper2  = np.array([180, 255, 255])
    yellow_lower = np.array([20,  100, 100]); yellow_upper = np.array([35,  255, 255])
    green_lower  = np.array([40,  80,  80]);  green_upper  = np.array([90,  255, 255])

    mask_red    = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2)
    )
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_green  = cv2.inRange(hsv, green_lower,  green_upper)

    total = roi.shape[0] * roi.shape[1]
    if total == 0:
        return "unknown", 0.0

    count_r = cv2.countNonZero(mask_red)
    count_y = cv2.countNonZero(mask_yellow)
    count_g = cv2.countNonZero(mask_green)

    counts = {"red": count_r, "yellow": count_y, "green": count_g}
    best_color = max(counts, key=counts.get)
    best_count = counts[best_color]

    if best_count / total < 0.04:
        return "unknown", 0.0

    confidence = min(best_count / total * 5, 1.0)
    return best_color, round(confidence, 3)


def run_detection(image_bgr: np.ndarray, model) -> dict:
    """
    YOLOv11で信号機を検出 → OpenCVで色判別
    """
    results = {
        "color": "none",
        "confidence_yolo": 0.0,
        "confidence_color": 0.0,
        "boxes": [],
        "annotated": image_bgr.copy(),
    }

    if model is None:
        return results

    yolo_results = model(image_bgr, verbose=False)[0]

    best_conf = 0.0
    best_box  = None
    best_color = "none"
    best_color_conf = 0.0

    for box in yolo_results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names.get(cls_id, "")

        # traffic light クラスのみ対象
        if "traffic" not in label.lower() and "signal" not in label.lower() and cls_id != 9:
            # COCO cls=9 は "traffic light"
            if cls_id != 9:
                continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = image_bgr[y1:y2, x1:x2]
        color, color_conf = analyze_traffic_light_color(roi)

        results["boxes"].append({
            "bbox": (x1, y1, x2, y2),
            "yolo_conf": conf,
            "color": color,
            "color_conf": color_conf,
        })

        if conf > best_conf:
            best_conf       = conf
            best_box        = (x1, y1, x2, y2)
            best_color      = color
            best_color_conf = color_conf

    results["color"]            = best_color
    results["confidence_yolo"]  = round(best_conf, 3)
    results["confidence_color"] = best_color_conf

    # アノテーション描画
    annotated = image_bgr.copy()
    color_map = {
        "red":     (0, 0, 255),
        "yellow":  (0, 220, 255),
        "green":   (0, 230, 100),
        "unknown": (160, 160, 160),
        "none":    (160, 160, 160),
    }
    for b in results["boxes"]:
        x1, y1, x2, y2 = b["bbox"]
        c = b["color"]
        rgb = color_map.get(c, (160, 160, 160))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), rgb, 3)
        label_text = f"{c.upper()} {b['yolo_conf']:.0%}"
        cv2.putText(annotated, label_text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, rgb, 2)

    results["annotated"] = annotated
    return results


def color_to_japanese(color: str) -> tuple[str, str, str, str]:
    """色 → (日本語, emoji, cssクラス, 音声メッセージ)"""
    mapping = {
        "red":     ("赤 — 止まれ",    "🔴", "signal-red",    "赤信号です。止まってください。"),
        "yellow":  ("黄 — 注意",      "🟡", "signal-yellow", "黄信号です。注意してください。"),
        "green":   ("青 — 進め",      "🟢", "signal-green",  "青信号です。渡れます。"),
        "unknown": ("判別不明",        "⚪", "signal-none",   "信号の色を判別できませんでした。"),
        "none":    ("信号機未検出",    "⚫", "signal-none",   "信号機が見つかりませんでした。"),
    }
    return mapping.get(color, mapping["none"])


def tts_html(message: str) -> str:
    """Web Speech API を使った音声読み上げHTML"""
    safe = message.replace("'", "\\'")
    return f"""
    <script>
    (function(){{
        if (!window._tts_done_{abs(hash(message))}) {{
            window._tts_done_{abs(hash(message))} = true;
            var u = new SpeechSynthesisUtterance('{safe}');
            u.lang = 'ja-JP';
            u.rate = 0.95;
            u.pitch = 1.1;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        }}
    }})();
    </script>
    """


# ─── メインUI ────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚦 信号機判別アプリ</h1>
  <p>YOLOv11 + OpenCV で信号機を検出・色判別し、音声でお知らせします</p>
</div>
""", unsafe_allow_html=True)

# モデル読み込み（キャッシュ）
@st.cache_resource(show_spinner="モデルを読み込み中…")
def get_model():
    return load_yolo_model()

model = get_model()

# タブ構成
tab2, tab1, tab3 = st.tabs(["📹 カメラ撮影", "📷 画像アップロード", "ℹ️ 使い方"])

# ── タブ1: 画像アップロード ────────────────────────────
with tab1:
    uploaded = st.file_uploader(
        "信号機の画像をアップロード",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 検出中…"):
            result = run_detection(img_bgr, model)

        color = result["color"]
        ja_label, emoji, css_cls, voice_msg = color_to_japanese(color)

        # 結果表示
        st.markdown(f"""
        <div class="signal-box {css_cls}">
            {emoji}
            <div class="label-{color if color in ['red','yellow','green'] else 'none'}">{ja_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # 信頼度バー
        if result["confidence_yolo"] > 0:
            yc  = int(result["confidence_yolo"]  * 100)
            cc  = int(result["confidence_color"] * 100)
            col_map = {"red":"#FF3B3B","yellow":"#FFD600","green":"#00E676"}
            bar_color = col_map.get(color, "#6B6B85")
            st.markdown(f"""
            <div class="conf-bar-wrap">
                <div class="conf-label">YOLO検出信頼度: {yc}%</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{yc}%;background:{bar_color}"></div></div>
            </div>
            <div class="conf-bar-wrap" style="margin-top:8px">
                <div class="conf-label">色判別信頼度: {cc}%</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{cc}%;background:{bar_color}"></div></div>
            </div>
            """, unsafe_allow_html=True)

        # 音声読み上げ
        st.components.v1.html(tts_html(voice_msg), height=0)

        # アノテーション済み画像
        with st.expander("🖼️ 検出結果画像を表示"):
            annotated_rgb = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

    else:
        st.markdown("""
        <div class="info-card" style="text-align:center;padding:2rem;">
            📂 画像ファイルをアップロードしてください<br>
            <span style="font-size:0.8rem">対応形式: JPG / PNG / WEBP</span>
        </div>
        """, unsafe_allow_html=True)

# ── タブ2: カメラ ─────────────────────────────────────
with tab2:
    # HTML5ネイティブカメラ（背面・大画面）＋ Streamlitへ画像を渡す
    st.components.v1.html("""
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body { background: #0A0A0F; }
      #wrapper {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 12px;
        padding: 8px 0;
      }
      video {
        width: 100%;
        max-height: 62vh;
        object-fit: cover;
        border-radius: 14px;
        border: 2px solid #2A2A3A;
        background: #000;
        display: block;
      }
      canvas { display: none; }
      #snapBtn {
        width: 100%;
        padding: 18px;
        font-size: 1.3rem;
        font-weight: 800;
        background: #12121A;
        color: #00E676;
        border: 2px solid #00E676;
        border-radius: 14px;
        cursor: pointer;
        font-family: 'Noto Sans JP', sans-serif;
        letter-spacing: 1px;
      }
      #snapBtn:active { background: #00E676; color: #0A0A0F; }
      #msg {
        color: #6B6B85;
        font-size: 0.82rem;
        text-align: center;
        font-family: sans-serif;
      }
    </style>
    <div id="wrapper">
      <video id="video" autoplay playsinline></video>
      <canvas id="canvas"></canvas>
      <button id="snapBtn">📸 撮影する</button>
      <div id="msg">背面カメラで信号機を撮影してください</div>
    </div>
    <script>
      const video   = document.getElementById('video');
      const canvas  = document.getElementById('canvas');
      const snapBtn = document.getElementById('snapBtn');
      const msg     = document.getElementById('msg');

      // 背面カメラで起動
      navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }
      }).then(stream => {
        video.srcObject = stream;
        msg.textContent = '📷 背面カメラ起動中';
      }).catch(err => {
        msg.textContent = '⚠️ カメラを起動できませんでした: ' + err.message;
      });

      // 撮影 → Base64 → Streamlit に送信
      snapBtn.addEventListener('click', () => {
        canvas.width  = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
        // Streamlit の parent frame に postMessage で送る
        window.parent.postMessage({ type: 'camera_snap', image: dataUrl }, '*');
        msg.textContent = '✅ 送信しました！少々お待ちください…';
        setTimeout(() => { msg.textContent = '📷 背面カメラ起動中'; }, 3000);
      });
    </script>
    """, height=600, scrolling=False)

    # postMessage を受け取るブリッジ（隠しinput経由）
    st.markdown("""
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'camera_snap') {
            // hidden input に base64 を入れて Streamlit のfile_uploaderをトリガー
            const input = window.parent.document.querySelector('input[data-testid="stFileUploaderDropzoneInput"]');
            if (input) {
                const arr = e.data.image.split(',');
                const mime = arr[0].match(/:(.*?);/)[1];
                const bstr = atob(arr[1]);
                let n = bstr.length;
                const u8arr = new Uint8Array(n);
                while(n--) { u8arr[n] = bstr.charCodeAt(n); }
                const file = new File([u8arr], 'camera.jpg', {type: mime});
                const dt = new DataTransfer();
                dt.items.add(file);
                input.files = dt.files;
                input.dispatchEvent(new Event('change', {bubbles: true}));
            }
        }
    });
    </script>
    """, unsafe_allow_html=True)

    st.markdown("<div style='color:#6B6B85;font-size:0.82rem;text-align:center;margin-top:-8px'>↓ または画像ファイルを直接アップロード</div>", unsafe_allow_html=True)
    camera_img = st.file_uploader("カメラ画像", type=["jpg","jpeg","png"], label_visibility="collapsed", key="cam_upload")

    if camera_img:
        file_bytes = np.asarray(bytearray(camera_img.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 検出中…"):
            result = run_detection(img_bgr, model)

        color = result["color"]
        ja_label, emoji, css_cls, voice_msg = color_to_japanese(color)

        st.markdown(f"""
        <div class="signal-box {css_cls}">
            {emoji}
            <div class="label-{color if color in ['red','yellow','green'] else 'none'}">{ja_label}</div>
        </div>
        """, unsafe_allow_html=True)

        st.components.v1.html(tts_html(voice_msg), height=0)

        if result["confidence_yolo"] > 0:
            yc = int(result["confidence_yolo"] * 100)
            cc = int(result["confidence_color"] * 100)
            col_map = {"red":"#FF3B3B","yellow":"#FFD600","green":"#00E676"}
            bar_color = col_map.get(color, "#6B6B85")
            st.markdown(f"""
            <div class="conf-bar-wrap">
                <div class="conf-label">YOLO検出信頼度: {yc}%</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{yc}%;background:{bar_color}"></div></div>
            </div>
            <div class="conf-bar-wrap" style="margin-top:8px">
                <div class="conf-label">色判別信頼度: {cc}%</div>
                <div class="conf-bar"><div class="conf-fill" style="width:{cc}%;background:{bar_color}"></div></div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🖼️ 検出結果画像を表示"):
            annotated_rgb = cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

# ── タブ3: 使い方 ──────────────────────────────────────
with tab3:
    st.markdown("""
    <div class="info-card">
        <strong>📱 使い方</strong><br><br>
        1. <strong>画像アップロード</strong> タブから信号機の写真をアップロード<br>
        2. または <strong>カメラ撮影</strong> タブでその場で撮影<br>
        3. 自動で信号機を検出・色を判別し、<strong>音声で読み上げ</strong>ます<br><br>
        🔴 赤信号 → 「赤信号です。止まってください。」<br>
        🟡 黄信号 → 「黄信号です。注意してください。」<br>
        🟢 青信号 → 「青信号です。渡れます。」
    </div>

    <div class="info-card" style="margin-top:0.8rem">
        <strong>⚙️ 技術構成</strong><br><br>
        • <strong>物体検出</strong>: YOLOv11 (Ultralytics)<br>
        • <strong>色判別</strong>: OpenCV (HSV色空間解析)<br>
        • <strong>音声</strong>: Web Speech API (ブラウザ内蔵)<br>
        • <strong>UI</strong>: Streamlit
    </div>

    <div class="info-card" style="margin-top:0.8rem">
        <strong>📦 カスタムモデルの使用方法</strong><br><br>
        独自学習した <code>yolo11_traffic.pt</code> を<br>
        <code>models/</code> フォルダに配置してください。<br><br>
        推奨データセット:<br>
        • <a href="https://universe.roboflow.com/search?q=traffic+light" style="color:#00E676">Roboflow Universe — Traffic Light</a>
    </div>
    """, unsafe_allow_html=True)

# フッター
st.markdown("""
<hr style="border-color:#2A2A3A;margin-top:2rem">
<div style="text-align:center;color:#6B6B85;font-size:0.8rem;padding-bottom:1rem">
    視覚障害者支援ツール | Built with YOLOv11 + OpenCV + Streamlit
</div>
""", unsafe_allow_html=True)
