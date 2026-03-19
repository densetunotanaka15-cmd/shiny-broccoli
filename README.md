# 🚦 視覚障害者向け 信号機判別ウェブアプリ

**YOLOv11** で信号機を検出し、**OpenCV** で赤・黄・青を判別して **音声** でお知らせするアクセシビリティアプリです。

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ 機能

| 機能 | 説明 |
|------|------|
| 🔍 **物体検出** | YOLOv11 で画像内の信号機を自動検出 |
| 🎨 **色判別** | OpenCV (HSV色空間) で赤・黄・青を高精度判別 |
| 🔊 **音声読み上げ** | Web Speech API でブラウザが日本語音声を再生 |
| 📷 **カメラ対応** | スマートフォンカメラで直接撮影・判別 |
| 📊 **信頼度表示** | 検出・色判別の信頼度をビジュアル表示 |

---

## 🚀 Streamlit Cloud へのデプロイ

### 1. このリポジトリをフォーク or クローン

```bash
git clone https://github.com/YOUR_USERNAME/traffic-signal-detector.git
cd traffic-signal-detector
```

### 2. Streamlit Cloud で公開

1. [share.streamlit.io](https://share.streamlit.io) にアクセス
2. `New app` → GitHubリポジトリを選択
3. `Main file path`: `app.py` を指定
4. `Deploy!` をクリック

> **モデルについて**: デフォルトでは `yolo11n.pt`（COCO汎用モデル）を使用します。  
> 信号機専用モデルを使う場合は [カスタムモデルの準備](#カスタムモデルの準備) を参照してください。

---

## 💻 ローカル実行

```bash
# 依存パッケージをインストール
pip install -r requirements.txt

# アプリを起動
streamlit run app.py
```

---

## 📁 ディレクトリ構成

```
traffic-signal-detector/
├── app.py                  # Streamlit メインアプリ
├── requirements.txt        # 依存パッケージ
├── README.md               # このファイル
├── models/
│   └── yolo11_traffic.pt   # カスタムモデル（任意）
└── .streamlit/
    └── config.toml         # Streamlit 設定
```

---

## 🔧 技術構成

```
入力画像
   │
   ▼
YOLOv11 (ultralytics)
   │  ← 信号機のバウンディングボックスを検出
   ▼
ROI (信号機領域) 切り出し
   │
   ▼
OpenCV HSV色空間解析
   │  ← 赤/黄/青マスクでピクセル数比較
   ▼
色判別結果
   │
   ▼
Web Speech API
   └→ 「赤信号です。止まってください。」
```

### 色判別 HSV範囲

| 色 | H (色相) | S (彩度) | V (明度) |
|----|---------|---------|---------|
| 赤 | 0–10 / 160–180 | 120–255 | 70–255 |
| 黄 | 20–35 | 100–255 | 100–255 |
| 青 | 40–90 | 80–255 | 80–255 |

---

## 📦 カスタムモデルの準備

より高精度な判別には、信号機専用データセットで学習したモデルを使用してください。

### 推奨データセット

- [Roboflow Universe — Traffic Light](https://universe.roboflow.com/search?q=traffic+light)
- [LISA Traffic Light Dataset](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset)
- [Bosch Small Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/content/bosch-small-traffic-lights-dataset)

### YOLOv11 でファインチューニング

```python
from ultralytics import YOLO

# ベースモデルを読み込む
model = YOLO("yolo11n.pt")

# 学習実行
results = model.train(
    data="traffic_light.yaml",   # データセット設定ファイル
    epochs=100,
    imgsz=640,
    batch=16,
    name="yolo11_traffic",
)

# 学習済みモデルを models/ に配置
# runs/detect/yolo11_traffic/weights/best.pt → models/yolo11_traffic.pt
```

### データセット設定ファイル例 (`traffic_light.yaml`)

```yaml
path: ./dataset
train: images/train
val:   images/val

nc: 3  # クラス数
names:
  0: red
  1: yellow
  2: green
```

---

## ♿ アクセシビリティへの配慮

- **大きく明確な表示**: 色名と絵文字で視認しやすく表示
- **音声読み上げ**: 画像を見なくても結果がわかる
- **高コントラストUI**: ダークテーマ + 信号色の明確な配色
- **スマートフォン対応**: カメラ入力で外出先でも使用可能

---

## 📄 ライセンス

MIT License

---

## 🙏 使用ライブラリ

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Streamlit](https://streamlit.io/)

