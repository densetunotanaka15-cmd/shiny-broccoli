# 🚦 信号機判別支援アプリ（改良版）

YOLO11 を **COCO TrafficLight データセットでファインチューニング**し、  
検出した信号機領域に対して **OpenCV HSV 解析**で色を判定する、視覚障害者支援アプリです。

---

## ✨ 改良点（旧バージョンとの差分）

| 項目 | 旧バージョン | 改良版 |
|------|------------|--------|
| 検出モデル | YOLO11 汎用重み | COCO TrafficLight Fine-tuned |
| 色解析対象 | 画像全体 | 検出 BBox 内のみ（精度向上） |
| 前処理 | なし | CLAHE（夜間・逆光対応） |
| 誤検出抑制 | なし | 明度・彩度マスクで点灯部位に限定 |
| 信頼度表示 | なし | あり（% 表示） |

---

## 🗂 ファイル構成

```
shiny-broccoli/
├── prepare_coco.py   # COCOデータセットから TrafficLight を抽出・変換
├── dataset.yaml      # YOLO学習用データセット設定
├── train.py          # YOLO11 ファインチューニングスクリプト
├── app.py            # Streamlit アプリ本体
├── requirements.txt  # 依存ライブラリ
└── README.md
```

---

## 🚀 セットアップ手順

### 1. リポジトリをクローン

```bash
git clone https://github.com/densetunotanaka15-cmd/shiny-broccoli.git
cd shiny-broccoli
```

### 2. 依存ライブラリをインストール

```bash
pip install -r requirements.txt
```

### 3. COCO データセットを準備

[COCO Dataset](https://cocodataset.org/#download) から以下をダウンロードし、展開してください。

- `train2017.zip`（画像）
- `val2017.zip`（画像）
- `annotations_trainval2017.zip`（アノテーション）

```
coco/
  images/
    train2017/
    val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

TrafficLight のみを抽出して YOLO 形式に変換します。

```bash
python prepare_coco.py \
    --coco_dir /path/to/coco \
    --output_dir ./data/coco_trafficlight
```

### 4. ファインチューニング

```bash
python train.py
```

完了すると `runs/train/trafficlight_v1/weights/best.pt` が生成されます。  
GPU がない場合は `train.py` 内の `DEVICE = "cpu"` に変更してください。

> **目安:** NVIDIA RTX 3060（12GB）で 50 エポック約 2〜3 時間

### 5. アプリを起動

```bash
streamlit run app.py
```

---

## 🔊 音声読み上げ内容

| 判定結果 | 読み上げ |
|---------|---------|
| 青信号  | 「青信号です。進めます。」 |
| 赤信号  | 「赤信号です。止まってください。」 |
| 黄信号  | 「黄信号です。注意してください。」 |
| 未検出  | 「信号機が見つかりませんでした。正面から大きく写してください。」 |
| 判定不能 | 「信号機は検出されましたが、色を判定できませんでした。」 |

---

## 📦 使用技術

| ライブラリ | 用途 |
|-----------|------|
| Streamlit | Web アプリ UI |
| Ultralytics YOLO11 | 信号機検出（Fine-tuned） |
| OpenCV | HSV 色判定・CLAHE 前処理 |
| Pillow | 画像処理 |
| Web Speech API | 音声読み上げ（ブラウザ標準） |

---

## 🛠 モデルのチューニング目安

`train.py` の以下の値を調整することで精度・速度のバランスを変えられます。

```python
MODEL_BASE  = "yolo11n.pt"   # n→s→m→l→x で精度向上（速度は低下）
EPOCHS      = 50             # データが多い場合は 100 程度まで増やす
BATCH_SIZE  = 16             # VRAM 不足時は 8 に下げる
CONF_THRESH = 0.40           # app.py 側：低くすると検出増・誤検出も増
```

---

## ⚠️ 注意事項

本アプリは補助的なツールです。  
必ず周囲の音・誘導鈴・歩行者用信号の音を確認して安全を確保してください。
