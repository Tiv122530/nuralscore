# ハイブリッド型スコアカードOCRシステム

紙のゴルフスコアカードをデジタル化するハイブリッド型OCRシステムです。

## 特徴

- 📸 **画像前処理**: 回転補正、台形補正、適応的二値化で精度向上
- 🔍 **OCR処理**: EasyOCRで高速文字認識
- 🤖 **AI Vision**: 低信頼度領域をGPT-4 Visionで再確認
- ✅ **データベース照合**: gdo.sqliteのパー情報と照合して妥当性検証
- 📊 **JSON出力**: 指定形式のJSONで結果を出力

## システム構成

```
┌─────────────┐
│ スコアカード │
│   画像入力   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 画像前処理   │  ← image_preprocessor.py
│ - 回転補正   │
│ - 台形補正   │
│ - 二値化     │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ OCR処理     │  ← ocr_processor.py
│ (EasyOCR)   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 信頼度チェック│
└──────┬──────┘
       │
       ├─ 高信頼度 ─────────┐
       │                    │
       └─ 低信頼度 ─┐        │
                    ▼        │
              ┌──────────┐  │
              │ AI Vision │  │
              │ 再確認    │  │
              └────┬─────┘  │
                    │        │
                    └────────┤
                             ▼
                    ┌─────────────┐
                    │ DB照合・検証 │  ← db_validator.py
                    │ (gdo.sqlite) │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ JSON変換    │  ← json_formatter.py
                    │ 出力        │
                    └─────────────┘
```

## セットアップ

### 1. 依存関係のインストール

```powershell
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example`をコピーして`.env`を作成：

```powershell
Copy-Item .env.example .env
```

`.env`ファイルを編集してOpenAI APIキーを設定：

```
OPENAI_API_KEY=sk-your-api-key-here
DEBUG_MODE=True
OUTPUT_DIR=./output
OCR_CONFIDENCE_THRESHOLD=0.7
```

## 使い方

### 基本的な使用方法

```powershell
python main.py
```

デフォルトでは`IMG_1873.jpg`を処理します。

### プログラムから使用

```python
from main import HybridScorecardOCR

# システム初期化
ocr_system = HybridScorecardOCR(
    debug_mode=True,        # 中間画像を保存
    use_ai_vision=True      # AI Vision使用
)

# スコアカード処理
result = ocr_system.process_scorecard(
    image_path="scorecard.jpg",
    course_id="112106",     # ゴルフ場ID（オプション）
    asset_id=27,
    image_url="http://example.com/image.jpg"
)

print(result)
```

## 各モジュールのテスト

### 画像前処理のテスト

```powershell
python image_preprocessor.py
```

中間画像が`output/`ディレクトリに保存されます。

### OCR処理のテスト

```powershell
python ocr_processor.py
```

### データベース照合のテスト

```powershell
python db_validator.py
```

### AI Visionのテスト

```powershell
python ai_vision.py
```

## 出力形式

```json
{
  "assetId": 27,
  "imageUrl": "http://minio:9000/golf-score/ocr/sample.jpg",
  "rawText": null,
  "aiResult": {
    "players": [
      {
        "name": "佐藤太郎",
        "scores": [
          {"hole": 1, "strokes": 6, "putts": 2},
          {"hole": 2, "strokes": 5, "putts": 1},
          ...
        ]
      }
    ],
    "unknown": [],
    "confidence": 0.95
  }
}
```

## トラブルシューティング

### EasyOCRのモデルダウンロードエラー

初回実行時にモデルをダウンロードします。ネットワーク接続を確認してください。

### OpenAI APIエラー

- APIキーが正しく設定されているか確認
- API使用量の上限をチェック

### 画像前処理が失敗する

- 画像が横向きの場合、EXIF情報で自動回転します
- スコアカードが検出できない場合、台形補正はスキップされます

## ファイル構成

```
scoreocr/
├── main.py                 # メインパイプライン
├── image_preprocessor.py   # 画像前処理
├── ocr_processor.py        # OCR処理
├── db_validator.py         # データベース照合
├── ai_vision.py           # AI Vision処理
├── json_formatter.py      # JSON変換
├── requirements.txt       # 依存関係
├── .env.example          # 環境変数サンプル
├── gdo.sqlite            # ゴルフ場データベース
├── IMG_1873.jpg          # サンプル画像
└── output/               # 出力ディレクトリ
    ├── 01_original.jpg   # 元画像
    ├── 02_rotated.jpg    # 回転補正後
    ├── 03_perspective.jpg # 台形補正後
    ├── 04_gray.jpg       # グレースケール
    ├── 05_binary.jpg     # 二値化
    ├── 06_denoised.jpg   # ノイズ除去
    └── result.json       # 最終結果
```

## パフォーマンス

- **OCRのみ**: 0.5-2秒/画像
- **ハイブリッド（AI併用）**: 2-5秒/画像
- **精度**: 90-98%（明瞭な手書き）

## コスト

- **OCRのみ**: 無料
- **ハイブリッド**: $0.003-0.01/画像（AI呼び出しを最小化）
- **AI Visionのみ**: $0.01-0.03/画像

## お問い合わせ

商用利用や質問については以下までご連絡ください：

📧 muzui122530@gmail.com

