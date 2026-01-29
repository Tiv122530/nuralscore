"""
AI Vision フォールバックモジュール
OpenAI GPT-4 Vision / Google Geminiでの画像認識と低信頼度領域の修正
"""

import base64
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json

from openai import OpenAI
import google.generativeai as genai
import numpy as np
import cv2
from PIL import Image

logger = logging.getLogger(__name__)


class AIVisionProcessor:
    """OpenAI GPT-4 Vision / Google Geminiを使った画像認識クラス"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None):
        """
        Args:
            api_key: APIキー（Noneの場合は環境変数から取得）
            model: 使用するモデル（Noneの場合は環境変数から取得）
            provider: プロバイダー (openai/gemini、Noneの場合は環境変数から取得)
        """
        self.provider = provider or os.getenv("AI_PROVIDER", "openai")
        
        if self.provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEYが設定されていません")
            
            self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logger.info(f"Gemini クライアント初期化完了 (モデル: {self.model})")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEYが設定されていません")
            
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI クライアント初期化完了 (モデル: {self.model})")
    
    def analyze_scorecard(self, image_path: str, course_name: Optional[str] = None) -> Dict:
        """
        スコアカード全体をAI Visionで解析
        
        Args:
            image_path: 画像ファイルのパス
            course_name: ゴルフ場名（ヒントとして提供）
            
        Returns:
            {
                'players': [{'name': str, 'scores': [...]}],
                'unknown': [str],
                'confidence': float,
                'raw_response': str
            }
        """
        logger.info(f"AI Vision解析開始: {image_path}")
        
        if self.provider == "gemini":
            return self._analyze_with_gemini(image_path)
        else:
            return self._analyze_with_openai(image_path)
    
    def _analyze_with_gemini(self, image_path: str) -> Dict:
        """Gemini APIで解析"""
        try:
            # 画像を読み込み
            img = Image.open(image_path)
            
            # プロンプト
            prompt = """You are an OCR assistant for golf scorecards written in Japanese.
Extract players and per-hole scores from the provided scorecard image.
Return ONLY valid JSON matching this schema:
{
  "players": [
    {
      "name": string,
      "scores": [
        {
          "hole": integer,
          "strokes": integer,
          "putts": integer|null
        }
      ]
    }
  ],
  "unknown": [string],
  "confidence": number
}

CRITICAL Rules for Japanese Names:
- Player names are written in Japanese (Kanji, Hiragana, or Katakana).
- Read Japanese characters VERY CAREFULLY. Look at each stroke and component.
- Common Japanese surnames: 田中、佐藤、鈴木、高橋、渡辺、伊藤、山本、中村、小林、加藤
- If handwritten, consider similar-looking kanji (e.g., 土/士, 未/末, 己/巳).
- Extract the FULL name exactly as written, not just partial characters.
- If unsure about a character, include it in "unknown" with details.

Other Rules:
- Always return JSON, no markdown.
- Hole numbers must be 1-18.
- If a value is missing or unreadable, use null and add a note to unknown.
- If player name is completely unreadable, use "Player 1", "Player 2", etc.

Read the scorecard and return JSON only."""
            
            # API呼び出し
            response = self.client.generate_content([prompt, img])
            result_text = response.text
            
            logger.info("Gemini解析完了")
            logger.debug(f"Gemini生レスポンス: {result_text[:500]}...")
            
            # JSON部分を抽出
            result_json = self._extract_json(result_text)
            
            return {
                'players': result_json.get('players', []),
                'unknown': result_json.get('unknown', []),
                'confidence': result_json.get('confidence', 0.95),
                'raw_response': result_text
            }
            
        except Exception as e:
            logger.error(f"Gemini APIエラー: {e}")
            return {
                'players': [],
                'unknown': [f"API Error: {str(e)}"],
                'confidence': 0.0,
                'raw_response': str(e)
            }
    
    def _analyze_with_openai(self, image_path: str) -> Dict:
        """OpenAI APIで解析"""
        # 画像をBase64エンコード
        image_base64 = self._encode_image(image_path)
        
        # API呼び出し
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an OCR assistant for golf scorecards.
Extract players and per-hole scores from the provided scorecard image.
Return ONLY valid JSON matching this schema:
{
  "players": [
    {
      "name": string,
      "scores": [
        {
          "hole": integer,
          "strokes": integer,
          "putts": integer|null
        }
      ]
    }
  ],
  "unknown": [string],
  "confidence": number
}

Rules:
- Always return JSON, no markdown.
- Hole numbers must be 1-18.
- If a value is missing or unreadable, use null and add a note to unknown.
- If player name is missing, use "Player 1", "Player 2", etc."""
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Read the scorecard and return JSON only."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=2000
            )
            
            result_text = response.choices[0].message.content
            logger.info("AI Vision解析完了")
            logger.debug(f"AI Vision生レスポンス: {result_text[:500]}...")  # 最初の500文字をログ
            
            # JSON部分を抽出
            result_json = self._extract_json(result_text)
            
            return {
                'players': result_json.get('players', []),
                'unknown': result_json.get('unknown', []),
                'confidence': result_json.get('confidence', 0.95),
                'raw_response': result_text
            }
            
        except Exception as e:
            logger.error(f"AI Vision APIエラー: {e}")
            return {
                'players': [],
                'unknown': [f"API Error: {str(e)}"],
                'confidence': 0.0,
                'raw_response': str(e)
            }
    
    def verify_region(self, image: np.ndarray, bbox: List, expected_type: str = "number") -> Dict:
        """
        特定の領域を AI Vision で検証（低信頼度領域の再確認用）
        
        Args:
            image: 画像（OpenCV形式）
            bbox: 境界ボックス [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            expected_type: 期待するテキストのタイプ（'number', 'name'）
            
        Returns:
            {
                'text': str,
                'confidence': float
            }
        """
        # 領域を切り出し
        cropped = self._crop_region(image, bbox)
        
        # 一時ファイルに保存
        temp_path = Path("./output/temp_crop.jpg")
        temp_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(temp_path), cropped)
        
        # Base64エンコード
        image_base64 = self._encode_image(str(temp_path))
        
        # プロンプト作成
        if expected_type == "number":
            prompt = "この画像に写っている数字を読み取ってください。数字のみを返してください。"
        else:
            prompt = "この画像に写っているテキストを読み取ってください。テキストのみを返してください。"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=50
            )
            
            text = response.choices[0].message.content.strip()
            
            # 一時ファイル削除
            temp_path.unlink()
            
            return {
                'text': text,
                'confidence': 0.9
            }
            
        except Exception as e:
            logger.error(f"AI Vision領域検証エラー: {e}")
            return {
                'text': "",
                'confidence': 0.0
            }
    
    def _encode_image(self, image_path: str) -> str:
        """画像をBase64エンコード"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _crop_region(self, image: np.ndarray, bbox: List) -> np.ndarray:
        """境界ボックスで画像を切り出し"""
        # bboxから矩形領域を計算
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        cropped = image[y_min:y_max, x_min:x_max]
        return cropped
    
    def _build_scorecard_prompt(self, course_name: Optional[str] = None) -> str:
        """スコアカード解析用のプロンプトを構築"""
        prompt = """この画像はゴルフのスコアカードです。以下の情報を読み取ってJSON形式で返してください：

1. プレイヤー名
2. 各ホール（1-18）のスコア（打数とパット数）

JSON形式:
{
  "players": [
    {
      "name": "プレイヤー名",
      "scores": [
        {"hole": 1, "strokes": 打数, "putts": パット数},
        {"hole": 2, "strokes": 打数, "putts": パット数},
        ...
      ]
    }
  ]
}

"""
        
        if course_name:
            prompt += f"\nゴルフ場名: {course_name}\n"
        
        prompt += "\n注意: 手書きの数字を正確に読み取ってください。不明な場合はnullを返してください。"
        
        return prompt
    
    def _extract_json(self, text: str) -> Dict:
        """テキストからJSON部分を抽出"""
        try:
            # ```json ... ``` で囲まれている場合
            if "```json" in text:
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "```" in text:
                start = text.find("```") + 3
                end = text.find("```", start)
                json_str = text[start:end].strip()
            else:
                # JSON部分を探す
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"JSON抽出エラー: {e}")
            return {}


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    # 環境変数チェック
    if not os.getenv("OPENAI_API_KEY"):
        print("警告: OPENAI_API_KEYが設定されていません")
        print(".envファイルを作成してAPIキーを設定してください")
    else:
        processor = AIVisionProcessor()
        
        # サンプル画像で実行
        if Path("IMG_1873.jpg").exists():
            result = processor.analyze_scorecard("IMG_1873.jpg")
            
            print("\n=== AI Vision解析結果 ===")
            print(f"プレイヤー数: {len(result['players'])}")
            print(f"信頼度: {result['confidence']:.2f}")
            
            for i, player in enumerate(result['players'], 1):
                print(f"\nプレイヤー {i}: {player.get('name', '不明')}")
                scores = player.get('scores', [])
                print(f"  スコア数: {len(scores)}")
        else:
            print("IMG_1873.jpgが見つかりません")
