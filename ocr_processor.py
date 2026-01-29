"""
OCR処理モジュール
EasyOCRを使用した文字認識と信頼度スコア取得
"""

import easyocr
import numpy as np
from typing import List, Dict, Tuple
import logging
import re

logger = logging.getLogger(__name__)


class OCRProcessor:
    """EasyOCRを使った文字認識クラス"""
    
    def __init__(self, languages: List[str] = ['ja', 'en']):
        """
        Args:
            languages: 認識する言語リスト（日本語と英語）
        """
        logger.info(f"EasyOCR初期化中... 言語: {languages}")
        self.reader = easyocr.Reader(languages, gpu=False)
        logger.info("EasyOCR初期化完了")
    
    def extract_text(self, image: np.ndarray, detail: int = 1) -> List[Tuple]:
        """
        画像から文字を抽出
        
        Args:
            image: 入力画像（OpenCV形式）
            detail: 詳細レベル（1=境界ボックス+テキスト+信頼度）
            
        Returns:
            [(bbox, text, confidence), ...] のリスト
        """
        logger.info("OCR処理開始")
        results = self.reader.readtext(image, detail=detail)
        logger.info(f"OCR処理完了: {len(results)}個のテキスト検出")
        
        return results
    
    def extract_scorecard_data(self, image: np.ndarray) -> Dict:
        """
        スコアカード特化のデータ抽出
        
        Returns:
            {
                'players': [{'name': str, 'confidence': float}, ...],
                'scores': {
                    'player_index': {
                        'hole_number': {'strokes': int, 'putts': int, 'confidence': float}
                    }
                },
                'raw_ocr_results': [...],
                'average_confidence': float
            }
        """
        results = self.extract_text(image)
        
        # OCR結果を分類
        players = self._extract_player_names(results)
        scores = self._extract_scores(results)
        
        # 平均信頼度を計算
        confidences = [conf for _, _, conf in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'players': players,
            'scores': scores,
            'raw_ocr_results': results,
            'average_confidence': avg_confidence
        }
    
    def _extract_player_names(self, ocr_results: List[Tuple]) -> List[Dict]:
        """
        プレイヤー名を抽出
        上部に配置されている日本語テキストを名前と推定
        """
        players = []
        
        for bbox, text, confidence in ocr_results:
            # 日本語を含むテキスト
            if self._contains_japanese(text):
                # 境界ボックスのY座標（上部にあるか）
                y_position = bbox[0][1]  # 左上のY座標
                
                players.append({
                    'name': text,
                    'confidence': confidence,
                    'y_position': y_position
                })
        
        # Y座標でソート（上から順）
        players.sort(key=lambda x: x['y_position'])
        
        # Y座標情報を削除
        for player in players:
            del player['y_position']
        
        logger.info(f"プレイヤー名抽出: {len(players)}名")
        return players
    
    def _extract_scores(self, ocr_results: List[Tuple]) -> Dict:
        """
        スコアを抽出
        数字のテキストを位置関係から分類
        """
        scores = {}
        
        for bbox, text, confidence in ocr_results:
            # 数字のみのテキスト
            if self._is_number(text):
                x_position = bbox[0][0]  # 左上のX座標
                y_position = bbox[0][1]  # 左上のY座標
                
                # 簡易的な分類（実際にはより高度な位置解析が必要）
                scores[f"{x_position}_{y_position}"] = {
                    'text': text,
                    'confidence': confidence,
                    'x': x_position,
                    'y': y_position
                }
        
        logger.info(f"スコア抽出: {len(scores)}個")
        return scores
    
    def _contains_japanese(self, text: str) -> bool:
        """テキストに日本語（ひらがな、カタカナ、漢字）が含まれるか"""
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        return bool(japanese_pattern.search(text))
    
    def _is_number(self, text: str) -> bool:
        """テキストが数字のみか"""
        return text.strip().isdigit()
    
    def get_low_confidence_regions(self, ocr_results: List[Tuple], threshold: float = 0.7) -> List[Dict]:
        """
        信頼度が低い領域を抽出（AI Visionで再確認が必要な部分）
        
        Args:
            ocr_results: OCR結果
            threshold: 信頼度閾値
            
        Returns:
            低信頼度領域のリスト
        """
        low_confidence = []
        
        for bbox, text, confidence in ocr_results:
            if confidence < threshold:
                low_confidence.append({
                    'bbox': bbox,
                    'text': text,
                    'confidence': confidence
                })
        
        logger.info(f"低信頼度領域: {len(low_confidence)}個 (閾値: {threshold})")
        return low_confidence


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    import cv2
    
    # サンプル画像で実行
    image = cv2.imread("IMG_1873.jpg")
    
    if image is not None:
        processor = OCRProcessor()
        results = processor.extract_scorecard_data(image)
        
        print(f"\n=== プレイヤー名 ===")
        for player in results['players']:
            print(f"  {player['name']} (信頼度: {player['confidence']:.2f})")
        
        print(f"\n=== 統計 ===")
        print(f"平均信頼度: {results['average_confidence']:.2f}")
        print(f"検出テキスト数: {len(results['raw_ocr_results'])}")
        
        # 低信頼度領域
        low_conf = processor.get_low_confidence_regions(results['raw_ocr_results'])
        print(f"低信頼度領域: {len(low_conf)}個")
    else:
        print("画像ファイルが見つかりません")
