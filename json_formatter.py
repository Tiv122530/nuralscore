"""
JSON変換モジュール
抽出データを指定されたJSON形式に変換
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class JSONFormatter:
    """スコアデータをJSON形式に変換するクラス"""
    
    @staticmethod
    def format_result(
        asset_id: int,
        image_url: str,
        players_data: List[Dict],
        raw_text: Optional[str] = None,
        confidence: float = 0.95
    ) -> Dict:
        """
        最終的なJSON形式に変換
        
        Args:
            asset_id: アセットID
            image_url: 画像URL
            players_data: プレイヤーデータのリスト
            raw_text: 生のOCRテキスト（オプション）
            confidence: 信頼度
            
        Returns:
            指定された形式のJSON
        """
        result = {
            "assetId": asset_id,
            "imageUrl": image_url,
            "rawText": raw_text,
            "aiResult": {
                "players": players_data,
                "unknown": [],  # 認識できなかったデータ
                "confidence": confidence
            }
        }
        
        logger.info(f"JSON変換完了: {len(players_data)}名のプレイヤー")
        return result
    
    @staticmethod
    def validate_player_data(player: Dict) -> bool:
        """
        プレイヤーデータの妥当性をチェック
        
        Args:
            player: プレイヤーデータ
            
        Returns:
            妥当性（True/False）
        """
        required_fields = ['name', 'scores']
        
        # 必須フィールドチェック
        for field in required_fields:
            if field not in player:
                logger.warning(f"プレイヤーデータに'{field}'がありません")
                return False
        
        # スコアデータのチェック
        scores = player.get('scores', [])
        if not isinstance(scores, list):
            logger.warning("スコアがリスト形式ではありません")
            return False
        
        for score in scores:
            if not isinstance(score, dict):
                continue
            
            # hole, strokes, puttsが必須
            if 'hole' not in score or 'strokes' not in score or 'putts' not in score:
                logger.warning(f"スコアデータに必須フィールドがありません: {score}")
                return False
        
        return True
    
    @staticmethod
    def merge_ocr_and_ai_results(
        ocr_players: List[Dict],
        ai_players: List[Dict],
        validation_results: Dict
    ) -> List[Dict]:
        """
        OCR結果とAI Vision結果をマージ
        
        Args:
            ocr_players: OCRで抽出したプレイヤーデータ
            ai_players: AI Visionで抽出したプレイヤーデータ
            validation_results: DB検証結果
            
        Returns:
            マージされたプレイヤーデータ
        """
        # AI Visionの結果を優先（信頼度が高いため）
        if ai_players and len(ai_players) > 0:
            logger.info("AI Vision結果を採用")
            return ai_players
        
        # AI結果がない場合はOCR結果
        if ocr_players and len(ocr_players) > 0:
            logger.info("OCR結果を採用")
            return ocr_players
        
        logger.warning("プレイヤーデータが取得できませんでした")
        return []
    
    @staticmethod
    def create_sample_output() -> Dict:
        """
        サンプル出力を生成（テスト用）
        
        Returns:
            サンプルJSON
        """
        return {
            "assetId": 27,
            "imageUrl": "http://minio:9000/golf-score/ocr/20260129050749524541.jpg",
            "rawText": None,
            "aiResult": {
                "players": [
                    {
                        "name": "市村",
                        "scores": [
                            {"hole": 1, "strokes": 6, "putts": 2},
                            {"hole": 2, "strokes": 5, "putts": 1},
                            {"hole": 3, "strokes": 3, "putts": 1},
                            {"hole": 4, "strokes": 5, "putts": 2},
                            {"hole": 5, "strokes": 4, "putts": 1},
                            {"hole": 6, "strokes": 3, "putts": 1},
                            {"hole": 7, "strokes": 4, "putts": 1},
                            {"hole": 8, "strokes": 5, "putts": 2},
                            {"hole": 9, "strokes": 6, "putts": 1},
                            {"hole": 10, "strokes": 6, "putts": 1},
                            {"hole": 11, "strokes": 5, "putts": 2},
                            {"hole": 12, "strokes": 3, "putts": 3},
                            {"hole": 13, "strokes": 4, "putts": 1},
                            {"hole": 14, "strokes": 4, "putts": 2},
                            {"hole": 15, "strokes": 3, "putts": 1},
                            {"hole": 16, "strokes": 4, "putts": 2},
                            {"hole": 17, "strokes": 4, "putts": 2},
                            {"hole": 18, "strokes": 5, "putts": 2}
                        ]
                    },
                    {
                        "name": "瀬谷 邦",
                        "scores": [
                            {"hole": 1, "strokes": 4, "putts": 1},
                            {"hole": 2, "strokes": 4, "putts": 2},
                            {"hole": 3, "strokes": 3, "putts": 1},
                            {"hole": 4, "strokes": 4, "putts": 1},
                            {"hole": 5, "strokes": 5, "putts": 2},
                            {"hole": 6, "strokes": 4, "putts": 1},
                            {"hole": 7, "strokes": 3, "putts": 2},
                            {"hole": 8, "strokes": 5, "putts": 1},
                            {"hole": 9, "strokes": 6, "putts": 1},
                            {"hole": 10, "strokes": 4, "putts": 1},
                            {"hole": 11, "strokes": 4, "putts": 1},
                            {"hole": 12, "strokes": 3, "putts": 1},
                            {"hole": 13, "strokes": 4, "putts": 1},
                            {"hole": 14, "strokes": 5, "putts": 1},
                            {"hole": 15, "strokes": 7, "putts": 1},
                            {"hole": 16, "strokes": 8, "putts": 2},
                            {"hole": 17, "strokes": 10, "putts": 1},
                            {"hole": 18, "strokes": 12, "putts": 2}
                        ]
                    }
                ],
                "unknown": [],
                "confidence": 0.95
            }
        }


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    import json
    
    # サンプル出力を生成
    sample = JSONFormatter.create_sample_output()
    
    print("=== サンプルJSON出力 ===")
    print(json.dumps(sample, ensure_ascii=False, indent=2))
    
    # バリデーションテスト
    print("\n=== バリデーションテスト ===")
    for player in sample['aiResult']['players']:
        is_valid = JSONFormatter.validate_player_data(player)
        print(f"{player['name']}: {'OK' if is_valid else 'NG'}")
