"""
メインパイプライン
ハイブリッド型スコアカードOCRシステム
"""

import os
import json
import logging
import cv2
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

from image_preprocessor import ImagePreprocessor
from ocr_processor import OCRProcessor
from db_validator import DatabaseValidator
from ai_vision import AIVisionProcessor
from json_formatter import JSONFormatter

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridScorecardOCR:
    """ハイブリッド型スコアカードOCRシステムのメインクラス"""
    
    def __init__(
        self,
        db_path: str = "gdo.sqlite",
        debug_mode: bool = False,
        output_dir: str = "./output",
        confidence_threshold: float = 0.7,
        use_ai_vision: bool = True,
        ai_vision_only: bool = False
    ):
        """
        Args:
            db_path: データベースファイルのパス
            debug_mode: デバッグモード（中間画像を保存）
            output_dir: 出力ディレクトリ
            confidence_threshold: OCR信頼度閾値
            use_ai_vision: AI Visionを使用するか
            ai_vision_only: AI Vision専用モード（OCRはスキップ、向き検出のみ）
        """
        self.db_path = db_path
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir)
        self.confidence_threshold = confidence_threshold
        self.use_ai_vision = use_ai_vision
        self.ai_vision_only = ai_vision_only
        
        # 各モジュールを初期化
        self.preprocessor = ImagePreprocessor(debug_mode=debug_mode, output_dir=output_dir)
        self.ocr_processor = OCRProcessor()
        
        # AI Visionは使用する場合のみ初期化
        self.ai_processor = None
        if use_ai_vision:
            try:
                # プロバイダーを環境変数から取得
                provider = os.getenv('AI_PROVIDER', 'openai')
                self.ai_processor = AIVisionProcessor(provider=provider)
                if ai_vision_only:
                    logger.info(f"AI Vision専用モード: ON（OCRは向き検出のみ使用、プロバイダー: {provider}）")
                else:
                    logger.info(f"ハイブリッドモード: ON（OCR + AI Vision、プロバイダー: {provider}）")
            except ValueError as e:
                logger.warning(f"AI Vision初期化失敗: {e}")
                logger.warning("OCRのみで動作します")
                self.use_ai_vision = False
                self.ai_vision_only = False
        else:
            logger.info("AI Vision使用: OFF")
        
        self.output_dir.mkdir(exist_ok=True)
    
    def process_scorecard(
        self,
        image_path: str,
        course_id: Optional[str] = None,
        asset_id: int = 1,
        image_url: str = ""
    ) -> Dict:
        """
        スコアカード画像を処理してJSON形式で返す
        
        Args:
            image_path: 画像ファイルのパス
            course_id: ゴルフ場ID（既知の場合）
            asset_id: アセットID
            image_url: 画像URL
            
        Returns:
            処理結果のJSON
        """
        logger.info(f"=== スコアカード処理開始: {image_path} ===")
        
        # 1. 画像前処理
        logger.info("ステップ1: 画像前処理")
        original_image, processed_image = self.preprocessor.process(image_path)
        
        # 2. OCR処理
        logger.info("ステップ2: OCR処理")
        ocr_results = None
        ocr_confidence = 0
        low_confidence_regions = []
        
        # AI Vision専用モードの場合はOCRスキップ
        if not self.ai_vision_only:
            ocr_results = self.ocr_processor.extract_scorecard_data(processed_image)
            ocr_confidence = ocr_results['average_confidence']
            logger.info(f"OCR平均信頼度: {ocr_confidence:.2f}")
            
            # 3. 低信頼度チェック
            low_confidence_regions = self.ocr_processor.get_low_confidence_regions(
                ocr_results['raw_ocr_results'],
                threshold=self.confidence_threshold
            )
        else:
            logger.info("OCR処理スキップ（AI Vision専用モード）")
        
        # 4. AI Vision判定
        ai_results = None
        use_ai = False
        
        if self.use_ai_vision and self.ai_processor:
            if self.ai_vision_only:
                # AI Vision専用モードでは常にAI使用
                logger.info("ステップ3: AI Vision実行（専用モード）")
                use_ai = True
            elif ocr_confidence < self.confidence_threshold or len(low_confidence_regions) > 5:
                # OCR信頼度が低い、または低信頼度領域が多い場合にAI使用
                logger.info("ステップ3: AI Vision実行（OCR信頼度が低いため）")
                use_ai = True
            else:
                logger.info("ステップ3: AI Visionスキップ（OCR信頼度が十分）")
        
        if use_ai:
            # 補正済み画像を一時ファイルに保存してAI Visionに送信
            temp_processed_path = self.output_dir / "temp_processed_for_ai.jpg"
            cv2.imwrite(str(temp_processed_path), processed_image)
            ai_results = self.ai_processor.analyze_scorecard(str(temp_processed_path))
        
        # 5. データベース照合
        logger.info("ステップ4: データベース照合")
        validation_results = None
        
        if course_id:
            with DatabaseValidator(self.db_path) as db:
                # AI結果またはOCR結果からプレイヤーデータを取得
                players_to_validate = []
                if ai_results and ai_results.get('players'):
                    players_to_validate = ai_results['players']
                
                # 各プレイヤーのスコアを検証
                for player in players_to_validate:
                    scores = player.get('scores', [])
                    validation = db.validate_player_scores(course_id, scores)
                    
                    if validation['warnings']:
                        logger.warning(f"プレイヤー '{player.get('name')}' のスコアに警告:")
                        for warning in validation['warnings']:
                            logger.warning(f"  - {warning}")
        
        # 6. 結果統合
        logger.info("ステップ5: 結果統合")
        final_players = []
        
        if ai_results and ai_results.get('players'):
            # AI結果を優先
            final_players = ai_results['players']
            final_confidence = ai_results.get('confidence', 0.95)
            logger.info("最終結果: AI Vision採用")
        else:
            # OCR結果を使用（簡易的な変換）
            # 注: 実際にはOCR結果から構造化データを抽出する高度な処理が必要
            final_players = self._convert_ocr_to_players(ocr_results)
            final_confidence = ocr_confidence
            logger.info("最終結果: OCR採用")
        
        # 7. JSON形式に変換
        logger.info("ステップ6: JSON変換")
        result = JSONFormatter.format_result(
            asset_id=asset_id,
            image_url=image_url or f"file://{Path(image_path).absolute()}",
            players_data=final_players,
            raw_text=None,
            confidence=final_confidence
        )
        
        # 結果を保存
        output_path = self.output_dir / "result.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"=== 処理完了: 結果を {output_path} に保存 ===")
        return result
    
    def _convert_ocr_to_players(self, ocr_results: Dict) -> list:
        """
        OCR結果をプレイヤーデータに変換（簡易版）
        
        注: 実際にはOCRの位置情報から表構造を解析する高度な処理が必要
        """
        players = []
        
        # プレイヤー名のみ抽出
        for player_info in ocr_results.get('players', []):
            players.append({
                'name': player_info['name'],
                'scores': []  # スコアは未実装（表構造解析が必要）
            })
        
        if not players:
            # プレイヤーが見つからない場合は空のプレイヤーを追加
            players.append({
                'name': '不明',
                'scores': []
            })
        
        logger.warning("OCR結果からのプレイヤーデータ変換は簡易版です")
        return players


def main():
    """メイン関数"""
    # 環境変数読み込み
    load_dotenv()
    
    # 設定
    IMAGE_PATH = "IMG_1873.jpg"
    COURSE_ID = None  # ゴルフ場IDが分かる場合は指定
    DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"
    USE_AI_VISION = True  # AI Visionを使用するか
    AI_VISION_ONLY = os.getenv("AI_VISION_ONLY", "True").lower() == "true"  # AI Vision専用モード
    
    # システム初期化
    ocr_system = HybridScorecardOCR(
        debug_mode=DEBUG_MODE,
        use_ai_vision=USE_AI_VISION,
        ai_vision_only=AI_VISION_ONLY
    )
    
    # スコアカード処理
    if Path(IMAGE_PATH).exists():
        result = ocr_system.process_scorecard(
            image_path=IMAGE_PATH,
            course_id=COURSE_ID,
            asset_id=27,
            image_url="http://minio:9000/golf-score/ocr/sample.jpg"
        )
        
        # 結果表示
        print("\n" + "="*60)
        print("処理結果")
        print("="*60)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        logger.error(f"画像ファイルが見つかりません: {IMAGE_PATH}")
        print(f"\nエラー: {IMAGE_PATH} が見つかりません")
        print("サンプル画像を配置してから実行してください")


if __name__ == "__main__":
    main()
