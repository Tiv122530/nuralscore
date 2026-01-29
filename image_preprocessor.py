"""
画像前処理モジュール
スコアカード画像の前処理（回転補正、台形補正、二値化、ノイズ除去）
"""

import cv2
import numpy as np
from PIL import Image, ExifTags
from pathlib import Path
from typing import Tuple, Optional
import logging
import easyocr

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """スコアカード画像の前処理を行うクラス"""
    
    def __init__(self, debug_mode: bool = False, output_dir: str = "./output"):
        self.debug_mode = debug_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.ocr_reader = easyocr.Reader(['ja', 'en'], gpu=False)
        
    def process(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        画像を前処理して返す
        
        Args:
            image_path: 入力画像のパス
            
        Returns:
            (original_image, processed_image): 元画像と前処理済み画像のタプル
        """
        logger.info(f"画像前処理開始: {image_path}")
        
        # 1. 画像読み込みとEXIF回転補正
        original = self._load_and_fix_orientation(image_path)
        self._save_debug_image(original, "01_original.jpg")
        
        # 2. 回転角度検出と補正
        rotated = self._auto_rotate(original)
        self._save_debug_image(rotated, "02_rotated.jpg")
        
        # 3. 台形補正（スコアカード検出と透視変換）
        perspective_corrected = self._perspective_transform(rotated)
        self._save_debug_image(perspective_corrected, "03_perspective.jpg")
        
        # 4. グレースケール変換
        gray = cv2.cvtColor(perspective_corrected, cv2.COLOR_BGR2GRAY)
        self._save_debug_image(gray, "04_gray.jpg")
        
        # 5. 適応的二値化
        binary = self._adaptive_threshold(gray)
        self._save_debug_image(binary, "05_binary.jpg")
        
        # 6. ノイズ除去
        denoised = self._remove_noise(binary)
        self._save_debug_image(denoised, "06_denoised.jpg")
        
        logger.info("画像前処理完了")
        return original, denoised
    
    def _load_and_fix_orientation(self, image_path: str) -> np.ndarray:
        """EXIF情報を読み取って画像を正しい向きに回転"""
        pil_img = Image.open(image_path)
        
        # EXIF情報から回転角度を取得
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = pil_img._getexif()
            
            if exif is not None:
                orientation_value = exif.get(orientation)
                
                if orientation_value == 3:
                    pil_img = pil_img.rotate(180, expand=True)
                elif orientation_value == 6:
                    pil_img = pil_img.rotate(270, expand=True)
                elif orientation_value == 8:
                    pil_img = pil_img.rotate(90, expand=True)
        except (AttributeError, KeyError, IndexError):
            # EXIF情報がない場合はそのまま
            pass
        
        # PILからOpenCV形式に変換
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def _auto_rotate(self, image: np.ndarray) -> np.ndarray:
        """
        画像の傾きを自動検出して補正
        複数の手法を組み合わせて最適な回転角度を検出
        """
        # 方法1: 文字方向ベースの回転検出（最も正確）
        angle = self._detect_rotation_by_text(image)
        
        if angle is not None:
            logger.info(f"文字ベースで回転角度検出: {angle:.2f}度")
            return self._rotate_image(image, angle)
        
        # 方法2: 輪郭ベースの回転検出（フォールバック）
        logger.info("文字ベースの回転検出が失敗、輪郭ベースを使用")
        angle = self._detect_rotation_by_contours(image)
        
        if angle is not None:
            logger.info(f"輪郭ベースで回転角度検出: {angle:.2f}度")
            return self._rotate_image(image, angle)
        
        # 方法3: ハフ変換による直線検出（最終フォールバック）
        logger.info("輪郭ベースの回転検出も失敗、ハフ変換を使用")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            logger.warning("直線が検出できませんでした。回転補正をスキップします。")
            return image
        
        # 角度のヒストグラムを作成
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta) - 90
            angles.append(angle_deg)
        
        angle_median = np.median(angles)
        logger.info(f"ハフ変換で回転角度検出: {angle_median:.2f}度")
        
        return self._rotate_image(image, angle_median)
    
    def _detect_rotation_by_contours(self, image: np.ndarray) -> float:
        """
        輪郭検出による回転角度の推定
        スコアカードの矩形領域を検出して角度を計算
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 前処理：ノイズ除去とエッジ検出
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 輪郭検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 最大の輪郭を探す
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 面積が小さすぎる場合はスキップ
        if cv2.contourArea(largest_contour) < image.shape[0] * image.shape[1] * 0.1:
            return None
        
        # 最小外接矩形を取得
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # 角度を-45度から45度の範囲に正規化
        if angle < -45:
            angle = 90 + angle
        
        # 90度回転が必要な場合を判定
        # minAreaRectは幅>高さの場合に0度、高さ>幅の場合に-90度を返す
        width, height = rect[1]
        if width < height:
            angle = angle - 90
        
        return angle
    
    def _detect_rotation_by_text(self, image: np.ndarray) -> Optional[float]:
        """
        OCRで検出されたテキストの信頼度を比較して正しい向きを判定
        0度、90度、180度、270度の4方向を試して最適な向きを選択
        
        Returns:
            推定された回転角度（度）、検出できない場合はNone
        """
        try:
            logger.info("文字方向ベースの回転検出を開始（4方向チェック）...")
            
            best_angle = 0
            best_score = -1
            best_count = 0
            
            # 4つの方向を試す（0, 90, 180, 270度）
            for test_angle in [0, 90, 180, 270]:
                # 画像を回転
                if test_angle == 0:
                    test_image = image.copy()
                else:
                    test_image = self._rotate_by_90_degrees(image, test_angle)
                
                # OCRで文字を検出
                results = self.ocr_reader.readtext(test_image)
                
                if not results:
                    continue
                
                # 信頼度の平均とテキスト数を計算
                confidences = [detection[2] for detection in results]
                avg_confidence = np.mean(confidences)
                text_count = len(results)
                
                # スコア = 平均信頼度 × テキスト数の重み付け
                # テキスト数が多い方が正しい向きの可能性が高い
                score = avg_confidence * (1 + text_count / 100)
                
                logger.info(f"  {test_angle}度: テキスト{text_count}個, 平均信頼度{avg_confidence:.3f}, スコア{score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_angle = test_angle
                    best_count = text_count
            
            if best_score < 0:
                logger.warning("どの方向でも文字が検出できませんでした")
                return None
            
            logger.info(f"最適な向き: {best_angle}度（テキスト{best_count}個, スコア{best_score:.3f}）")
            
            # 0度が最適ならさらに微調整
            if best_angle == 0:
                fine_angle = self._detect_fine_rotation(image)
                if fine_angle is not None and abs(fine_angle) > 2:
                    logger.info(f"微調整: {fine_angle:.2f}度")
                    return fine_angle
                return 0
            
            # 90度単位の回転が必要
            return best_angle
            
        except Exception as e:
            logger.warning(f"文字方向検出中にエラー: {e}")
            return None
    
    def _rotate_by_90_degrees(self, image: np.ndarray, angle: int) -> np.ndarray:
        """
        画像を90度単位で回転
        
        Args:
            image: 入力画像
            angle: 回転角度（90, 180, 270のいずれか）
        
        Returns:
            回転された画像
        """
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        else:
            return image
    
    def _detect_fine_rotation(self, image: np.ndarray) -> Optional[float]:
        """
        正しい向きの画像の微細な傾きを検出（数度程度の補正）
        
        Returns:
            推定された回転角度（度）、検出できない場合はNone
        """
        try:
            results = self.ocr_reader.readtext(image)
            
            if not results or len(results) < 3:
                return None
            
            angles = []
            
            for detection in results:
                bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                
                # 左上と右上の2点を使って角度を計算
                pt1 = np.array(bbox[0])  # 左上
                pt2 = np.array(bbox[1])  # 右上
                
                # ベクトルから角度を計算
                dx = pt2[0] - pt1[0]
                dy = pt2[1] - pt1[1]
                
                # 水平線からの角度を計算（度単位）
                angle = np.degrees(np.arctan2(dy, dx))
                
                # -45度から45度の範囲に正規化
                while angle > 45:
                    angle -= 90
                while angle < -45:
                    angle += 90
                
                angles.append(angle)
            
            if not angles:
                return None
            
            # 中央値を使用（外れ値に強い）
            median_angle = np.median(angles)
            
            return median_angle
            
        except Exception as e:
            logger.warning(f"微調整検出中にエラー: {e}")
            return None
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        画像を指定角度回転
        
        Args:
            image: 入力画像
            angle: 回転角度（度）
        
        Returns:
            回転された画像
        """
        # 90度単位の回転は専用関数を使用（高速・高品質）
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270 or angle == -90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif abs(angle) < 0.1:
            return image
        
        # 任意角度の回転
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 回転後の画像サイズを計算
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # 回転行列を調整
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(image, matrix, (new_w, new_h), 
                                  flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        
        if self.debug_mode:
            self._save_debug_image(rotated, "02_rotated.jpg")
        
        return rotated
    
    def _perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """
        台形補正：スコアカードの四隅を検出して正面視に変換
        検出できない場合は元画像を返す
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 最大の四角形を探す
        max_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                # 輪郭を多角形近似
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # 四角形かつ十分な大きさの場合
                if len(approx) == 4 and area > image.shape[0] * image.shape[1] * 0.3:
                    max_area = area
                    best_contour = approx
        
        if best_contour is None:
            logger.warning("スコアカードの四隅が検出できませんでした。台形補正をスキップします。")
            return image
        
        # 四隅の座標を取得
        points = best_contour.reshape(4, 2)
        
        # 座標を並び替え（左上、右上、右下、左下）
        rect = self._order_points(points)
        
        # 変換後のサイズを計算
        width = max(
            np.linalg.norm(rect[0] - rect[1]),
            np.linalg.norm(rect[2] - rect[3])
        )
        height = max(
            np.linalg.norm(rect[0] - rect[3]),
            np.linalg.norm(rect[1] - rect[2])
        )
        
        # 変換先の座標
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # 透視変換
        matrix = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, matrix, (int(width), int(height)))
        
        logger.info("台形補正完了")
        return warped
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """四隅の座標を左上、右上、右下、左下の順に並べ替え"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 合計が最小=左上、最大=右下
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 差が最小=右上、最大=左下
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        適応的二値化
        照明ムラに強い局所的な閾値処理
        """
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=21,  # 近傍領域のサイズ
            C=10  # 閾値調整定数
        )
        
        return binary
    
    def _remove_noise(self, binary: np.ndarray) -> np.ndarray:
        """
        ノイズ除去
        モルフォロジー演算で小さなノイズを除去
        """
        # ノイズ除去（Opening: 侵食→膨張）
        kernel = np.ones((2, 2), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # 穴埋め（Closing: 膨張→侵食）
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return closing
    
    def _save_debug_image(self, image: np.ndarray, filename: str):
        """デバッグモード時に中間画像を保存"""
        if self.debug_mode:
            output_path = self.output_dir / filename
            cv2.imwrite(str(output_path), image)
            logger.debug(f"デバッグ画像保存: {output_path}")


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    preprocessor = ImagePreprocessor(debug_mode=True, output_dir="./output")
    original, processed = preprocessor.process("IMG_1873.jpg")
    
    print(f"元画像サイズ: {original.shape}")
    print(f"処理済み画像サイズ: {processed.shape}")
