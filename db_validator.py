"""
データベース照合モジュール
gdo.sqliteからゴルフ場のパー情報を取得し、スコアの妥当性を検証
"""

import sqlite3
import json
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DatabaseValidator:
    """ゴルフ場データベースとの照合・検証クラス"""
    
    def __init__(self, db_path: str = "gdo.sqlite"):
        self.db_path = db_path
        self.conn = None
    
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def search_course_by_name(self, course_name: str) -> List[Dict]:
        """
        ゴルフ場名で検索
        
        Args:
            course_name: ゴルフ場名（部分一致）
            
        Returns:
            マッチしたゴルフ場のリスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, address, tel, raw_json
            FROM courses
            WHERE name LIKE ?
        """, (f"%{course_name}%",))
        
        results = []
        for row in cursor.fetchall():
            course_id, name, address, tel, raw_json = row
            course_data = json.loads(raw_json)
            results.append({
                'id': course_id,
                'name': name,
                'address': address,
                'tel': tel,
                'yardage_data': course_data.get('yardage', {})
            })
        
        logger.info(f"ゴルフ場検索: '{course_name}' -> {len(results)}件")
        return results
    
    def get_course_by_id(self, course_id: str) -> Optional[Dict]:
        """
        ゴルフ場IDで取得
        
        Args:
            course_id: ゴルフ場ID
            
        Returns:
            ゴルフ場データ
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, address, tel, raw_json
            FROM courses
            WHERE id = ?
        """, (course_id,))
        
        row = cursor.fetchone()
        if row:
            course_id, name, address, tel, raw_json = row
            course_data = json.loads(raw_json)
            return {
                'id': course_id,
                'name': name,
                'address': address,
                'tel': tel,
                'yardage_data': course_data.get('yardage', {})
            }
        return None
    
    def get_par_info(self, course_id: str) -> Dict[int, int]:
        """
        ゴルフ場の全ホールのパー情報を取得
        
        Args:
            course_id: ゴルフ場ID
            
        Returns:
            {hole_number: par} の辞書
        """
        course = self.get_course_by_id(course_id)
        if not course:
            logger.warning(f"ゴルフ場が見つかりません: {course_id}")
            return {}
        
        par_info = {}
        yardage_data = course['yardage_data']
        
        # 各ブロック（OUT/IN等）のパー情報を統合
        for block in yardage_data.get('blocks', []):
            par_dict = block.get('par', {})
            for hole_str, par in par_dict.items():
                hole_num = int(hole_str)
                par_info[hole_num] = par
        
        logger.info(f"パー情報取得: {course['name']} - {len(par_info)}ホール")
        return par_info
    
    def validate_score(self, hole: int, strokes: int, par: int) -> Tuple[bool, str]:
        """
        スコアの妥当性を検証
        
        Args:
            hole: ホール番号
            strokes: 打数
            par: パー
            
        Returns:
            (is_valid, message): 妥当性とメッセージ
        """
        # 基本的な範囲チェック
        if strokes < 1:
            return False, f"ホール{hole}: 打数が1未満 ({strokes})"
        
        if strokes > par + 10:
            return False, f"ホール{hole}: 打数が異常に多い (パー{par}で{strokes}打)"
        
        # パー3で10打以上は警告
        if par == 3 and strokes >= 10:
            return False, f"ホール{hole}: パー3で{strokes}打（要確認）"
        
        # パー4で12打以上は警告
        if par == 4 and strokes >= 12:
            return False, f"ホール{hole}: パー4で{strokes}打（要確認）"
        
        # パー5で15打以上は警告
        if par == 5 and strokes >= 15:
            return False, f"ホール{hole}: パー5で{strokes}打（要確認）"
        
        return True, f"ホール{hole}: OK"
    
    def validate_player_scores(self, course_id: str, player_scores: List[Dict]) -> Dict:
        """
        プレイヤーのスコア全体を検証
        
        Args:
            course_id: ゴルフ場ID
            player_scores: [{'hole': int, 'strokes': int, 'putts': int}, ...]
            
        Returns:
            {
                'is_valid': bool,
                'errors': [str],
                'warnings': [str],
                'par_info': {hole: par}
            }
        """
        par_info = self.get_par_info(course_id)
        
        if not par_info:
            return {
                'is_valid': False,
                'errors': ["ゴルフ場のパー情報が取得できません"],
                'warnings': [],
                'par_info': {}
            }
        
        errors = []
        warnings = []
        
        for score in player_scores:
            hole = score.get('hole')
            strokes = score.get('strokes')
            
            if hole not in par_info:
                errors.append(f"ホール{hole}のパー情報がありません")
                continue
            
            par = par_info[hole]
            is_valid, message = self.validate_score(hole, strokes, par)
            
            if not is_valid:
                warnings.append(message)
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'par_info': par_info
        }
    
    def list_all_courses(self, limit: int = 10) -> List[Dict]:
        """
        全ゴルフ場を一覧表示（テスト用）
        
        Args:
            limit: 取得件数
            
        Returns:
            ゴルフ場リスト
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, name, address
            FROM courses
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'name': row[1],
                'address': row[2]
            })
        
        return results


if __name__ == "__main__":
    # テスト実行
    logging.basicConfig(level=logging.INFO)
    
    with DatabaseValidator() as db:
        # ゴルフ場一覧
        print("=== ゴルフ場一覧（最初の5件） ===")
        courses = db.list_all_courses(limit=5)
        for course in courses:
            print(f"  {course['id']}: {course['name']}")
        
        # 特定のゴルフ場のパー情報
        if courses:
            test_course_id = courses[0]['id']
            print(f"\n=== {courses[0]['name']} のパー情報 ===")
            par_info = db.get_par_info(test_course_id)
            for hole, par in sorted(par_info.items()):
                print(f"  ホール{hole}: パー{par}")
            
            # スコア検証テスト
            print(f"\n=== スコア検証テスト ===")
            test_scores = [
                {'hole': 1, 'strokes': 4, 'putts': 2},
                {'hole': 2, 'strokes': 15, 'putts': 3},  # 異常値
                {'hole': 3, 'strokes': 3, 'putts': 1},
            ]
            
            validation = db.validate_player_scores(test_course_id, test_scores)
            print(f"検証結果: {'OK' if validation['is_valid'] else 'エラー'}")
            if validation['errors']:
                print("エラー:")
                for error in validation['errors']:
                    print(f"  - {error}")
            if validation['warnings']:
                print("警告:")
                for warning in validation['warnings']:
                    print(f"  - {warning}")
