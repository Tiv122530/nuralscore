"""
データベースから日立ゴルフクラブを検索してパー情報を取得
"""
import sqlite3
import json

conn = sqlite3.connect('gdo.sqlite')
cursor = conn.cursor()

# 日立ゴルフクラブを検索
cursor.execute("SELECT id, name, address, raw_json FROM courses WHERE name LIKE ?", ('%日立%',))
results = cursor.fetchall()

print('日立ゴルフクラブの検索結果:')
print('=' * 60)
for r in results:
    course_id = r[0]
    name = r[1]
    address = r[2]
    raw_json = r[3]
    
    print(f'ID: {course_id}')
    print(f'名前: {name}')
    print(f'住所: {address}')
    print('-' * 60)
    
    # JSONからパー情報を取得
    try:
        course_data = json.loads(raw_json)
        yardage = course_data.get('yardage', {})
        blocks = yardage.get('blocks', [])
        
        print(f'\nコース情報（{len(blocks)}ブロック）:')
        total_par = 0
        
        for block in blocks:
            label = block.get('label', '?')
            par_dict = block.get('par', {})
            holes = block.get('holes', [])
            
            print(f'\n  {label}コース:')
            block_par = 0
            for hole in holes:
                hole_str = str(hole)
                par = par_dict.get(hole_str, 0)
                block_par += par
                print(f'    ホール{hole}: パー{par}')
            
            print(f'    {label}合計: {block_par}')
            total_par += block_par
        
        print(f'\n  トータルパー: {total_par}')
        
    except json.JSONDecodeError as e:
        print(f'JSON解析エラー: {e}')
    
    print('=' * 60)

conn.close()
