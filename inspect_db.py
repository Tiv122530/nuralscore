import sqlite3

conn = sqlite3.connect('gdo.sqlite')
cursor = conn.cursor()

# テーブル一覧を取得
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("=== テーブル一覧 ===")
for table in tables:
    print(f"  - {table}")

print("\n=== 各テーブルのスキーマ ===")
for table in tables:
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table}'")
    schema = cursor.fetchone()[0]
    print(f"\n{schema}")
    
    # サンプルデータを取得
    cursor.execute(f"SELECT * FROM {table} LIMIT 3")
    rows = cursor.fetchall()
    if rows:
        print(f"\nサンプルデータ（最初の3行）:")
        for row in rows:
            print(f"  {row}")

conn.close()
