import json
import pymysql

# DB 연결
conn = pymysql.connect(
    host='127.0.0.1',
    port=3307,
    user='root',
    password='root',
    db='recipe_db',
    charset='utf8mb4'
)
cursor = conn.cursor()

# JSON 불러오기
with open('recipes_fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 배치 커밋용 변수
batch_size = 1000
batch = []

for i, item in enumerate(data, start=1):
    title = item.get("title", "")
    ingredients = item.get("prepare", "")
    tools = item.get("kitchenware", "")
    content = item.get("step", "")
    
    batch.append((title, ingredients, tools, content))
    
    # 1000개마다 insert & commit
    if i % batch_size == 0:
        cursor.executemany(
            "INSERT INTO recipe (title, ingredients, tools, content) VALUES (%s, %s, %s, %s)",
            batch
        )
        conn.commit()
        print(f"✅ {i}개 삽입 완료")
        batch.clear()

# 남은 데이터 처리
if batch:
    cursor.executemany(
        "INSERT INTO recipe (title, ingredients, tools, content) VALUES (%s, %s, %s, %s)",
        batch
    )
    conn.commit()
    print(f"✅ 마지막 {len(batch)}개 삽입 완료")

cursor.close()
conn.close()
print("🎉 전체 데이터 삽입 완료")