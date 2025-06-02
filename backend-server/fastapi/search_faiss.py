import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from app.db import SessionLocal  # 기존 DB 연결 코드 재사용
import math

INDEX_SAVE_PATH = "faiss_store/index.faiss"
META_SAVE_PATH = "faiss_store/metadata.pkl"

def recipe_to_text(row):
    return f"{row['title']} 레시피의 주요 재료는 {row['ingredients']}입니다. 이 조합은 풍부한 맛을 내는 데 중요한 역할을 합니다."

# 모델 로딩
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
dimension = model.get_sentence_embedding_dimension()

# FAISS 인덱스 로딩
index = faiss.read_index(INDEX_SAVE_PATH)

# 메타데이터 로딩
with open(META_SAVE_PATH, "rb") as f:
    metadata = pickle.load(f)

# 사용자 입력 임베딩
user_ingredients = ["고추장","계란","김치"]
query_sentence = f"요리에 사용된 재료는 {', '.join(user_ingredients)}입니다."
query_embedding = model.encode([query_sentence])
query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

# 유사한 벡터 20개 검색
k = 100
D, I = index.search(np.array(query_embedding), k)

# DB 연결
session = SessionLocal()

print(f"\n📌 입력한 재료로 만들 수 있는 요리 추천 결과입니다! ({', '.join(user_ingredients)})")
result_found = False  # 결과 유무 플래그

seen_titles = set()

# 중복 제거 (레시피 ID 기준) + distance도 같이 관리
recipe_best_result = {}

for idx, distance in zip(I[0], D[0]):
    if idx < len(metadata):
        doc = metadata[idx]
        recipe_id = doc.get("id")
        # 같은 recipe_id 중에서는 distance가 더 작은 것(= 더 유사한 것)만 저장
        if (recipe_id not in recipe_best_result) or (distance < recipe_best_result[recipe_id][1]):
            recipe_best_result[recipe_id] = (idx, distance)

# 정리된 결과
unique_results = list(recipe_best_result.values())

# 중복 제거된 결과 출력
sorted_results = sorted(unique_results, key=lambda x: x[1])
for idx, distance in sorted_results:
    if idx >= len(metadata):
        continue  # 방어: FAISS 인덱스가 메타데이터보다 클 경우
    try:
        doc = metadata[idx]
        recipe_ingredients = set(doc.get("ingredients", "").replace(" ", "").split(","))
        user_ingredient_set = set(user_ingredients)
        match_score = len(user_ingredient_set & recipe_ingredients) / (len(recipe_ingredients) or 1)
        distance_score = 1 / (1 + distance)  # 거리 기반 유사도 변환
        if match_score == 1.0:
            final_score = 1.0
        else:
            final_score = 0.4 * distance_score + 0.6 * match_score  # 재료 일치 비중 강화

        if match_score >= 0.3:  # 재료 30% 이상 매칭 필터링
            recipe_id = metadata[idx]["id"]
            recipe = session.execute("SELECT title, ingredients, content FROM recipe WHERE id = :id", {"id": recipe_id}).fetchone()
            if recipe and recipe.title not in seen_titles:
                seen_titles.add(recipe.title)
                content_preview = recipe.content
                if not isinstance(content_preview, str):
                    content_preview = str(content_preview)
                content_preview = content_preview[:150]
                print(f"\n🍽️ {recipe.title}")
                print(f"🥬 재료: {recipe.ingredients}")
                print(f"📖 내용: {content_preview}...")
                result_found = True
    except Exception as e:
        print(f"❗ 예외 발생 (idx={idx}): {e}")
        continue

if not result_found:
    print("❗ 검색된 결과가 없습니다. 유사도 조건이나 입력 재료를 조정해보세요.")