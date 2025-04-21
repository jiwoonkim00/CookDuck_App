import json

input_path = "recipes.json"
output_path = "recipes_fixed.json"

fixed_lines = []
with open(input_path, "r", encoding="utf-8", errors="ignore") as infile:
    for line in infile:
        line = line.strip()
        if line:
            try:
                obj = json.loads(line)
                if isinstance(obj, str):
                    obj = json.loads(obj)
                fixed_lines.append(obj)
            except json.JSONDecodeError:
                print("⚠️ JSON 파싱 오류 줄:", line)

with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(fixed_lines, outfile, ensure_ascii=False, indent=2)

print("✅ recipes_fixed.json 생성 완료")