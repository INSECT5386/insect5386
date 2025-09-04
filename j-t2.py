import json

# JSON 파일 불러오기 (예: data.json)
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 변환된 텍스트 저장할 리스트
texts = []

for item in data:
    text = item.get("text", "")
    # <start>와 <end> 추가
    new_text = "<start> " + text + " <end>"
    texts.append(new_text)

# txt 파일로 저장
with open("converted.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")
