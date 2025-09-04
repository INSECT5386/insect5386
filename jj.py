import json

# JSON 파일 불러오기 (예: data.json)
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# JSONL로 저장
with open("converted.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        text = item.get("text", "")
        # <start>, <end> 추가
        item["text"] = "<start> " + text + " <end>"
        # 한 줄씩 JSON 문자열로 저장
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
