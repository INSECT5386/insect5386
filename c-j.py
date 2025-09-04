import csv
import json

csv_path = "train.csv"
jsonl_path = "dataset.jsonl"
ban_words = ["Qwen", "qwen", "알리바바"]  # 여기에 제외할 단어 추가

def contains_ban_word(text):
    return any(ban in text for ban in ban_words)

with open(csv_path, newline="", encoding="utf-8") as csv_file, \
     open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
    
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        human_text = row["ko"].strip()
        gpt_text = row["en"].strip()
        
        # 하나라도 금지어 포함 시 건너뛰기
        if contains_ban_word(human_text) or contains_ban_word(gpt_text):
            continue
        
        conversation = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": gpt_text}
            ]
        }
        
        jsonl_file.write(json.dumps(conversation, ensure_ascii=False) + "\n")

print(f"✅ 변환 완료! '{jsonl_path}' 생성됨")
