import json

input_file = r"C:\Users\yuchan\Serve_Project\abaa.jsonl"  # question, answer 있는 원본 JSONL
output_file = "conversations1.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        data = json.loads(line.strip())
        # question/answer 컬럼을 가져와서 conversations 구조로 변환
        conv = {
            "conversations": [
                {"from": "human", "value": data["question"]},
                {"from": "gpt", "value": data["answer"]}
            ]
        }
        f_out.write(json.dumps(conv, ensure_ascii=False) + "\n")
