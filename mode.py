import json

input_path = "dataset3.jsonl"
output_path = "conversations3.jsonl"

with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        prompt = obj.get("instruction", "")
        answer = obj.get("output", "")

        conv_obj = {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer}
            ]
        }

        fout.write(json.dumps(conv_obj, ensure_ascii=False) + "\n")
