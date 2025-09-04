import json

input_path = r"C:\Users\yuchan\Serve_Project\output1.jsonl"   # 원본 파일
output_path = "output2.jsonl" # 필터링 결과 저장

max_len = 100

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        data = json.loads(line.strip())
        conv = data["conversations"]

        human_text = next((c["value"] for c in conv if c["from"] == "human"), "")
        gpt_text = next((c["value"] for c in conv if c["from"] == "gpt"), "")

        if len(human_text) + len(gpt_text) <= max_len:
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
