import json
import random


import os, json, requests

def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")

DATA_PATH = "converted.jsonl"
SPM_PATH = "ko_unigram.model"

if not os.path.exists(DATA_PATH):
    download_file(
        "https://huggingface.co/datasets/Yuchan5386/KoWrite-dataset/resolve/main/merged_qa_pairs.jsonl?download=true",
        DATA_PATH
    )

input_path = DATA_PATH
output_path = "data_shuffled_1.jsonl"

# 파일 읽기
with open(input_path, "r", encoding="utf-8") as infile:
    data = [json.loads(line) for line in infile]

# 랜덤 셔플
random.shuffle(data)

# 저장
with open(output_path, "w", encoding="utf-8") as outfile:
    for item in data:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
