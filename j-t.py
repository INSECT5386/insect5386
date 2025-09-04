import json
import requests

# =======================
# 0) 파일 다운로드 함수
# =======================
def download_file(url, save_path):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)
    print(f"✅ {save_path} 저장됨")

# =======================
# 1) 데이터 및 토크나이저 다운로드
# =======================
download_file(
    "https://huggingface.co/datasets/Yuchan5386/KoWrite-dataset/resolve/main/data_shuffled_1.jsonl?download=true",
    "data.jsonl"
)

jsonl_path = 'data.jsonl'
txt_path = "dataset.txt"

with open(jsonl_path, "r", encoding="utf-8") as jf, \
     open(txt_path, "w", encoding="utf-8") as tf:
    
    for line in jf:
        data = json.loads(line)
        conv = data.get("conversations", [])
        
        # human과 gpt 값 추출
        human_text = ""
        gpt_text = ""
        for turn in conv:
            if turn.get("from") == "human":
                human_text = turn.get("value", "").strip()
            elif turn.get("from") == "gpt":
                gpt_text = turn.get("value", "").strip()
        
        if human_text and gpt_text:
            tf.write(f"<start> {human_text} <sep> {gpt_text} <end>\n")

print(f"✅ 변환 완료! '{txt_path}' 생성됨")
