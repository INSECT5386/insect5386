import pandas as pd
import numpy as np
import json

# Parquet 파일 불러오기
df = pd.read_parquet("d.parquet", engine="pyarrow")

# JSON 직렬화용 함수: NumPy 배열은 리스트로 변환
def convert_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj

# JSONL 파일로 저장
with open("data2.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        row_dict = {k: convert_for_json(v) for k, v in row.items()}
        json_line = json.dumps(row_dict, ensure_ascii=False)
        f.write(json_line + "\n")

print("✅ Parquet → JSONL 변환 완료!")
