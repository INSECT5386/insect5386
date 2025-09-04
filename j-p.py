import pandas as pd

# 1️⃣ JSONL 파일 경로
jsonl_file = r"C:\Users\yuchan\Serve_Project\y_pair_structure.jsonl"  # 변환할 JSONL 파일
parquet_file = "data.parquet"  # 저장할 Parquet 파일

# 2️⃣ JSONL 읽기
# lines=True 옵션은 JSONL 형식(한 줄마다 JSON) 읽을 때 필수
df = pd.read_json(jsonl_file, lines=True)

# 3️⃣ Parquet으로 저장
df.to_parquet(parquet_file, engine='pyarrow', index=False)

print(f"✅ 변환 완료: {parquet_file}")
