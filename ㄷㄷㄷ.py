import glob

# 합칠 JSONL 파일들이 있는 폴더
folder_path = r"C:\Users\yuchan\Serve_Project\data"

# 합쳐서 만들 최종 파일
output_file = "merged_qa_pairs.jsonl"

# 폴더 안의 모든 jsonl 파일 찾기
jsonl_files = glob.glob(f"{folder_path}/*.jsonl")

with open(output_file, "w", encoding="utf-8") as out_f:
    for file in jsonl_files:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                out_f.write(line)

print(f"{len(jsonl_files)}개의 JSONL 파일을 '{output_file}'로 합쳤습니다.")
