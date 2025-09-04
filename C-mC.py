import pandas as pd

# 파일 리스트
file_list = ["file1.csv", "file2.csv", "file3.csv", "file4.csv"]
all_sentences = []

for f in file_list:
    df = pd.read_csv(f)
    if "sentence" in df.columns:
        sentences = df["sentence"].dropna().tolist()
        all_sentences.extend(sentences)
    else:
        print(f"⚠️ {f}에 sentence 컬럼이 없습니다.")

# 통합 CSV 저장
output_file = "all_sentences.csv"
pd.DataFrame({"sentence": all_sentences}).to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ 총 {len(all_sentences)}개의 문장을 {output_file}로 저장 완료!")
