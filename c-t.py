import pandas as pd

# CSV 불러오기
df = pd.read_csv("all_sentences.csv")

# txt 저장할 파일
output_file = "all_sentences.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for sentence in df["sentence"].dropna():
        line = f"<start> {sentence.strip()} <end>\n"
        f.write(line)

print(f"✅ {len(df)}개의 문장을 {output_file}로 저장 완료!")
