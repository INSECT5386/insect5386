import sentencepiece as spm
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
    "https://huggingface.co/datasets/Yuchan5386/Smolwrite-dataset/resolve/main/dataset.txt?download=true",
    "dataset.txt"
)

def train_unigram_tokenizer(input_file, model_prefix=None, vocab_size=None):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",
        character_coverage=1.0,
        num_threads=12,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        user_defined_symbols=["<start>", "<end>", "<pad>", "<sep>"]  # <-- 여기서 <unk> 빼기!
    )


    print(f"✅ '{model_prefix}.model' / '{model_prefix}.vocab' 생성 완료!")

# 사용 예
if __name__ == "__main__":
    train_unigram_tokenizer('dataset.txt', model_prefix="unigram", vocab_size=51200)
