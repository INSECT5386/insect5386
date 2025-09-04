import json
import sentencepiece as spm
"""

# JSONL 파일 경로
jsonl_path = "dataset.jsonl"

# KO / EN 텍스트 모으기
ko_texts = []
en_texts = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        for conv in item.get("conversations", []):
            if conv["from"] == "human":
                ko_texts.append(conv["value"].strip())
            elif conv["from"] == "gpt":
                en_texts.append(conv["value"].strip())

# 임시 텍스트 파일로 저장 (토크나이저 학습용)
with open("ko_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(ko_texts))

with open("en_corpus.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(en_texts))

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
        user_defined_symbols=["<start>", "<sep>", "<end>", "<pad>"]  # <-- 여기서 <unk> 빼기!
    )


    print(f"✅ '{model_prefix}.model' / '{model_prefix}.vocab' 생성 완료!")
"""
def train_bpe_tokenizer(input_file, model_prefix=None, vocab_size=None):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        num_threads=12,
        bos_id=1,
        eos_id=2,
        unk_id=0,
        pad_id=3,
        user_defined_symbols=["<start>", "<sep>", "<end>", "<pad>"]  # <-- 여기서 <unk> 빼기!
    )


    print(f"✅ '{model_prefix}.model' / '{model_prefix}.vocab' 생성 완료!")


# 사용 예
if __name__ == "__main__":
    #train_unigram_tokenizer('ko_corpus.txt', model_prefix="ko_unigram", vocab_size=72550)
    train_bpe_tokenizer('en_corpus.txt', model_prefix="en_bpe", vocab_size=72550)
