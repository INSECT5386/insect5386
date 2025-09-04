import json

jsonl_path = "C:/Users/yuchan/Serve_Project/intent_dataset.jsonl"
txt_path = "C:/Users/yuchan/Serve_Project/intent_corpus.txt"

with open(jsonl_path, "r", encoding="utf-8") as f_in, open(txt_path, "w", encoding="utf-8") as f_out:
    for line in f_in:
        item = json.loads(line)
        text = item["text"].strip()
        if text:
            f_out.write(text + "\n")

print(f"텍스트 코퍼스 저장 완료: {txt_path}")

import sentencepiece as spm

txt_path = "C:/Users/yuchan/Serve_Project/intent_corpus.txt"
model_prefix = "intent_spm"
vocab_size = 729  # 필요에 맞게 조정 가능

spm.SentencePieceTrainer.train(
    input=txt_path,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=0.9995,  # 한국어 포함하려면 거의 1.0 가까이
    model_type="unigram"  # unigram, bpe, char, word 중 선택 가능
)

print("SentencePiece 모델 학습 완료!")
