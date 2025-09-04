import sentencepiece as spm

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
