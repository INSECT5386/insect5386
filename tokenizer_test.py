
import sentencepiece as spm

# 불러오기
sp = spm.SentencePieceProcessor(model_file=r"C:\Users\yuchan\Serve_Project\data-tokenizer\unigram.model")


# 테스트 문장
text = """
그럼 예비군 소대장은 현역이 되는 건가요

관계 법률 상 군복무기간은 임용되거나 입영한 날로부터요 전역한 날이 속하는 달까지의 연월수로 계산하게 되어 있습니다 예비역 지휘관으로 근무한 기간은 군 복무 기간에 포함되지 않습니다

세종대왕
이순신

"""

# 텍스트를 ID 시퀀스로 변환
ids = sp.encode(text, out_type=int)
print("Token IDs:", ids)

# 텍스트를 서브워드 토큰으로 변환
tokens = sp.encode(text, out_type=str)
print("Subword Tokens:", tokens)

# 다시 텍스트로 복원
decoded = sp.decode(ids)
print("Decoded Text:", decoded)
