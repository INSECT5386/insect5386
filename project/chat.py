import random
from collections import defaultdict, deque
import numpy as np
import time
import sentencepiece as spm

# --- top_p 샘플링 함수 ---
def top_p_sampling(words, probs, p=0.9):
    sorted_pairs = sorted(zip(words, probs), key=lambda x: x[1], reverse=True)
    cum_prob = 0.0
    filtered = []
    for word, prob in sorted_pairs:
        cum_prob += prob
        filtered.append((word, prob))
        if cum_prob >= p:
            break
    total = sum(prob for _, prob in filtered)
    normed = [(w, prob / total) for w, prob in filtered]
    return random.choices([w for w, _ in normed], weights=[p for _, p in normed])[0]

# --- PST 노드 ---
class PSTNode:
    def __init__(self):
        self.next_words = defaultdict(int)
        self.total = 0
        self.children = {}

# --- PST 모델 ---
class PST:
    def __init__(self, max_depth=4):
        self.root = PSTNode()
        self.max_depth = max_depth

    def train(self, tokenized_sentences):
        for tokens in tokenized_sentences:
            tokens = tokens + ["<EOS>"]
            for i in range(len(tokens)):
                for d in range(self.max_depth + 1):
                    if i - d < 0:
                        continue
                    context = tuple(tokens[i - d:i])
                    word = tokens[i]
                    node = self._get_node(context, create=True)
                    node.next_words[word] += 1
                    node.total += 1

    def _get_node(self, context, create=False):
        node = self.root
        for w in context:
            if w not in node.children:
                if create:
                    node.children[w] = PSTNode()
                else:
                    return None
            node = node.children[w]
        return node

    def predict_next_token(self, context, temperature=1.0, top_p=0.9, encoder=None, alpha=0.7):
        context = deque(context, maxlen=self.max_depth)
        for d in range(len(context), -1, -1):
            sub_context = tuple(list(context)[-d:]) if d > 0 else tuple()
            node = self._get_node(sub_context)
            if node and node.total > 0:
                words, counts = zip(*node.next_words.items())
                probs = [c / node.total for c in counts]

                if temperature != 1.0:
                    probs = [p ** (1.0 / temperature) for p in probs]
                    s = sum(probs)
                    probs = [p / s for p in probs]

                if encoder is not None:
                    context_vector = encoder.encode(list(context))
                    p_enc = encoder.encoder_probs(context_vector, words)
                    probs = encoder.combine_probs(probs, p_enc, alpha=alpha)

                if "Bot:" in words:
                    idx = words.index("Bot:")
                    words = list(words)
                    probs = list(probs)
                    del words[idx]
                    del probs[idx]

                    # ⚠️ 중요! 확률 다시 정규화
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                    else:
                        return "<EOS>"  # 예측할 게 없으면 끝내기


                if "User:" in words:
                    idx = words.index("User:")
                    words = list(words)
                    probs = list(probs)
                    del words[idx]
                    del probs[idx]

                    # ⚠️ 중요! 확률 다시 정규화
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                    else:
                        return "<EOS>"  # 예측할 게 없으면 끝내기



                next_word = top_p_sampling(words, probs, p=top_p)
                return next_word
        return "<EOS>"


    def generate_sentence_stream(self, seed=None, max_len=70, temperature=10.0, top_p=0.9, encoder=None, alpha=0.7):
        if seed is None:
            context = []
        elif isinstance(seed, str):
            context = seed.strip().split()
        else:
            context = list(seed)
        generated = context.copy()
        for _ in range(max_len):
            next_token = self.predict_next_token(context, temperature=temperature, top_p=top_p, encoder=encoder, alpha=alpha)
            if next_token == "<EOS>":
                break
            generated.append(next_token)
            context.append(next_token)
            if len(context) > self.max_depth:
                context = context[-self.max_depth:]
            yield next_token


class TransformerEncoder:
    def __init__(self, vocab, embed_dim=1024, num_heads=16):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.vocab = vocab
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.word2idx = {w:i for i,w in enumerate(vocab)}
        self.vocab_size = len(vocab)

        # 임베딩 초기화
        self.embeddings = np.random.randn(self.vocab_size, embed_dim) * 0.1

        # 멀티헤드용 Q, K, V 가중치: (embed_dim, embed_dim)
        self.Wq = np.random.randn(embed_dim, embed_dim) * 0.1
        self.Wk = np.random.randn(embed_dim, embed_dim) * 0.1
        self.Wv = np.random.randn(embed_dim, embed_dim) * 0.1
        self.Wo = np.random.randn(embed_dim, embed_dim) * 0.1

        # FFN 가중치
        self.W1 = np.random.randn(embed_dim, embed_dim * 4) * 0.1
        self.b1 = np.zeros(embed_dim * 4)
        self.W2 = np.random.randn(embed_dim * 4, embed_dim) * 0.1
        self.b2 = np.zeros(embed_dim)

        # LayerNorm 변수
        self.eps = 1e-6
        self.ln_scale1 = np.ones(embed_dim)
        self.ln_bias1 = np.zeros(embed_dim)
        self.ln_scale2 = np.ones(embed_dim)
        self.ln_bias2 = np.zeros(embed_dim)

    def layer_norm(self, x, scale, bias):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return scale * x_norm + bias

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def split_heads(self, x):
        # x shape: (seq_len, embed_dim)
        seq_len = x.shape[0]
        # (seq_len, num_heads, head_dim)
        return x.reshape(seq_len, self.num_heads, self.head_dim)

    def combine_heads(self, x):
        # x shape: (seq_len, num_heads, head_dim)
        seq_len = x.shape[0]
        return x.reshape(seq_len, self.embed_dim)

    def encode(self, tokens):
        seq_len = len(tokens)
        x = np.zeros((seq_len, self.embed_dim))
        for i, w in enumerate(tokens):
            idx = self.word2idx.get(w, None)
            if idx is None:
                x[i] = np.zeros(self.embed_dim)
            else:
                x[i] = self.embeddings[idx]

        # 선형변환 Q, K, V (seq_len, embed_dim)
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        # 멀티헤드로 나누기 (seq_len, num_heads, head_dim)
        Qh = self.split_heads(Q)
        Kh = self.split_heads(K)
        Vh = self.split_heads(V)

        # scaled dot-product attention 계산
        # Qh @ Kh^T : (seq_len, num_heads, head_dim) @ (seq_len, num_heads, head_dim)^T 
        # --> (num_heads, seq_len, seq_len) 형태로 바꿔서 계산할 필요 있음

        # 편의를 위해 전치 및 브로드캐스트 처리
        scores = np.zeros((self.num_heads, seq_len, seq_len))
        for h in range(self.num_heads):
            q = Qh[:, h, :]  # (seq_len, head_dim)
            k = Kh[:, h, :]  # (seq_len, head_dim)
            scores[h] = (q @ k.T) / np.sqrt(self.head_dim)

        attn = np.zeros_like(scores)
        for h in range(self.num_heads):
            attn[h] = self.softmax(scores[h])

        # attention 결과와 Vh 곱하기 (num_heads, seq_len, seq_len) @ (seq_len, head_dim) -> (seq_len, head_dim)
        attn_out = np.zeros((seq_len, self.num_heads, self.head_dim))
        for h in range(self.num_heads):
            attn_out[:, h, :] = attn[h] @ Vh[:, h, :]

        # 멀티헤드 결과 합치기
        attn_out_combined = self.combine_heads(attn_out)  # (seq_len, embed_dim)
        attn_out_proj = attn_out_combined @ self.Wo  # (seq_len, embed_dim)

        # Residual + LayerNorm
        x = self.layer_norm(x + attn_out_proj, self.ln_scale1, self.ln_bias1)

        # FFN
        ffn_out = np.tanh(x @ self.W1 + self.b1)
        ffn_out = ffn_out @ self.W2 + self.b2

        # Residual + LayerNorm
        out = self.layer_norm(x + ffn_out, self.ln_scale2, self.ln_bias2)

        # 문맥 벡터로 각 위치 벡터 평균값 리턴
        context_vector = out.mean(axis=0)
        return context_vector

    def encoder_probs(self, context_vector, candidate_words):
        scores = []
        for w in candidate_words:
            idx = self.word2idx.get(w, None)
            if idx is None:
                scores.append(-1e9)
            else:
                emb = self.embeddings[idx]
                score = np.dot(context_vector, emb)
                scores.append(score)
        max_score = max(scores)
        exp_scores = [np.exp(s - max_score) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [s / sum_exp for s in exp_scores]
        return probs

    def combine_probs(self, p_pst, p_enc, alpha=0.1):
        p_pst = np.array(p_pst)
        p_enc = np.array(p_enc)
        p_pst = p_pst / (p_pst.sum() + 1e-8)
        p_enc = p_enc / (p_enc.sum() + 1e-8)
        p_final = alpha * p_pst + (1 - alpha) * p_enc
        p_final = p_final / (p_final.sum() + 1e-8)
        return p_final.tolist()



# --- 메인 함수 ---
if __name__ == "__main__":
    # 1) SentencePiece 모델 로드 (미리 학습된 모델 필요)
    spm_model_path = "korean_spm_bpe.model"
    spm_tokenizer = spm.SentencePieceProcessor()
    spm_tokenizer.Load(spm_model_path)

    # 2) 코퍼스 불러오기 및 SentencePiece 토큰화
    with open("corpus.txt", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    tokenized_corpus = [spm_tokenizer.EncodeAsPieces(line) for line in corpus]

    # 3) vocab 준비 (SentencePiece vocab 그대로 사용)
    vocab = spm_tokenizer.GetPieceSize()
    vocab_list = [spm_tokenizer.IdToPiece(i) for i in range(vocab)]

    # 4) 인코더, PST 초기화 & 학습
    encoder = TransformerEncoder(vocab_list)
    pst = PST(max_depth=3)  # max_depth 줄임 (노드 폭발 방지)
    pst.train(tokenized_corpus)

    print("✅ PST 학습 완료!")

    while True:
        user_input = input("\n컨텍스트 입력 (종료: exit): ").strip()
        if user_input.lower() == "exit":
            break
        bot_response = ""
        ""
        print("생성문장: ", end='', flush=True)
        for word in pst.generate_sentence_stream(seed=user_input, max_len=70, temperature=0.7, top_p=0.9, encoder=encoder, alpha=0.7):
            clean_word = word.replace('▁', ' ')
            print(clean_word, end='', flush=True)
            bot_response += clean_word
            time.sleep(0.05)
        print()
