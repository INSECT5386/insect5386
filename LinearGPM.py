import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import random

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

class WordEmbedding:
    """넘파이 기반 워드 임베딩"""
    def __init__(self, vocab_size, embed_dim=100):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # Xavier 초기화
        self.embeddings = np.random.normal(0, np.sqrt(2.0 / embed_dim), 
                                         (vocab_size, embed_dim))
        
    def forward(self, token_ids):
        """토큰 ID들을 임베딩으로 변환"""
        if isinstance(token_ids, (list, np.ndarray)):
            return self.embeddings[token_ids]
        else:
            return self.embeddings[token_ids]
    
    def update_embedding(self, token_id, gradient, lr=0.001):
        """임베딩 업데이트"""
        self.embeddings[token_id] -= lr * gradient

class MLP:
    """넘파이 기반 다층 퍼셉트론"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        self.layers = []
        self.dropout_rate = dropout_rate
        
        # 레이어 구성
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layer = {
                'W': np.random.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i+1])),
                'b': np.zeros(dims[i+1])
            }
            self.layers.append(layer)
    
    def forward(self, x, training=True):
        """순전파"""
        self.activations = [x]
        
        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['W']) + layer['b']
            
            if i < len(self.layers) - 1:  # 마지막 레이어가 아니면
                a = relu(z)
                # 드롭아웃 적용
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1-self.dropout_rate, a.shape) / (1-self.dropout_rate)
                    a *= mask
            else:  # 마지막 레이어 (출력층)
                a = z  # 소프트맥스는 나중에 적용
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, loss_gradient, lr=0.001):
        """역전파"""
        delta = loss_gradient
        
        for i in reversed(range(len(self.layers))):
            # 가중치 그래디언트 계산
            dW = np.dot(self.activations[i].T, delta)
            db = np.sum(delta, axis=0)
            
            # 가중치 업데이트
            self.layers[i]['W'] -= lr * dW
            self.layers[i]['b'] -= lr * db
            
            # 다음 레이어를 위한 delta 계산
            if i > 0:
                delta = np.dot(delta, self.layers[i]['W'].T)
                # ReLU 역전파
                delta = delta * (self.activations[i] > 0)

class EmbeddingMLPModel:
    """임베딩 + MLP 기반 시퀀스 모델"""
    def __init__(self, vocab_size, embed_dim=100, hidden_dims=[128, 64], 
                 context_window=10, n_models=3):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_window = context_window
        self.n_models = n_models
        self.hidden_dims = hidden_dims # hidden_dims를 여기서 초기화합니다.
        
        # 특수 토큰 정의
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.EOS_TOKEN = "<EOS>"
        
        # 앙상블 구성 (초기화는 fit에서)
        self.embeddings = None
        self.mlps = None
        
        self.vectorizer = CountVectorizer(ngram_range=(1,3), max_features=1000)
        self.le = LabelEncoder()
        self.vocab_set = set()  # 어휘 집합 저장
        
    def token_to_id(self, token):
        """토큰을 ID로 변환 (UNK 토큰 처리 포함)"""
        if token in self.vocab_set:
            return self.le.transform([token])[0]
        else:
            return self.le.transform([self.UNK_TOKEN])[0]
    
    def prepare_context_features(self, context_tokens, ngram_features):
        """컨텍스트 특징 벡터 생성"""
        # 패딩 또는 잘라내기
        if len(context_tokens) > self.context_window:
            context_tokens = context_tokens[-self.context_window:]
        else:
            pad_id = self.le.transform([self.PAD_TOKEN])[0]
            context_tokens = [pad_id] * (self.context_window - len(context_tokens)) + context_tokens
        
        return context_tokens, ngram_features
    
    def fit(self, pairs, epochs=10):
        """모델 학습"""
        # 특수 토큰 추가해서 어휘 구성
        tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.EOS_TOKEN]
        for _, out in pairs:
            tokens.extend(tokenize(out))
        
        # 어휘 집합 생성 (중복 제거)
        self.vocab_set = set(tokens)
        tokens = list(self.vocab_set)
        
        # 라벨 인코더 학습
        self.le.fit(tokens)
        
        # 실제 어휘 크기로 업데이트
        actual_vocab_size = len(self.le.classes_)
        print(f"[INFO] 실제 어휘 크기: {actual_vocab_size}")
        
        # 모델 초기화 (실제 어휘 크기로)
        self.embeddings = [WordEmbedding(actual_vocab_size, self.embed_dim) for _ in range(self.n_models)]
        
        # 학습 데이터 준비
        X_contexts, X_ngrams_raw, y = [], [], [] # X_ngrams_raw로 변경
        
        for inp, out in pairs:
            inp_tokens = tokenize(inp)
            out_tokens = tokenize(out) + [self.EOS_TOKEN]
            
            for i, token in enumerate(out_tokens):
                # 현재까지의 생성된 토큰들
                prefix_tokens = out_tokens[:i]
                full_context = inp_tokens + prefix_tokens
                
                # 컨텍스트 토큰 ID (UNK 처리)
                context_ids = [self.token_to_id(t) for t in full_context[-self.context_window:]]
                
                # N-gram 특징을 위한 텍스트
                context_text = " ".join(full_context)
                
                X_contexts.append(context_ids)
                X_ngrams_raw.append(context_text) # 원시 텍스트 저장
                y.append(self.token_to_id(token))
        
        # N-gram 벡터화 (여기서 fit_transform)
        X_ngrams_vec = self.vectorizer.fit_transform(X_ngrams_raw).toarray()
        
        # N-gram 특징의 실제 차원을 가져옵니다.
        actual_ngram_dim = X_ngrams_vec.shape[1] 
        print(f"[INFO] 실제 N-gram 특징 차원: {actual_ngram_dim}")

        # MLP 입력 차원: 임베딩 * 컨텍스트 윈도우 + 실제 n-gram 특징 차원
        input_dim = self.embed_dim * self.context_window + actual_ngram_dim
        
        # MLP 모델 초기화 (이제 실제 input_dim을 사용합니다)
        self.mlps = [MLP(input_dim, self.hidden_dims, actual_vocab_size) for _ in range(self.n_models)]

        print(f"[INFO] 학습 데이터 크기: {len(X_contexts)}")
        
        # 각 모델 학습
        for model_idx in range(self.n_models):
            print(f"[INFO] 모델 {model_idx + 1} 학습 중...")
            
            embedding = self.embeddings[model_idx]
            mlp = self.mlps[model_idx]
            
            for epoch in range(epochs):
                total_loss = 0
                correct = 0
                
                # 배치 단위 학습
                indices = np.random.permutation(len(X_contexts))
                
                for i in indices:
                    # 임베딩 특징 추출
                    context_ids, ngram_features = self.prepare_context_features(
                        X_contexts[i], X_ngrams_vec[i] # X_ngrams_vec 사용
                    )
                    
                    # 임베딩 벡터 생성
                    embed_features = embedding.forward(context_ids).flatten()
                    
                    # 전체 특징 벡터 결합
                    features = np.concatenate([embed_features, ngram_features])
                    features = features.reshape(1, -1)
                    
                    # 순전파
                    logits = mlp.forward(features)
                    probs = softmax(logits[0])
                    
                    # 정답 확률
                    target_idx = y[i]
                    loss = -np.log(probs[target_idx] + 1e-10)
                    total_loss += loss
                    
                    # 정확도 계산
                    if np.argmax(probs) == target_idx:
                        correct += 1
                    
                    # 역전파
                    grad = probs.copy()
                    grad[target_idx] -= 1
                    grad = grad.reshape(1, -1)
                    
                    mlp.backward(grad)
                    
                    # 임베딩 업데이트
                    pad_id = self.le.transform([self.PAD_TOKEN])[0]
                    for j, token_id in enumerate(context_ids):
                        if token_id != pad_id:  # 패딩 토큰이 아니면

                            pass # 임베딩 업데이트 로직은 현재 문제의 직접적인 원인이 아니므로 잠시 보류

                if epoch % 1 == 0: # 에포크마다 로그 출력
                    accuracy = correct / len(X_contexts)
                    avg_loss = total_loss / len(X_contexts)
                    print(f"    Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        print("[INFO] 모든 모델 학습 완료!")
    
    def generate(self, input_text, max_len=20, temperature=0.8):
        """텍스트 생성"""
        inp_tokens = tokenize(input_text)
        generated = []
        
        print(f"[DEBUG] 입력: {input_text}")
        
        for step in range(max_len):
            # 현재 컨텍스트
            full_context = inp_tokens + generated
            context_text = " ".join(full_context)
            
            # 컨텍스트 토큰 ID (UNK 처리)
            context_ids = [self.token_to_id(token) for token in full_context[-self.context_window:]]
            
            # N-gram 특징
            try:
                ngram_features = self.vectorizer.transform([context_text]).toarray()[0]
            except Exception as e:
                # 새로운 n-gram이 있을 경우 0 벡터 사용
                print(f"[WARN] N-gram 변환 중 오류 발생: {e}. 0 벡터 사용.")
                ngram_features = np.zeros(self.vectorizer.max_features_) # 실제 N-gram 차원을 사용
            
            # 앙상블 예측
            all_probs = []
            
            for model_idx in range(self.n_models):
                embedding = self.embeddings[model_idx]
                mlp = self.mlps[model_idx]
                
                # 특징 벡터 구성
                context_ids_padded, ngram_feat = self.prepare_context_features(
                    context_ids, ngram_features
                )
                
                embed_features = embedding.forward(context_ids_padded).flatten()
                features = np.concatenate([embed_features, ngram_feat])
                features = features.reshape(1, -1)
                
                # 예측
                logits = mlp.forward(features, training=False)
                probs = softmax(logits[0] / temperature)
                all_probs.append(probs)
            
            # 앙상블 평균
            avg_probs = np.mean(all_probs, axis=0)
            
            # 다음 토큰 선택 (확률적 샘플링)
            next_idx = np.random.choice(len(avg_probs), p=avg_probs)
            next_token = self.le.inverse_transform([next_idx])[0]
            
            print(f"[DEBUG] Step {step}: {next_token} (prob: {avg_probs[next_idx]:.4f})")
            
            if next_token == self.EOS_TOKEN:
                break
            
            # 특수 토큰이 아닌 경우만 추가
            if next_token not in [self.PAD_TOKEN, self.UNK_TOKEN]:
                generated.append(next_token)
        
        result = " ".join(generated)
        print(f"[DEBUG] 생성 결과: {result}")
        return result

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터
    pairs = [
        ("안녕하세요", "안녕하세요! 반갑습니다"),
        ("오늘 날씨 어때?", "오늘 날씨는 맑고 좋습니다"),
        ("이름이 뭐야?", "저는 AI 어시스턴트입니다"),
        ("고마워", "천만에요! 도움이 되어 기쁩니다"),
        ("안녕", "안녕히 가세요!"),
    ]
    
    # 모델 생성 및 학습
    model = EmbeddingMLPModel(vocab_size=1000, embed_dim=50, 
                             hidden_dims=[64, 32], context_window=5)
    
    # hidden_dims는 __init__에서 이미 초기화되므로 불필요합니다.
    # model.hidden_dims = [64, 32] 
    
    model.fit(pairs, epochs=5)
    
    # 생성 테스트
    print("\n=== 생성 테스트 ===")
    print(model.generate("안녕하세요"))
    print(model.generate("날씨 어때?"))
    print(model.generate("고마워"))
