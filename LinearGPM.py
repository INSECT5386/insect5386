import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import random
import requests

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://raw.githubusercontent.com/INSECT5386/SeQRoN/main/data.csv?spm=a2ty_o01.29997173.0.0.7edec921fx1gz3&file=data.csv', 'MLdata.csv')


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
            # NumPy 배열로 변환하여 인덱싱
            return self.embeddings[np.array(token_ids)]
        else:
            return self.embeddings[token_ids]
    
    def update_embedding(self, token_id, gradient, lr=0.001):
        """임베딩 업데이트"""
        # gradient는 (1, embed_dim) 형태여야 함
        self.embeddings[token_id] -= lr * gradient.flatten() # flatten으로 1차원으로 만듦

class MLP:
    """넘파이 기반 다층 퍼셉트론"""
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        self.layers = []
        self.dropout_rate = dropout_rate
        self.activations = [] # 순전파의 각 층 활성화 값을 저장
        self.zs = [] # 순전파의 각 층 가중합 값을 저장 (ReLU 역전파를 위해)

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
        self.zs = []
        
        for i, layer in enumerate(self.layers):
            z = np.dot(self.activations[-1], layer['W']) + layer['b']
            self.zs.append(z) # z 값 저장

            if i < len(self.layers) - 1:  # 마지막 레이어가 아니면
                a = relu(z)
                # 드롭아웃 적용
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1-self.dropout_rate, a.shape) / (1-self.dropout_rate)
                    a *= mask
                self.activations.append(a)
            else:  # 마지막 레이어 (출력층)
                a = z  # 소프트맥스는 나중에 적용
                self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, loss_gradient, lr=0.001):
        """역전파"""
        delta = loss_gradient
        
        # 입력에 대한 최종 그래디언트를 저장
        input_gradient = None 

        for i in reversed(range(len(self.layers))):
            current_activation = self.activations[i] # 이전 레이어의 활성화 출력 (현재 레이어의 입력)
            
            # 가중치 그래디언트 계산
            dW = np.dot(current_activation.T, delta)
            db = np.sum(delta, axis=0)
            
            # 가중치 업데이트
            self.layers[i]['W'] -= lr * dW
            self.layers[i]['b'] -= lr * db
            
            # 다음 레이어를 위한 delta 계산 (이전 레이어의 출력, 즉 현재 레이어의 입력에 대한 그래디언트)
            if i > 0: # 입력 레이어가 아니면
                # 현재 레이어의 입력에 대한 그래디언트 (다음 층으로 전달될 delta)
                delta = np.dot(delta, self.layers[i]['W'].T)
                # ReLU 활성화 함수의 역전파
                delta = delta * (self.zs[i-1] > 0) # 이전 층의 z값 사용
            else: # 입력 레이어
                input_gradient = np.dot(delta, self.layers[i]['W'].T)
        
        return input_gradient # MLP의 입력에 대한 그래디언트 반환

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
        # le.transform은 입력이 리스트여야 함
        if token in self.vocab_set:
            return self.le.transform([token])[0]
        else:
            return self.le.transform([self.UNK_TOKEN])[0]
    
    def prepare_context_features(self, context_token_ids, ngram_features):
        """컨텍스트 특징 벡터 생성"""
        # 패딩 또는 잘라내기
        if len(context_token_ids) > self.context_window:
            context_token_ids = context_token_ids[-self.context_window:]
        else:
            pad_id = self.le.transform([self.PAD_TOKEN])[0]
            context_token_ids = [pad_id] * (self.context_window - len(context_token_ids)) + context_token_ids
        
        return context_token_ids, ngram_features # 이제 context_token_ids는 항상 context_window 길이

    def fit(self, pairs, epochs=10, lr=0.001):
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
        X_contexts, X_ngrams_raw, y = [], [], [] 
        
        for inp, out in pairs:
            inp_tokens = tokenize(inp)
            out_tokens = tokenize(out) + [self.EOS_TOKEN]
            
            for i, token in enumerate(out_tokens):
                prefix_tokens = out_tokens[:i]
                full_context = inp_tokens + prefix_tokens
                
                # 컨텍스트 토큰 ID (UNK 처리)
                context_ids_for_sample = [self.token_to_id(t) for t in full_context] # 모든 토큰을 ID로 변환

                # N-gram 특징을 위한 텍스트
                context_text = " ".join(full_context)
                
                X_contexts.append(context_ids_for_sample) # 원본 토큰 ID 리스트 저장 (prepare_context_features에서 처리)
                X_ngrams_raw.append(context_text) 
                y.append(self.token_to_id(token))
        
        # N-gram 벡터화 (여기서 fit_transform)
        X_ngrams_vec = self.vectorizer.fit_transform(X_ngrams_raw).toarray()
        
        actual_ngram_dim = X_ngrams_vec.shape[1] 
        print(f"[INFO] 실제 N-gram 특징 차원: {actual_ngram_dim}")

        input_dim = self.embed_dim * self.context_window + actual_ngram_dim
        
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
                    # 임베딩 특징 추출 및 N-gram 특징 준비
                    context_token_ids_padded, ngram_features_processed = self.prepare_context_features(
                        X_contexts[i], X_ngrams_vec[i]
                    )
                    
                    # 임베딩 벡터 생성
                    embed_features = embedding.forward(context_token_ids_padded).flatten()
                    
                    # 전체 특징 벡터 결합
                    features = np.concatenate([embed_features, ngram_features_processed])
                    features = features.reshape(1, -1) # (1, input_dim)
                    
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
                    grad = probs.copy() # (vocab_size,)
                    grad[target_idx] -= 1
                    grad = grad.reshape(1, -1) # (1, vocab_size)
                    
                    # MLP 역전파 수행 및 입력 그래디언트 얻기
                    input_grad = mlp.backward(grad, lr=lr) # (1, input_dim)

                    # 임베딩 부분에 해당하는 그래디언트 추출
                    # input_grad는 (1, embed_dim * context_window + ngram_dim) 형태
                    embedding_grad_flat = input_grad[:, :self.embed_dim * self.context_window].flatten()

                    # 각 임베딩 토큰에 대한 그래디언트 분배 및 업데이트
                    # context_token_ids_padded는 (context_window,) 길이
                    # embedding_grad_flat은 (embed_dim * context_window,) 길이
                    # 이를 (context_window, embed_dim) 형태로 재구성해야 함
                    embedding_grad_reshaped = embedding_grad_flat.reshape(self.context_window, self.embed_dim)

                    for j, token_id in enumerate(context_token_ids_padded):
                        # 패딩 토큰이 아니면 업데이트
                        # PAD_TOKEN의 ID를 미리 얻어두는 것이 효율적
                        if token_id != self.le.transform([self.PAD_TOKEN])[0]:
                            embedding.update_embedding(token_id, embedding_grad_reshaped[j], lr=lr)

                if epoch % 1 == 0: 
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
            context_ids_for_sample = [self.token_to_id(token) for token in full_context]
            
            # N-gram 특징
            try:
                # transform은 2D 배열을 기대하므로 [context_text]로 전달
                ngram_features = self.vectorizer.transform([context_text]).toarray()[0]
            except Exception as e:
                print(f"[WARN] N-gram 변환 중 오류 발생: {e}. 0 벡터 사용.")
                # self.vectorizer.max_features_가 실제 피처 수를 알려줌
                ngram_features = np.zeros(self.vectorizer.max_features_) 
            
            # 앙상블 예측
            all_probs = []
            
            for model_idx in range(self.n_models):
                embedding = self.embeddings[model_idx]
                mlp = self.mlps[model_idx]
                
                # 특징 벡터 구성
                context_ids_padded, ngram_feat_processed = self.prepare_context_features(
                    context_ids_for_sample, ngram_features
                )
                
                embed_features = embedding.forward(context_ids_padded).flatten()
                features = np.concatenate([embed_features, ngram_feat_processed])
                features = features.reshape(1, -1)
                
                # 예측 (추론 시에는 training=False)
                logits = mlp.forward(features, training=False)
                probs = softmax(logits[0] / temperature)
                all_probs.append(probs)
            
            # 앙상블 평균
            avg_probs = np.mean(all_probs, axis=0)
            
            # 다음 토큰 선택 (확률적 샘플링)
            # 확률 분포가 0인 경우를 방지하기 위해 정규화
            avg_probs = avg_probs / np.sum(avg_probs)
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



if __name__ == "__main__":
    # 데이터셋 경로와 읽기
    import csv

    csv_path = "MLdata.csv"  # 여기에 네 데이터셋 경로 넣어라

    # CSV에서 (input, output) 쌍 읽기
    pairs = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inp = row["input_text"].strip()
            out = row["output_text"].strip()
            pairs.append((inp, out))

    
    # 모델 생성 및 학습
    model = EmbeddingMLPModel(vocab_size=1000, embed_dim=50, 
                             hidden_dims=[64, 32], context_window=5, n_models=3)
    
    model.fit(pairs, epochs=10, lr=0.005) # 학습률 조정
    
    # 생성 테스트
    print("\n=== 생성 테스트 ===")
    print(f"User: 안녕하세요 -> Model: {model.generate('안녕하세요')}")
    print(f"User: 날씨 어때? -> Model: {model.generate('날씨 어때?')}")
    print(f"User: 고마워 -> Model: {model.generate('고마워')}")
    print(f"User: 배고파 -> Model: {model.generate('배고파')}")
    print(f"User: 사랑해 -> Model: {model.generate('사랑해')}")

