import random
from collections import defaultdict, deque
import sys
import time

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

class PSTNode:
    def __init__(self):
        self.next_words = defaultdict(int)
        self.total = 0
        self.children = {}

class PST:
    def __init__(self, max_depth=3):
        self.root = PSTNode()
        self.max_depth = max_depth

    def train(self, sentences):
        for sentence in sentences:
            tokens = sentence.strip().split() + ["<EOS>"]
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

    def predict_next_token(self, context, temperature=1.0, top_p=0.9):
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

                next_word = top_p_sampling(words, probs, p=top_p)
                return next_word
        return "<EOS>"

    # 제너레이터로 변경해서 단어씩 yield
    def generate_sentence_stream(self, seed=None, max_len=20, temperature=1.0, top_p=0.9):
        if seed:
            context = seed.strip().split()
        else:
            context = []
        generated = context.copy()

        for _ in range(max_len):
            next_token = self.predict_next_token(context, temperature=temperature, top_p=top_p)
            if next_token == "<EOS>":
                break
            generated.append(next_token)
            context.append(next_token)
            if len(context) > self.max_depth:
                context = context[-self.max_depth:]
            yield next_token

if __name__ == "__main__":
    with open("corpus.txt", encoding="utf-8") as f:
        corpus = [line.strip() for line in f if line.strip()]

    pst = PST(max_depth=4)
    pst.train(corpus)

    print("PST 학습 완료!")

    while True:
        user_input = input("\n컨텍스트 입력 (종료: exit): ").strip()
        if user_input.lower() == "exit":
            break

        print("생성문장: ", end='', flush=True)
        for word in pst.generate_sentence_stream(seed=user_input, max_len=70, temperature=0.7, top_p=0.9):
            print(word, end=' ', flush=True)
            time.sleep(0.1)  # 살짝 텀 주기 (원하면 삭제 가능)
        print()