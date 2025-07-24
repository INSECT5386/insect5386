import random
from collections import defaultdict, Counter
import re
import time
import math
import sentencepiece as spm
import requests

# ⬇️ 파일 다운로드 함수
def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/gpt_pretrain_small/resolve/main/remaining.txt?download=true', 'sampled.txt')
download_file('https://huggingface.co/datasets/Yuchan5386/gpt_pretrain_small/resolve/main/remaining.txt?download=true', 'remaining.txt')
download_file('https://huggingface.co/datasets/Yuchan5386/gpt_pretrain_small/resolve/main/gptbpe.model?download=true', 'gptbpe.model')


# BPE 모델 로드
sp = spm.SentencePieceProcessor()
sp.load('gptbpe.model')

def tokenize(text):
    return sp.encode(text, out_type=str)

QUESTION_TOKEN = "<Q>"
ANSWER_TOKEN = "<A>"
EOS_TOKEN = "<EOS>"  # 이미 있음


class PSAState:
    def __init__(self):
        self.transitions = defaultdict(int)  # token -> count
        self.total = 0  # 총 전이 횟수

    def add_transition(self, token):
        self.transitions[token] += 1
        self.total += 1

    def sample_next(self, temperature=1.0):
        tokens = list(self.transitions.keys())
        counts = list(self.transitions.values())

        # 안정적 softmax 계산
        max_count = max(counts)
        exps = [math.exp((c - max_count) / temperature) for c in counts]
        sum_exps = sum(exps)
        probs = [e / sum_exps for e in exps]

        return random.choices(tokens, probs)[0]

class SASPQnA:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.question_states = {}
        self.answer_states = {}
        self.qa_pairs = []
        self.qa_init_contexts = {}
        self.initial_prompt = None
        self.conversation_history = []

    def _add_sequence(self, tokens, state_dict):
        n = len(tokens)
        for i in range(n):
            for l in range(1, min(self.max_depth, i+1) + 1):
                suffix = tuple(tokens[i-l+1:i+1])
                prefix = suffix[:-1]
                if prefix not in state_dict:
                    state_dict[prefix] = PSAState()
                state_dict[prefix].add_transition(suffix[-1])

    def add_qa_pair(self, question_tokens, answer_tokens):
        # 질문과 답변 사이에 특수 토큰 넣기
        q_seq = [QUESTION_TOKEN] + question_tokens + [EOS_TOKEN]
        a_seq = [ANSWER_TOKEN] + answer_tokens + [EOS_TOKEN]

        self._add_sequence(q_seq, self.question_states)
        self._add_sequence(a_seq, self.answer_states)

        self.qa_pairs.append((q_seq, a_seq))

        for length in range(1, min(self.max_depth, len(q_seq)) + 1):
            prefix = tuple(q_seq[-length:])
            if prefix not in self.qa_init_contexts:
                self.qa_init_contexts[prefix] = []
            if a_seq[:self.max_depth-1] not in self.qa_init_contexts[prefix]:
                self.qa_init_contexts[prefix].append(a_seq[:self.max_depth-1])

    def set_initial_prompt(self, prompt_text):
        self.initial_prompt = tokenize(prompt_text)

    def levenshtein_distance(a, b, max_dist=2):
        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1
        dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
        for i in range(len(a)+1):
            dp[i][0] = i
        for j in range(len(b)+1):
            dp[0][j] = j
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 삭제
                    dp[i][j-1] + 1,      # 삽입
                    dp[i-1][j-1] + cost  # 교체
                )
                # 최적화: max_dist 넘으면 바로 탈출
                if min(dp[i][j], dp[i-1][j], dp[i][j-1]) > max_dist:
                    return max_dist + 1
        return dp[len(a)][len(b)]

    def find_best_question_state(self, query_tokens):
        max_dist = 2
        best_prefix = ()
        best_dist = max_dist + 1

        # 1) 완전 일치 우선
        for length in range(min(len(query_tokens), self.max_depth), 0, -1):
            prefix = tuple(query_tokens[-length:])
            if prefix in self.qa_init_contexts:
                return prefix

        # 2) 편집거리 기반 유사도 검색
        for length in range(min(len(query_tokens), self.max_depth), 0, -1):
            query_sub = tuple(query_tokens[-length:])
            for prefix in self.qa_init_contexts.keys():
                if len(prefix) == length:
                    dist = self.levenshtein_distance(prefix, query_sub, max_dist)
                    if dist <= max_dist and dist < best_dist:
                        best_dist = dist
                        best_prefix = prefix
                        if dist == 0:
                            return prefix
        return best_prefix

    def _is_similar(self, a, b, max_edits=1):
        if abs(len(a) - len(b)) > max_edits:
            return False
        edits = 0
        for x, y in zip(a, b):
            if x != y:
                edits += 1
                if edits > max_edits:
                    return False
        return True

    def generate_answer_stream(self, init_context=(), question_keywords=(), temperature=0.7, max_len=50, min_len=5):
        full_context = []
        if self.initial_prompt:
            full_context.extend(self.initial_prompt)
        full_context.extend(init_context)
        full_context.extend(question_keywords)

        context = list(full_context[-(self.max_depth - 1):]) if self.max_depth > 1 else []

        generated_len = 0
        first_token_generated = False

        skip_count = 0
        max_skip = 10  # 첫 토큰 구두점 무시 최대 횟수

        for _ in range(max_len):
            next_token = None
            for length in range(len(context), -1, -1):
                prefix = tuple(context[-length:]) if length > 0 else ()
                state = self.answer_states.get(prefix)
                if state and state.transitions:
                    candidate = state.sample_next(temperature)
                    if not first_token_generated and candidate in {',', '.', '!', '?'}:
                        skip_count += 1
                        if skip_count > max_skip:
                            next_token = candidate
                            break
                        continue
                    next_token = candidate
                    break

            if next_token is None:
                root_state = self.answer_states.get(())
                if root_state and root_state.transitions:
                    candidate = root_state.sample_next(temperature)
                    if not first_token_generated and candidate in {',', '.', '!', '?'}:
                        skip_count += 1
                        if skip_count > max_skip:
                            next_token = candidate
                        else:
                            continue
                    else:
                        next_token = candidate
                else:
                    break

            if next_token == '<EOS>' and generated_len < min_len:
                continue

            if next_token == '<EOS>':
                break

            context.append(next_token)
            generated_len += 1
            first_token_generated = True
            yield next_token


    # chat_stream에서 extract_keywords 제거

    def chat_stream(self, input_text, temperature=2.7):
        # 질문에 특수 토큰 <Q>와 <EOS> 추가
        query_tokens = [QUESTION_TOKEN] + tokenize(input_text) + [EOS_TOKEN]

        recent_context = []
        if self.conversation_history:
            recent_context = self.conversation_history[-(self.max_depth - 3):]

        question_state = self.find_best_question_state(query_tokens)

        init_contexts = self.qa_init_contexts.get(question_state, ())
        if isinstance(init_contexts, list) and init_contexts:
            if temperature < 1.0:
                init_context = tuple(init_contexts[0])
            else:
                init_context = tuple(random.choice(init_contexts))
        else:
            init_context = ()


        combined_init_context = tuple(recent_context) + init_context

        generated_tokens = []
        for token in self.generate_answer_stream(combined_init_context, query_tokens, temperature=temperature):
            generated_tokens.append(token)

        while generated_tokens and generated_tokens[0] in {',', '.', '!', '?'}:
            generated_tokens.pop(0)

        self.conversation_history.extend(query_tokens)
        self.conversation_history.extend(generated_tokens)

        for token in generated_tokens:
            yield token

def stream_print_bpe_tokens(token_generator):
    first = True
    for token in token_generator:
        if token.startswith('▁'):
            if first:
                print(token[1:], end='', flush=True)
                first = False
            else:
                print(' ' + token[1:], end='', flush=True)
        else:
            print(token, end='', flush=True)
        time.sleep(0.03)  # 토큰 하나 출력 후 잠깐 쉬기
    print()

import random

def load_txt_qa_pairs(file_path, max_samples=100_000):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines_buffer = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_buffer.append(line)
            # 2줄 모이면 하나의 QA 쌍으로 저장
            if len(lines_buffer) == 2:
                pairs.append(tuple(lines_buffer))
                lines_buffer.clear()
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    return pairs


def filter_special_tokens(tokens):
    specials = {QUESTION_TOKEN, ANSWER_TOKEN, EOS_TOKEN}
    return [t for t in tokens if t not in specials]



if __name__ == "__main__":
    print("코퍼스 로드 중...")
    qa_pairs = load_txt_qa_pairs("sampled.txt", max_samples=10_000)
    print(f"문장 {len(qa_pairs)*2}개 로드 완료!")

    bot = SASPQnA(max_depth=8)
    for q, a in qa_pairs:
        q_tokens = tokenize(q)
        a_tokens = tokenize(a)
        bot.add_qa_pair(q_tokens, a_tokens)


    # 초기 프롬프트 지정 (대화 초반 안정성 향상)
    bot.set_initial_prompt("안녕")

    print("학습 완료! 대화 시작 (종료: exit)")


    while True:
        user_input = input("질문: ").strip()
        if user_input.lower() == "exit":
            break

        print("답변: ", end="", flush=True)
        gen = bot.chat_stream(user_input, temperature=1.3)  # 2.7 → 1.3으로 낮춤
        filtered_tokens = filter_special_tokens(list(gen))
        stream_print_bpe_tokens(iter(filtered_tokens))
        time.sleep(0.03)
