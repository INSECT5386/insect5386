import random
from collections import defaultdict
import re
import time
import math
import sentencepiece as spm
import requests
from multiprocessing import Pool, cpu_count

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
EOS_TOKEN = "<EOS>"

class PSAState:
    def __init__(self):
        self.transitions = defaultdict(int)
        self.total = 0

    def add_transition(self, token):
        self.transitions[token] += 1
        self.total += 1

    def sample_next(self, temperature=1.0):
        tokens = list(self.transitions.keys())
        counts = list(self.transitions.values())
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

    def set_initial_prompt(self, prompt_text):
        self.initial_prompt = tokenize(prompt_text)

    def levenshtein_distance(self, a, b, max_dist=2):
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
                    dp[i-1][j] + 1,
                    dp[i][j-1] + 1,
                    dp[i-1][j-1] + cost
                )
                if min(dp[i][j], dp[i-1][j], dp[i][j-1]) > max_dist:
                    return max_dist + 1
        return dp[len(a)][len(b)]

    def find_best_question_state(self, query_tokens):
        max_dist = 2
        best_prefix = ()
        best_dist = max_dist + 1
        for length in range(min(len(query_tokens), self.max_depth), 0, -1):
            prefix = tuple(query_tokens[-length:])
            if prefix in self.qa_init_contexts:
                return prefix
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
        max_skip = 10

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
            if next_token == EOS_TOKEN and generated_len < min_len:
                continue
            if next_token == EOS_TOKEN:
                break
            context.append(next_token)
            generated_len += 1
            first_token_generated = True
            yield next_token

    def chat_stream(self, input_text, temperature=2.7):
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
        time.sleep(0.03)
    print()

def load_txt_qa_pairs(file_path, max_samples=100_000):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines_buffer = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines_buffer.append(line)
            if len(lines_buffer) == 2:
                pairs.append(tuple(lines_buffer))
                lines_buffer.clear()
    if len(pairs) > max_samples:
        pairs = random.sample(pairs, max_samples)
    return pairs

def filter_special_tokens(tokens):
    specials = {QUESTION_TOKEN, ANSWER_TOKEN, EOS_TOKEN}
    return [t for t in tokens if t not in specials]

def batchify(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# 병렬 처리용
def process_qa_pair(args):
    q_tokens, a_tokens, max_depth = args
    local_q_states = {}
    local_a_states = {}
    local_init_contexts = {}

    def _add_sequence(tokens, state_dict):
        n = len(tokens)
        for i in range(n):
            for l in range(1, min(max_depth, i+1) + 1):
                suffix = tuple(tokens[i-l+1:i+1])
                prefix = suffix[:-1]
                if prefix not in state_dict:
                    state_dict[prefix] = PSAState()
                state_dict[prefix].add_transition(suffix[-1])

    q_seq = [QUESTION_TOKEN] + q_tokens + [EOS_TOKEN]
    a_seq = [ANSWER_TOKEN] + a_tokens + [EOS_TOKEN]

    _add_sequence(q_seq, local_q_states)
    _add_sequence(a_seq, local_a_states)

    for length in range(1, min(max_depth, len(q_seq)) + 1):
        prefix = tuple(q_seq[-length:])
        if prefix not in local_init_contexts:
            local_init_contexts[prefix] = []
        if a_seq[:max_depth-1] not in local_init_contexts[prefix]:
            local_init_contexts[prefix].append(a_seq[:max_depth-1])

    return (local_q_states, local_a_states, local_init_contexts, q_seq, a_seq)

def merge_states(main, new):
    for k, v in new.items():
        if k not in main:
            main[k] = PSAState()
        for token, count in v.transitions.items():
            main[k].transitions[token] += count
            main[k].total += count
from tqdm import tqdm

if __name__ == "__main__":
    print("코퍼스 로드 중...")
    qa_pairs = load_txt_qa_pairs("sampled.txt", max_samples=10_000)
    print(f"문장 {len(qa_pairs)*2}개 로드 완료!")

    max_depth = 8
    batch_size = 1000
    bot = SASPQnA(max_depth=max_depth)

    with Pool(processes=cpu_count()) as pool:
        for batch in tqdm(batchify(qa_pairs, batch_size), total=(len(qa_pairs) + batch_size - 1) // batch_size, desc="미니배치 학습 중"):
            inputs = [(tokenize(q), tokenize(a), max_depth) for q, a in batch]
            results = pool.map(process_qa_pair, inputs)

            for q_states, a_states, init_ctxs, q_seq, a_seq in results:
                merge_states(bot.question_states, q_states)
                merge_states(bot.answer_states, a_states)
                for k, v in init_ctxs.items():
                    if k not in bot.qa_init_contexts:
                        bot.qa_init_contexts[k] = []
                    for vv in v:
                        if vv not in bot.qa_init_contexts[k]:
                            bot.qa_init_contexts[k].append(vv)
                bot.qa_pairs.append((q_seq, a_seq))

    bot.set_initial_prompt("안녕")
    print("✅ 병렬+미니배치 학습 완료! 대화 시작 (종료: exit)")

    while True:
        user_input = input("질문: ").strip()
        if user_input.lower() == "exit":
            break
        print("답변: ", end="", flush=True)
        gen = bot.chat_stream(user_input, temperature=1.3)
        filtered_tokens = filter_special_tokens(list(gen))
        stream_print_bpe_tokens(iter(filtered_tokens))
        time.sleep(0.03)
