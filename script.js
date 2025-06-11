document.addEventListener('DOMContentLoaded', () => {
    const chatHeader = document.getElementById('chat-header');
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const modelSuggestionsDiv = document.getElementById('model-suggestions');
    const modelButtons = document.querySelectorAll('.model-button');

    let selectedModel = null;
    let apiUrl = null;

    const models = [
        { name: 'Flexi', url: 'https://yuchan5386-flexi-api.hf.space/generate', desc: 'Flexi는 InteractGPT의 개선모델로, 다양한 분석 기능과 유연한 응답을 제공합니다.' },
        { name: 'KeraLux', url: 'https://yuchan5386-keralux-api.hf.space/generate', desc: 'KeraLux는 180만 개 한국어 데이터로 사전학습된 GPT 기반 모델로, 한국어 최적화와 자연스러운 대화를 지원합니다.' },
        { name: 'InteractGPT', url: 'https://yuchan5386-interactgpt-api.hf.space/generate', desc: 'InteractGPT는 대화형 GPT 모델로, 간단한 일상대화를 지원합니다.' },
    ];

    const displayMessage = (message, sender) => {
        const el = document.createElement('div');
        el.className = `chat-message ${sender}`;
        el.innerHTML = `<p>${message}</p>`;
        chatContainer.appendChild(el);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const selectModel = (name, url) => {
        selectedModel = name;
        apiUrl = url;
        chatHeader.textContent = `Ector.V - ${name} 모델 채팅`;
        displayMessage(`${name} 모델이 선택되었습니다. 이제 메시지를 입력하세요!`, 'system');
        promptInput.placeholder = `메시지를 입력하세요.`;
        modelSuggestionsDiv.classList.remove('active');
        promptInput.focus();
    };

    const filterModels = (query) => {
        return models.filter(model => model.name.toLowerCase().includes(query));
    };

    displayMessage('안녕하세요! Flexi, KeraLux, InteractGPT 중 하나의 모델을 선택하거나 메시지를 입력하여 채팅을 시작하세요.', 'system');

    promptInput.addEventListener('input', () => {
        const query = promptInput.value.trim().toLowerCase();
        modelSuggestionsDiv.innerHTML = '';

        if (query && !selectedModel) {
            const results = filterModels(query);
            if (results.length) {
                results.forEach(model => {
                    const item = document.createElement('div');
                    item.className = 'suggestion-item';
                    item.textContent = model.name;
                    item.onclick = () => {
                        selectModel(model.name, model.url);
                        promptInput.value = '';
                    };
                    modelSuggestionsDiv.appendChild(item);
                });
                modelSuggestionsDiv.classList.add('active');
            } else {
                modelSuggestionsDiv.classList.remove('active');
            }
        } else {
            modelSuggestionsDiv.classList.remove('active');
        }
    });

    modelButtons.forEach(button => {
        button.addEventListener('click', () => {
            const name = button.dataset.model;
            const model = models.find(m => m.name === name);
            if (model) selectModel(model.name, model.url);
        });
    });

    async function sendMessage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        if (!selectedModel) {
            const model = models.find(m => m.name.toLowerCase() === prompt.toLowerCase());
            if (model) {
                selectModel(model.name, model.url);
                promptInput.value = '';
            } else {
                displayMessage('먼저 모델을 선택해 주세요 (예: Flexi, KeraLux, InteractGPT)', 'system');
            }
            return;
        }

        displayMessage(prompt, 'user');
        promptInput.value = '';
        sendBtn.disabled = true;

        const botMessage = document.createElement('div');
        botMessage.className = 'chat-message bot';
        botMessage.innerHTML = `『 "${prompt}" 에 대한 응답을 생성 중... 』<br><br>`;
        chatContainer.appendChild(botMessage);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const response = await fetch(`${apiUrl}?${new URLSearchParams({ prompt })}`);
            if (!response.body) throw new Error("응답 바디가 없습니다.");

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            let receivedText = '';
            let isFirst = true;

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    botMessage.innerHTML = receivedText + "<br>[✔️ 완료]";
                    break;
                }
                const chunk = decoder.decode(value, { stream: true });
                receivedText += isFirst ? chunk.trimStart() : chunk;
                botMessage.innerHTML = receivedText;
                isFirst = false;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        } catch (err) {
            botMessage.innerHTML = `[❌ 오류 발생] ${err.message}`;
            console.error(err);
        } finally {
            sendBtn.disabled = false;
            promptInput.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    promptInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
