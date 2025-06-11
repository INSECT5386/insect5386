document.addEventListener('DOMContentLoaded', () => {
    // Get all necessary DOM elements
    const chatHeader = document.getElementById('chat-header');
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const modelSuggestionsDiv = document.getElementById('model-suggestions');
    const modelButtons = document.querySelectorAll('.model-button');

    let selectedModel = null;
    let apiUrl = null;

    // Define available models with their URLs and descriptions
    const models = [
        { name: 'Flexi', url: 'https://yuchan5386-flexi-api.hf.space/generate', desc: 'Flexi는 InteractGPT의 개선모델로, 다양한 분석 기능과 유연한 응답을 제공합니다.' },
        { name: 'KeraLux', url: 'https://yuchan5386-keralux-api.hf.space/generate', desc: 'KeraLux는 180만 개 한국어 데이터로 사전학습된 GPT 기반 모델로, 한국어 최적화와 자연스러운 대화를 지원합니다.' },
        { name: 'InteractGPT', url: 'https://yuchan5386-interactgpt-api.hf.space/generate', desc: 'InteractGPT는 대화형 GPT 모델로, 간단한 일상대화를 지원합니다' },
        { name: 'Flexi2', url: 'https://yuchan5386-flexi-2-api.hf.space/generate', desc: 'Flexi의 개선 모델입니다' }
    ];

    // Append a message to the chat container
    const displayMessage = (message, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);
        messageElement.innerHTML = `<p>${message}</p>`;
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    // Select model and update UI
    const selectModel = (modelName, modelUrl) => {
        selectedModel = modelName;
        apiUrl = modelUrl;
        chatHeader.textContent = `Ector.V - ${selectedModel} 모델 채팅`;
        displayMessage(`${selectedModel} 모델이 선택되었습니다. 이제 메시지를 입력하세요!`, 'system');
        promptInput.placeholder = `메시지를 입력하세요.`;
        modelSuggestionsDiv.classList.remove('active');
        promptInput.focus();
    };

    // Display initial greeting
    displayMessage('안녕하세요! Flexi, KeraLux, InteractGPT 중 하나의 모델을 선택하거나 메시지를 입력하여 채팅을 시작하세요.', 'system');

    // Handle input suggestions
    promptInput.addEventListener('input', () => {
        const query = promptInput.value.trim().toLowerCase();
        modelSuggestionsDiv.innerHTML = '';

        if (query.length > 0 && !selectedModel) {
            const filteredModels = models.filter(model => model.name.toLowerCase().includes(query));
            if (filteredModels.length > 0) {
                filteredModels.forEach(model => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.classList.add('suggestion-item');
                    suggestionItem.textContent = model.name;
                    suggestionItem.addEventListener('click', () => {
                        selectModel(model.name, model.url);
                        promptInput.value = '';
                    });
                    modelSuggestionsDiv.appendChild(suggestionItem);
                });
                modelSuggestionsDiv.classList.add('active');
            } else {
                modelSuggestionsDiv.classList.remove('active');
            }
        } else {
            modelSuggestionsDiv.classList.remove('active');
        }
    });

    // Handle model selection via buttons
    modelButtons.forEach(button => {
        button.addEventListener('click', () => {
            const modelName = button.dataset.model;
            const foundModel = models.find(m => m.name === modelName);
            if (foundModel) {
                selectModel(foundModel.name, foundModel.url);
            }
        });
    });

    // Send message function
    async function sendMessage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        if (!selectedModel) {
            const foundModel = models.find(model => model.name.toLowerCase() === prompt.toLowerCase());
            if (foundModel) {
                selectModel(foundModel.name, foundModel.url);
                promptInput.value = '';
                return;
            } else {
                displayMessage('먼저 모델을 선택해 주세요 (예: Flexi, KeraLux, InteractGPT)', 'system');
                return;
            }
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
            const params = new URLSearchParams({ prompt });
            const response = await fetch(`${apiUrl}?${params}`);

            if (!response.body) throw new Error("응답 바디가 없습니다.");

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            let isFirstChunk = true;
            let receivedText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    botMessage.innerHTML = receivedText + "<br>[✔️ 완료]";
                    break;
                }

                const text = decoder.decode(value, { stream: true });
                receivedText += isFirstChunk ? text.trimStart() : text;
                botMessage.innerHTML = receivedText;
                isFirstChunk = false;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        } catch (err) {
            botMessage.innerHTML = `[❌ 오류 발생] ${err.message}`;
            console.error('Error sending message:', err);
        } finally {
            sendBtn.disabled = false;
            promptInput.focus();
        }
    }

    // Send message on button click
    sendBtn.addEventListener('click', sendMessage);

    // Send message on Enter key
    promptInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
