document.addEventListener('DOMContentLoaded', () => {
    // Get all necessary DOM elements
    const chatHeader = document.getElementById('chat-header');
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const modelSuggestionsDiv = document.getElementById('model-suggestions'); // New element for suggestions

    let selectedModel = null;
    let apiUrl = null;

    // Define available models with their URLs and descriptions
    const models = [
        { name: 'Flexi', url: 'https://yuchan5386-flexi-api.hf.space/generate', desc: 'Flexi는 InteractGPT의 개선모델로, 다양한 분석 기능과 유연한 응답을 제공합니다.' },
        { name: 'KeraLux', url: 'https://yuchan5386-keralux-api.hf.space/generate', desc: 'KeraLux는 180만 개 한국어 데이터로 사전학습된 GPT 기반 모델로, 한국어 최적화와 자연스러운 대화를 지원합니다.' },
        { name: 'InteractGPT', url: 'https://yuchan5386-interactgpt-api.hf.space/generate', desc: 'InteractGPT는 대화형 GPT 모델로, 간단한 일상대화를 지원합니다' }
    ];

    /**
     * Appends a message to the chat container.
     * @param {string} message - The text content of the message.
     * @param {string} sender - 'user', 'bot', or 'system' to apply appropriate styling.
     */
    const displayMessage = (message, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);
        messageElement.innerHTML = `<p>${message}</p>`; // Use innerHTML for potential Markdown
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to the bottom
    };

    /**
     * Sets the selected model and updates the UI.
     * @param {string} modelName - The name of the selected model.
     * @param {string} modelUrl - The API URL for the selected model.
     */
    const selectModel = (modelName, modelUrl) => {
        selectedModel = modelName;
        apiUrl = modelUrl;
        chatHeader.textContent = `Ector.V - ${selectedModel} 모델 채팅`; // Update header
        displayMessage(`${selectedModel} 모델이 선택되었습니다. 이제 메시지를 입력하세요!`, 'system');
        promptInput.placeholder = `메시지를 입력하세요.`;
        modelSuggestionsDiv.classList.remove('active'); // Hide suggestions
        promptInput.focus(); // Keep focus on the input field
    };

    // Initial greeting message when the page loads
    displayMessage('안녕하세요! Flexi, KeraLux, InteractGPT 중 하나의 모델을 선택하거나 메시지를 입력하여 채팅을 시작하세요.', 'system');

    // Event listener for input changes to show model suggestions
    promptInput.addEventListener('input', () => {
        const query = promptInput.value.trim().toLowerCase();
        modelSuggestionsDiv.innerHTML = ''; // Clear previous suggestions

        // Only show suggestions if no model is selected and there's user input
        if (query.length > 0 && !selectedModel) {
            const filteredModels = models.filter(model =>
                model.name.toLowerCase().includes(query)
            );

            if (filteredModels.length > 0) {
                filteredModels.forEach(model => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.classList.add('suggestion-item');
                    suggestionItem.textContent = model.name;
                    suggestionItem.addEventListener('click', () => {
                        selectModel(model.name, model.url);
                        promptInput.value = ''; // Clear input after selection
                    });
                    modelSuggestionsDiv.appendChild(suggestionItem);
                });
                modelSuggestionsDiv.classList.add('active'); // Show suggestion box
            } else {
                modelSuggestionsDiv.classList.remove('active'); // Hide if no matches
            }
        } else {
            modelSuggestionsDiv.classList.remove('active'); // Hide if input is empty or model is already selected
        }
    });

    /**
     * Handles sending messages to the selected model.
     */
    async function sendMessage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // If no model is selected, try to select one from the input
        if (!selectedModel) {
            const foundModel = models.find(model => model.name.toLowerCase() === prompt.toLowerCase());
            if (foundModel) {
                selectModel(foundModel.name, foundModel.url);
                promptInput.value = ''; // Clear input after selection
                return; // Don't send the model name as a message
            } else {
                displayMessage('먼저 모델을 선택해 주세요 (예: Flexi, KeraLux, InteractGPT)', 'system');
                return;
            }
        }

        // Display user message
        displayMessage(prompt, 'user');
        promptInput.value = ''; // Clear the input field
        sendBtn.disabled = true; // Disable send button during response

        // Display a loading message for the bot's response
        const botMessage = document.createElement('div');
        botMessage.className = 'chat-message bot';
        botMessage.innerHTML = `『 "${prompt}" 에 대한 응답을 생성 중... 』<br><br>`;
        chatContainer.appendChild(botMessage);
        chatContainer.scrollTop = chatContainer.scrollHeight;

        try {
            const params = new URLSearchParams({ prompt });
            const response = await fetch(`${apiUrl}?${params}`);

            if (!response.body) {
                throw new Error("응답 바디가 없습니다.");
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            let isFirstChunk = true;
            let receivedText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    botMessage.innerHTML = receivedText + "<br>[✔️ 완료]"; // Add completion message
                    break;
                }

                const text = decoder.decode(value, { stream: true });

                // Remove the initial loading message if this is the first chunk
                if (isFirstChunk) {
                    receivedText = text.trimStart(); // Start with the actual response
                } else {
                    receivedText += text;
                }
                
                botMessage.innerHTML = receivedText; // Update content incrementally
                isFirstChunk = false;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        } catch (err) {
            botMessage.innerHTML = `[❌ 오류 발생] ${err.message}`; // Display error message
            console.error('Error sending message:', err);
        } finally {
            sendBtn.disabled = false; // Re-enable send button
            promptInput.focus(); // Focus back on input
        }
    }

    // Event listener for the send button
    sendBtn.addEventListener('click', sendMessage);

    // Event listener for Enter key in the prompt input
    promptInput.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) { // Prevent new line with Shift + Enter
            e.preventDefault(); // Prevent default Enter behavior (e.g., new line)
            sendMessage();
        }
    });
});
