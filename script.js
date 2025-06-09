document.addEventListener('DOMContentLoaded', () => {
    const chatHeader = document.getElementById('chat-header');
    const chatContainer = document.getElementById('chat-container');
    const promptInput = document.getElementById('prompt');
    const sendBtn = document.getElementById('send-btn');
    const modelSuggestionsDiv = document.getElementById('model-suggestions');

    let selectedModel = null;
    let selectedModelUrl = null;

    // Define available models
    const models = [
        { name: 'Flexi', url: 'https://yuchan5386-flexi-api.hf.space/generate', desc: '유연하고 다재다능한 응답을 생성합니다.' },
        { name: 'KeraLux', url: 'https://yuchan5386-keralux-api.hf.space/generate', desc: '정확하고 심층적인 지식을 제공합니다.' },
        { name: 'InteractGPT', url: 'https://yuchan5386-interactgpt-api.hf.space/generate', desc: '대화형 상호작용에 특화되어 있습니다.' }
    ];

    // Function to display a message in the chat
    const displayMessage = (message, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('chat-message', sender);
        messageElement.innerHTML = `<p>${message}</p>`; // Use innerHTML for potential Markdown rendering
        chatContainer.appendChild(messageElement);
        chatContainer.scrollTop = chatContainer.scrollHeight; // Scroll to bottom
    };

    // Function to handle model selection
    const selectModel = (modelName, modelUrl) => {
        selectedModel = modelName;
        selectedModelUrl = modelUrl;
        chatHeader.textContent = `채팅 중: ${selectedModel}`;
        displayMessage(`${selectedModel} 모델이 선택되었습니다. 이제 메시지를 입력하세요!`, 'system');
        promptInput.placeholder = `메시지를 입력하세요.`;
        modelSuggestionsDiv.classList.remove('active'); // Hide suggestions
        promptInput.focus(); // Focus on input after selection
    };

    // Show initial greeting or prompt
    displayMessage('안녕하세요! Flexi, KeraLux, InteractGPT 중 하나의 모델을 선택하거나 메시지를 입력하여 채팅을 시작하세요.', 'system');

    // Handle input change for model suggestions
    promptInput.addEventListener('input', () => {
        const query = promptInput.value.toLowerCase();
        modelSuggestionsDiv.innerHTML = ''; // Clear previous suggestions

        if (query.length > 0 && !selectedModel) { // Only show suggestions if no model is selected
            const filteredModels = models.filter(model =>
                model.name.toLowerCase().includes(query)
            );

            if (filteredModels.length > 0) {
                filteredModels.forEach(model => {
                    const suggestionItem = document.createElement('div');
                    suggestionItem.classList.add('suggestion-item');
                    suggestionItem.textContent = model.name;
                    suggestionItem.dataset.modelName = model.name;
                    suggestionItem.dataset.modelUrl = model.url;
                    suggestionItem.dataset.modelDesc = model.desc; // Store description if needed for future modals
                    suggestionItem.addEventListener('click', () => {
                        selectModel(model.name, model.url);
                        promptInput.value = ''; // Clear input after selection
                    });
                    modelSuggestionsDiv.appendChild(suggestionItem);
                });
                modelSuggestionsDiv.classList.add('active'); // Show suggestions
            } else {
                modelSuggestionsDiv.classList.remove('active'); // Hide if no matches
            }
        } else {
            modelSuggestionsDiv.classList.remove('active'); // Hide if input is empty or model selected
        }
    });

    // Handle sending messages
    sendBtn.addEventListener('click', async () => {
        const userPrompt = promptInput.value.trim();

        if (!userPrompt) return;

        // If no model is selected, try to select one from the input
        if (!selectedModel) {
            const foundModel = models.find(model => model.name.toLowerCase() === userPrompt.toLowerCase());
            if (foundModel) {
                selectModel(foundModel.name, foundModel.url);
                promptInput.value = ''; // Clear input after selection
                return; // Don't send the model name as a message
            } else {
                displayMessage('먼저 모델을 선택해 주세요 (예: Flexi, KeraLux, InteractGPT)', 'system');
                return;
            }
        }

        displayMessage(userPrompt, 'user');
        promptInput.value = ''; // Clear the input field

        try {
            displayMessage('응답 생성 중...', 'model'); // Show a loading message
            const response = await fetch(selectedModelUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ inputs: userPrompt }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const modelResponse = data.generated_text || '응답을 생성할 수 없습니다.';
            // Remove the loading message before displaying the actual response
            chatContainer.lastChild.remove(); 
            displayMessage(modelResponse, 'model');

        } catch (error) {
            console.error('Error sending message:', error);
            // Remove the loading message and display an error message
            chatContainer.lastChild.remove(); 
            displayMessage('메시지를 보내는 중 오류가 발생했습니다. 다시 시도해 주세요.', 'system');
        }
    });

    // Allow sending messages with Enter key
    promptInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });
});
