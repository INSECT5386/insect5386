import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

(async () => {
    const client = await Client.connect("Yuchan5386/Kossistant-1");

    const sendMessage = async () => {
        const userInput = document.querySelector("#userInput").value;
        if (!userInput.trim()) return;

        // 유저 입력을 chat-container에 추가
        const userMessageDiv = document.createElement('div');
        userMessageDiv.classList.add('message', 'user');
        const userBubble = document.createElement('div');
        userBubble.classList.add('bubble', 'user-bubble');
        userBubble.textContent = userInput;
        userMessageDiv.appendChild(userBubble);
        document.querySelector("#chatBox").appendChild(userMessageDiv);

        // 챗봇의 응답을 가져와서 표시
        const result = await client.predict("/chat_respond", [userInput]);
        const botMessageDiv = document.createElement('div');
        botMessageDiv.classList.add('message', 'bot');
        const botBubble = document.createElement('div');
        botBubble.classList.add('bubble', 'bot-bubble');
        botBubble.textContent = result.data[0];
        botMessageDiv.appendChild(botBubble);
        document.querySelector("#chatBox").appendChild(botMessageDiv);

        // 스크롤을 최신 메시지로 자동 이동
        const chatBox = document.querySelector("#chatBox");
        chatBox.scrollTop = chatBox.scrollHeight;

        // 입력 필드 초기화
        document.querySelector("#userInput").value = '';
    };

    document.querySelector("#sendBtn").addEventListener("click", sendMessage);
    document.querySelector("#userInput").addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
})();
