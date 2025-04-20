import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

(async () => {
    const client = await Client.connect("Yuchan5386/Kossistant-1");

    const sendMessage = async () => {
        const inputElem = document.querySelector("#userInput");
        const userInput = inputElem.value.trim();
        if (!userInput) return;

        const chatBox = document.querySelector("#chatBox");

        // 사용자 말풍선
        const userBubble = document.createElement("div");
        userBubble.className = "chat-bubble user";
        userBubble.innerText = userInput;
        chatBox.appendChild(userBubble);

        inputElem.value = "";
        chatBox.scrollTop = chatBox.scrollHeight;

        // Gradio 호출
        const result = await client.predict("/chat_respond", [userInput]);
        const rawText = result.data[0];
        const botOnly = rawText.split("봇:")[1]?.trim() || rawText;

        // 봇 말풍선
        const botBubble = document.createElement("div");
        botBubble.className = "chat-bubble bot";
        botBubble.innerText = botOnly;
        chatBox.appendChild(botBubble);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    document.querySelector("#sendBtn").addEventListener("click", sendMessage);
    document.querySelector("#userInput").addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
})();
