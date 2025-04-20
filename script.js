import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

(async () => {
    const client = await Client.connect("Yuchan5386/Kossistant-1");

    const chatBox = document.querySelector(".chat-box");
    const input = document.querySelector("#userInput");
    const button = document.querySelector("#sendBtn");

    const appendMessage = (text, type) => {
        const msg = document.createElement("div");
        msg.className = `message ${type === "user" ? "user-msg" : "bot-msg"}`;
        msg.innerHTML = text.replace(/\n/g, "<br>");
        chatBox.appendChild(msg);
        chatBox.scrollTop = chatBox.scrollHeight;
    };

    const sendMessage = async () => {
        const userInput = input.value.trim();
        if (!userInput) return;

        appendMessage(userInput, "user");
        input.value = "";
        input.disabled = true;
        button.disabled = true;

        try {
            const result = await client.predict("/chat_respond", [userInput]);
            appendMessage(result.data[0], "bot");
        } catch (err) {
            appendMessage("오류가 발생했어요. 다시 시도해 주세요.", "bot");
        }

        input.disabled = false;
        button.disabled = false;
        input.focus();
    };

    button.addEventListener("click", sendMessage);
    input.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
})();
