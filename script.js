import { Client } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

(async () => {
    const client = await Client.connect("Yuchan5386/Kossistant-1");

    const sendMessage = async () => {
        const userInput = document.querySelector("#userInput").value;
        if (!userInput.trim()) return;

        const result = await client.predict("/chat_respond", [userInput]);
        document.querySelector("#response").innerHTML = result.data[0].replace(/\n/g, "<br>");
    };

    document.querySelector("#sendBtn").addEventListener("click", sendMessage);
    document.querySelector("#userInput").addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendMessage();
    });
})();
