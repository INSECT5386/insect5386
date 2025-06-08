const modelSelection = document.getElementById('model-selection');  
const chatUI = document.getElementById('chat-ui');  
const chatHeader = document.getElementById('chat-header');  
const chatContainer = document.getElementById('chat-container');  
const promptInput = document.getElementById('prompt');  
const sendBtn = document.getElementById('send-btn');  
const backBtn = document.getElementById('back-btn');  

const modelDescriptionSection = document.getElementById('model-description');  
const modelNameElem = document.getElementById('model-name');  
const modelDescElem = document.getElementById('model-desc');  
const startChatBtn = document.getElementById('start-chat-btn');  
const cancelBtn = document.getElementById('cancel-btn');  

let selectedModel = null;  
let apiUrl = null;  

// 모델별 간단 설명  
const modelDescriptions = {  
  flexi: "Flexi는 InteractGPT의 개선모델로, 다양한 분석 기능과 유연한 응답을 제공합니다.",  
  keralux: "KeraLux는 180만 개 한국어 데이터로 사전학습된 GPT 기반 모델로, 한국어 최적화와 자연스러운 대화를 지원합니다.",
  interactgpt: "InteractGPT는 대화형 GPT 모델로, 간단한 일상대화를 지원합니다"
};  

// 1. 모델 선택 버튼 클릭 시 -> 설명 모달 띄우기  
modelSelection.querySelectorAll('button').forEach(btn => {  
  btn.addEventListener('click', () => {  
    selectedModel = btn.dataset.model;  
    apiUrl = btn.dataset.url;  

    modelNameElem.textContent = selectedModel.charAt(0).toUpperCase() + selectedModel.slice(1);  
    modelDescElem.textContent = modelDescriptions[selectedModel] || "설명 없음";  

    modelDescriptionSection.classList.add('active');  
  });  
});  

// 2. 설명 모달에서 '채팅 시작' 클릭  
startChatBtn.addEventListener('click', () => {  
  modelDescriptionSection.classList.remove('active');  

  chatHeader.textContent = `Ector.V - ${selectedModel} 모델 채팅`;  
  modelSelection.style.display = 'none';  
  chatUI.style.display = 'flex';  

  chatContainer.innerHTML = '';  
  promptInput.value = '';  
  promptInput.focus();  
});  

// 3. 설명 모달 취소 버튼  
cancelBtn.addEventListener('click', () => {  
  modelDescriptionSection.classList.remove('active');  
});  

// 4. 채팅방에서 '모델 선택으로 돌아가기' 버튼 클릭  
backBtn.addEventListener('click', () => {  
  selectedModel = null;  
  apiUrl = null;  

  chatUI.style.display = 'none';  
  modelSelection.style.display = 'block';  

  chatContainer.innerHTML = '';  
  promptInput.value = '';  
});  

// 5. 메시지 전송 함수  
async function sendMessage() {  
  const prompt = promptInput.value.trim();  
  if (!prompt) return;  

  // 유저 메시지 표시  
  const userMessage = document.createElement('div');  
  userMessage.className = 'message user';  
  userMessage.textContent = prompt;  
  chatContainer.appendChild(userMessage);  

  // 봇 메시지 자리 만들기  
  const botMessage = document.createElement('div');  
  botMessage.className = 'message bot';  
  botMessage.textContent = `『 "${prompt}" 에 대한 응답을 생성 중... 』\n\n`;  
  chatContainer.appendChild(botMessage);  

  chatContainer.scrollTop = chatContainer.scrollHeight;  
  promptInput.value = '';  
  sendBtn.disabled = true;  

  try {  
    const params = new URLSearchParams({ prompt });  
    const response = await fetch(`${apiUrl}?${params}`);  

    if (!response.body) throw new Error("응답 바디가 없습니다.");  

    const reader = response.body.getReader();  
    const decoder = new TextDecoder("utf-8");  

    let isFirstChunk = true;  
    while (true) {  
      const { done, value } = await reader.read();  
      if (done) {  
        botMessage.textContent += "\n[✔️ 완료]";  
        break;  
      }  

      const text = decoder.decode(value, { stream: true });  

      if (isFirstChunk && text.startsWith(prompt)) {  
        botMessage.textContent += text.slice(prompt.length).trimStart();  
      } else {  
        botMessage.textContent += text;  
      }  
      isFirstChunk = false;  
      chatContainer.scrollTop = chatContainer.scrollHeight;  
    }  
  } catch (err) {  
    botMessage.textContent = `[❌ 오류 발생] ${err.message}`;  
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

