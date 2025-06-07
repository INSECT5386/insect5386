document.addEventListener('DOMContentLoaded', () => {
  const modelSelection = document.getElementById('model-selection');
  const chatUI = document.getElementById('chat-ui');
  const chatContainer = document.getElementById('chat-container');
  const promptInput = document.getElementById('prompt');
  const sendBtn = document.getElementById('send-btn');
  const backBtn = document.getElementById('back-btn');
  const modelDescModal = document.getElementById('model-description');
  const modalContent = modelDescModal.querySelector('.modal-content');

  let selectedModel = null;
  let chatHistory = [];

  // 모델 버튼 클릭 시
  modelSelection.addEventListener('click', e => {
    if (e.target.tagName === 'BUTTON') {
      selectedModel = e.target.dataset.model;
      showModelDescription(selectedModel);
    }
  });

  // 설명 모달 보여주기
  function showModelDescription(model) {
    const descriptions = {
      'flexi': 'Flexi 모델은 최신 GPT 구조 기반으로 대화와 분석에 최적화된 모델입니다.',
      'hydra': 'Hydra 모델은 병렬 블럭이 재귀적으로 확장되어 복잡한 문제 해결에 뛰어납니다.',
      'blaze': 'Blaze 모델은 소각층과 장기층을 사용해 기억 복원 능력이 탁월합니다.'
    };
    modalContent.querySelector('h3').textContent = model.charAt(0).toUpperCase() + model.slice(1) + ' 모델 설명';
    modalContent.querySelector('p').textContent = descriptions[model] || '설명 정보가 없습니다.';
    modelDescModal.classList.add('active');
  }

  // 모달 배경이나 닫기 버튼 클릭 시 모달 닫기
  modelDescModal.addEventListener('click', e => {
    if (e.target === modelDescModal || e.target.classList.contains('close-btn')) {
      modelDescModal.classList.remove('active');
    }
  });

  // 모달 내부 닫기 버튼 생성 및 동작 추가
  if (!modalContent.querySelector('.close-btn')) {
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '닫기';
    closeBtn.className = 'close-btn';
    modalContent.appendChild(closeBtn);
    closeBtn.addEventListener('click', () => {
      modelDescModal.classList.remove('active');
    });
  }

  // 채팅 입력과 전송 처리
  sendBtn.addEventListener('click', sendMessage);
  promptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // 뒤로가기 버튼 클릭 시
  backBtn.addEventListener('click', () => {
    // 선택 취소 후 UI 전환
    selectedModel = null;
    chatHistory = [];
    chatContainer.innerHTML = '';
    chatUI.style.display = 'none';
    modelSelection.style.display = 'block';
    promptInput.value = '';
    sendBtn.disabled = false;
  });

  // 메시지 보내기 함수
  async function sendMessage() {
    const text = promptInput.value.trim();
    if (!text || !selectedModel) return;

    // 사용자 메시지 출력
    appendMessage('user', text);
    promptInput.value = '';
    sendBtn.disabled = true;

    chatHistory.push({ role: 'user', content: text });

    // 서버에 모델과 메시지 보내기 (여기서는 예시, 실제 API 호출 필요)
    try {
      const responseText = await fakeApiRequest(selectedModel, chatHistory);
      appendMessage('bot', responseText);
      chatHistory.push({ role: 'bot', content: responseText });
    } catch (error) {
      appendMessage('bot', '오류가 발생했습니다. 다시 시도해주세요.');
    } finally {
      sendBtn.disabled = false;
      scrollChatToBottom();
    }
  }

  // 메시지 영역에 추가
  function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.className = 'message ' + sender;
    div.textContent = text;
    chatContainer.appendChild(div);
    scrollChatToBottom();
  }

  // 스크롤을 가장 아래로
  function scrollChatToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // 가짜 API 요청 함수 (실제로는 서버 API 호출로 바꿔야 함)
  function fakeApiRequest(model, history) {
    return new Promise(resolve => {
      setTimeout(() => {
        resolve(`(${model} 모델 응답) "${history[history.length - 1].content}" 에 대한 답변입니다.`);
      }, 1200);
    });
  }

  // 초기 UI 설정
  chatUI.style.display = 'none';
  modelSelection.style.display = 'block';

  // 모델 선택 시 채팅 UI로 전환
  modelSelection.addEventListener('click', e => {
    if (e.target.tagName === 'BUTTON') {
      chatUI.style.display = 'flex';
      modelSelection.style.display = 'none';
      promptInput.focus();
    }
  });
});
