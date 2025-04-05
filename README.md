## 현재 제작 중인 모델
- **DoubleFusionNet**

## 제작 완료한 모델
- **SeQRoN**: Seq2Seq + Neuron

---

### 모델 정리

- **딥러닝 (Deep Learning, DL)**:
  - **자연어 처리 (NLP)**:
    1. **SeQRoN**: Seq2Seq + Neuron(합성어)
    2. **DoubleFusionNet**: 인코더 2개, 디코더 2개, 어텐션 메커니즘 탑재 된 챗봇

```
input_layer_6 (None, 24)
        │
  Embedding (vocab_q, 50)
        │
  ┌────────────────────────┐
  │ encoder_lstm_1         │
  │ (None, 24, 50), (None, 50), (None, 50) │
  └────────────────────────┘
         │ encoder_output_1
         ▼
    encoder_combined (24, 100)

────────────────────────────────────────────────────────────────────

input_layer_7 (None, 36)
         │
  Embedding (vocab_a, 50)
         │
  ┌────────────────────────┐       ┌────────────────────────┐
  │ decoder_lstm_1         │       │ decoder_lstm_2         │
  │ (None, 36, 50), (None, 50), (None, 50) │   (None, 36, 50), (None, 50), (None, 50) │
  └────────────────────────┘       └────────────────────────┘
         │ decoder_output_1                │ decoder_output_2
         └────────┬────────┘               └────────┬────────┘
                  ▼                           ▼
        Concatenate([decoder_output_1, decoder_output_2]) 
                  ▼
         decoder_combined (36, 100)

────────────────────────────────────────────────────────────────────

     decoder_combined ─┐
                        │
                        ▼
       AdditiveAttention([decoder_combined, encoder_combined]) 
                        ▼
               context_vector (36, 100)
                        ▼
              Dropout(0.2) → Dense(50) → Adjusted context_vector
                        ▼
  Concatenate([decoder_combined, context_vector]) 
                        ▼
         decoder_context_concat (36, 150) 
                        ▼
   TimeDistributed(Dense(vocab_size)) → Final Prediction (36, vocab_size)
