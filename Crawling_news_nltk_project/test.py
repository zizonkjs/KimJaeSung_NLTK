import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

# 1. BertLSTMClassifier 모델 정의 (학습에 사용된 구조와 동일해야 함)
class BertLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, output_dim, lstm_layers, bidirectional, dropout):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = bert_output.last_hidden_state  # BERT의 마지막 은닉 상태
        lstm_output, _ = self.lstm(sequence_output)  # LSTM에 입력
        lstm_output = self.dropout(lstm_output[:, -1, :])  # 마지막 타임스텝 출력 사용
        logits = self.fc(lstm_output)
        return logits

# 2. 학습된 모델 불러오기
def load_model(model_path, device):
    model = BertLSTMClassifier(
        bert_model_name='monologg/koelectra-base-v3-discriminator',
        hidden_dim=256,  # Hidden dimension size (수정 가능)
        output_dim=4,    # Output categories (4개로 설정)
        lstm_layers=2,   # LSTM layers 수
        bidirectional=True,  # Bidirectional 설정
        dropout=0.3      # Dropout rate
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    return model

# 3. 카테고리 매핑 (4개 카테고리로 설정)
category_mapping = {
    0: '글로벌 경제 관련 뉴스기사입니다.',
    1: '금융 관련 뉴스기사입니다.',
    2: '부동산 관련 뉴스기사입니다.',
    3: '증권 관련 뉴스기사입니다.'
}

# 4. 뉴스 기사 입력 받아서 예측하기
def predict_category(model, tokenizer, text, max_len, category_mapping, device):
    # 입력 텍스트를 BERT 입력 형태로 전처리
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False
    )

    input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)

    # 모델로부터 예측값 얻기
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()

    # 예측된 레이블에 해당하는 카테고리 출력
    category_message = category_mapping.get(prediction, "카테고리를 알 수 없습니다.")
    return category_message

# 5. 실행 부분
if __name__ == "__main__":
    # 디바이스 설정 (GPU가 있으면 사용, 없으면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 사전 학습된 모델 가중치 파일 경로 (사용자가 제공한 파일 경로로 대체 필요)
    model_path = r"C:\Users\kimjaesung\9.자연어처리\Crawling_news_nltk_project\bert_lstm_model.pth"

    # 모델과 토크나이저 로드
    loaded_model = load_model(model_path, device)
    tokenizer = BertTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

    # 테스트할 뉴스 기사 입력 (예시)
    new_article = """강남 재건축 단지, 분양가 상한제 완화 기대감에 매수세 증가

서울 강남구 주요 재건축 단지에서 분양가 상한제 완화 기대감으로 매수세가 증가하고 있다. 최근 정부가 재건축 규제 완화 가능성을 시사한 가운데, 강남권 재건축 단지의 가격이 상승세를 보이고 있다. 전문가들은 "재건축 단지의 경우 추가 상승 여력이 크다"며 투자자들의 관심이 집중되고 있다고 분석했다."""

    # 예측 결과 출력
    predicted_message = predict_category(loaded_model, tokenizer, new_article, max_len=128, category_mapping=category_mapping, device=device)
    print(predicted_message)
