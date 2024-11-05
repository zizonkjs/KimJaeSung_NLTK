#!/usr/bin/env python3
import sys
import io
import torch
import torch.nn as nn
import cgi
import cgitb
from transformers import ElectraTokenizer, BertModel
import torch.nn.functional as F

# 표준 출력의 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# CGI 에러 추적 활성화
cgitb.enable()

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
        
        sequence_output = bert_output.last_hidden_state
        lstm_output, _ = self.lstm(sequence_output)
        lstm_output = self.dropout(lstm_output[:, -1, :])
        logits = self.fc(lstm_output)
        return logits

# 2. 학습된 모델 불러오기
def load_model(model_path, device):
    model = BertLSTMClassifier(
        bert_model_name='monologg/koelectra-base-v3-discriminator',
        hidden_dim=256,  
        output_dim=4,    
        lstm_layers=2,   
        bidirectional=True,  
        dropout=0.3
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model

# 3. 카테고리 매핑
category_mapping = {
    0: '글로벌 경제 관련 뉴스기사입니다.',
    1: '금융 관련 뉴스기사입니다.',
    2: '부동산 관련 뉴스기사입니다.',
    3: '증권 관련 뉴스기사입니다.'
}

# 4. 뉴스 기사 입력 받아서 예측하기
def predict_category(model, tokenizer, text, max_len, category_mapping, device):
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

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(F.softmax(outputs, dim=1), dim=1).item()

    category_message = category_mapping.get(prediction, "카테고리를 알 수 없습니다.")
    return category_message

# CGI 시작 (입력 처리 및 결과 반환)
print("Content-Type: text/html; charset=UTF-8")  
print()

# 5. HTML 양식에서 기사 내용 받아오기
form = cgi.FieldStorage()
article_text = form.getvalue('article_text', '')

# 6. 모델과 토크나이저 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "C:/Users/kimjaesung/9.자연어처리/Crawling_news_nltk_project/bert_lstm_model.pth"
loaded_model = load_model(model_path, device)
tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')

# 7. 예측 실행
if article_text:
    predicted_message = predict_category(loaded_model, tokenizer, article_text, max_len=128, category_mapping=category_mapping, device=device)
else:
    predicted_message = "뉴스 기사 내용을 입력해주세요."

# 8. HTML 출력 (결과 보여주기)
print(f"""
<html>
<head><title>뉴스 기사 분류 결과</title></head>
<body>
    <h1>뉴스 기사 분류 결과</h1>
    <p>입력된 기사 내용: {article_text}</p>
    <h2>예측된 카테고리: {predicted_message}</h2>
    <form method="post" action="/cgi-bin/news_BERT_LSTM_model.py">
        <textarea name="article_text" rows="10" cols="50"></textarea><br>
        <input type="submit" value="예측하기">
    </form>
</body>
</html>
""")

