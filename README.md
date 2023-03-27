# Pytorch를 활용한 NLP 모델링

[Wanted 프리온보딩 AI/ML 코스](https://www.wanted.co.kr/events/pre_onboarding_course_9)에서 수행한 프로젝트 / 과제들을 정리합니다.
- [선발 과제](https://github.com/Woodywarhol9/wanted_pre_onboarding/tree/main/%EC%84%A0%EB%B0%9C%EA%B3%BC%EC%A0%9C) : `Tokenizer`, `Tf-idf Vectorizer`  구현
<br>

- [기업 과제1](https://github.com/Woodywarhol9/wanted-pre-onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C1) : `문자열 매칭` 알고리즘 구현
<br>

- [기업 과제2](https://github.com/Woodywarhol9/wanted-pre-onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C2) : 유튜브 `데이터 분석` 및 `인기도 지표` 개발
<br>

- [기업 과제3](https://github.com/Woodywarhol9/wanted-pre-onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C3) : `KLUE - STS` 성능 개선
<br>

- [기업 과제4](https://github.com/Woodywarhol9/wanted-pre-onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C4) : 스포츠 기사 `요약문 생성` 및 `평가 지표` 개발
---
- [일일 과제](https://github.com/Woodywarhol9/wanted-pre-onboarding/tree/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C) : `Pytorch`를 활용한 `NLP` 모델링


##### [Week 2 - 1](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_1_assignment.ipynb)
- `huggingface`에서 pretrained `tokenizer`, `bert` 를 불러와 구조 확인
- `layer` 마다 `embedding` 추출 및 `cosine-similarity` 측정
<br>

##### [Week 2 - 2](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_2_assignment.ipynb)
- `Bert`에 `binary-classifier`를 연결해 [nsmc](https://github.com/e9t/nsmc) 데이터로 `fine-tuning`
- `fine-tuning` : `free`, `unfreeze` 방법
<br>

##### [Week 2 - 3](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_3_assignment.ipynb)
- `Custom Dataset`, `Custom collate_fn` 구현
- 훈련(train), 검증(valid) `Dataloader` 구성
<br>

##### [Week 2 - 4](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_4_assignment.ipynb)
- `helper.py` 모듈의 `Class`, `function`를 import
- `accuracy()` 함수 구현하여 모델의 예측 정확도 확인
<br>

##### [Week 3 - 1](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week3_1_assignment.ipynb)
- `Skip-gram` 방식의 `word2vec` 구현하기
    - `Corpus` : [tokenized ko-wikipedia](https://ratsgo.github.io/embedding/downloaddata.html) 
    - `stop-words` : https://www.ranks.nl/stopwords/korean
- `Negative Sampling` 구현하기
- `word2vec` 학습 및 `gensim`으로 결과 확인
<br>

##### [Week 3 - 2](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week3_2_assginment.ipynb)
- `WordPiece Tokenizer` 학습 및 결과 확인
<br>

##### [Week 3 - 4](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week3_4_assginment.ipynb)
- `Transformers` 논문 구현
- 참고 : https://nlp.seas.harvard.edu/2018/04/03/attention.html
<br>

##### [Week 4](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week4_tensorboard.ipynb)
- `Tensorboard` 사용하기
    - `Graph`, `Metrics`, `Text` 시각화