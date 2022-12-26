# Pytorch for NLP

[Wanted 프리온보딩 AI/ML 코스](https://www.wanted.co.kr/events/pre_onboarding_course_9)에서 수행한 일일 과제들을 정리합니다.

</br>

##### [Week 2 - 1](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_1_assignment.ipynb)
- `huggingface`에서 pretrained `tokenizer`, `bert` 를 불러와 구조 확인
- `layer` 마다 `embedding` 추출 및 `cosine-similarity` 측정

##### [Week 2 - 2](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_2_assignment.ipynb)
- `Bert`에 `binary-classifier`를 연결해 [nsmc](https://github.com/e9t/nsmc) 데이터로 `fine-tuning`
- `fine-tuning` : `free`, `unfreeze` 방법

##### [Week 2 - 3](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_3_assignment.ipynb)
- `Custom Dataset`, `Custom collate_fn` 구현
- 훈련(train), 검증(valid) `Dataloader` 구성

##### [Week 2 - 4](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week2_4_assignment.ipynb)
- `helper.py` 모듈의 `Class`, `function`를 import
- `accuracy()` 함수 구현하여 모델의 예측 정확도 확인

##### [Week 3 - 1](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week3_1_assignment.ipynb)
- `Skip-gram` 방식의 `word2vec` 구현하기
    - `Corpus` : [tokenized ko-wikipedia](https://ratsgo.github.io/embedding/downloaddata.html) 
    - `stop-words` : https://www.ranks.nl/stopwords/korean
- `Negative Sampling` 구현하기
- `word2vec` 학습 및 `gensim`으로 결과 확인

##### [Week 3 - 2](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week3_2_assginment.ipynb)
- `WordPiece Tokenizer` 학습 및 결과 확인

##### [Week 3 - 4](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week3_4_assginment.ipynb)
- `Transformers` 논문 구현
- 참고 : https://nlp.seas.harvard.edu/2018/04/03/attention.html

##### [Week 4](https://github.com/woodywarhol9/wanted-pre-onboarding/blob/main/%EC%9D%BC%EC%9D%BC%EA%B3%BC%EC%A0%9C/Week4_tensorboard.ipynb)
- `Tensorboard` 사용하기
    - `Graph`, `Metrics`, `Text` 시각화