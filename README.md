# 프리 온보딩 코스

[Wanted 프리온보딩 AI/ML 코스](https://www.wanted.co.kr/events/pre_onboarding_course_9)에서 수행한 프로젝트 / 과제들을 정리합니다.

</br>

[선발 과제](https://github.com/Woodywarhol9/wanted_pre_onboarding/tree/main/%EC%84%A0%EB%B0%9C%EA%B3%BC%EC%A0%9C) : `Tokenizer`, `TfidfVectorizer`  구현

</br>

[기업 과제1](https://github.com/Woodywarhol9/wanted_pre_onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C1) : `문자열 매칭` 알고리즘 구현

</br>

[기업 과제2]() : 유튜브 `데이터 분석` 및 `인기도 지표` 개발

</br>

[기업 과제3](https://github.com/Woodywarhol9/wanted_pre_onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C3) : `KLUE - STS` 성능 개선

</br>

[기업 과제4](https://github.com/Woodywarhol9/wanted_pre_onboarding/tree/main/%EA%B8%B0%EC%97%85%EA%B3%BC%EC%A0%9C4) : 스포츠 기사 `요약문 생성` 및 `평가 지표` 개발

---

##### Week 2 - 1
`huggingface`에서 pretrained `tokenizer`, `bert` 를 불러와 구조 확인

`layer` 마다 `embedding` 추출 및 `cosine-similarity` 측정   


</br>

##### Week 2 - 2
`Bert`에 `binary-classifier`를 연결해 [nsmc](https://github.com/e9t/nsmc) 데이터로 `fine-tuning`

`fine-tuning` - `free`, `unfreeze`

</br>

##### Week 2 - 3
`Custom Dataset`, `Custom collate_fn` 구현
훈련(train), 검증(valid) `Dataloader` 구성

</br>

##### Week 2 - 4
`helper.py` 모듈의 `Class`, `function`를 import
`accuracy()` 함수 구현하여 모델의 예측 정확도 확인

</br>

##### Week 3 - 1
`Skip-gram` 방식의 `word2vec` 구현하기
- `Skip-gram` 방식에 맞는 `CustomDataset` 구현
- `Negative Sampling` 구현
- `word2vec` 학습 및 `gensim`으로 결과 확인
- `Corpus` : [tokenized ko-wikipedia](https://ratsgo.github.io/embedding/downloaddata.html) 
- `stop-words` : https://www.ranks.nl/stopwords/korean

</br>

##### Week 3 - 2
`WordPiece Tokenizer` 학습 및 결과 확인

</br>

##### Week 3 - 4
`Transformers` 논문 클론 코딩
참고 : https://nlp.seas.harvard.edu/2018/04/03/attention.html

</br>

##### Week 4
`Tensorboard` 사용하기
- `Graph`, `Metrics`, `Text` 시각화