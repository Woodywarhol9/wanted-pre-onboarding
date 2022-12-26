# Tokenizer, Tf-idf Vectorizer 구현하기

### **Tokenizer**
```python
class Tokenizer():
  def __init__(self):
    self.word_dict = {'oov': 0}
    self.fit_checker = False
  
  def preprocessing(self, sequences):
    """
    텍스트 전처리
    """
  def fit(self, sequences):
    """
    어휘 사전 구축
    """
  def transform(self, sequences):
    """
    토큰화 진행
    """
  def fit_transform(self, sequences):
    self.fit(sequences)
    result = self.transform(sequences)
    return result
```

**1-1. `preprocessing()`**

텍스트 전처리를 하는 함수입니다.

- input: 여러 영어 문장이 포함된 list 입니다. ex) ['I go to school.', 'I LIKE pizza!']
- output: 각 문장을 토큰화한 결과로, nested list 형태입니다. ex) [['i', 'go', 'to', 'school'], ['i', 'like', 'pizza']]
- 조건 1: 입력된 문장에 대해서 소문자로의 변환과 특수문자 제거를 수행합니다.
- 조건 2: 토큰화는 white space 단위로 수행합니다.
```python
  def preprocessing(self, sequences):
    result = []
    """
    문제 1-1
    """
    for sentence in sequences:
      # 소문자 변환
      temp = sentence.lower() 
      # 특수 문자 제거
      temp = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', temp) 
      # 공백 기준 분리
      temp = temp.split(" ") 
      result.append(temp)
    
    return result
```
**1-2. `fit()`**

어휘 사전을 구축하는 함수입니다.

- input: 여러 영어 문장이 포함된 list 입니다. ex) ['I go to school.', 'I LIKE pizza!']
- 조건 1: 위에서 만든 `preprocessing` 함수를 이용하여 각 문장에 대해 토큰화를 수행합니다.
- 조건 2: 각각의 토큰을 정수 인덱싱 하기 위한 어휘 사전(`self.word_dict`)을 생성합니다.
    - 주어진 코드에 있는 `self.word_dict`를 활용합니다.
```python
  def fit(self, sequences):
    self.fit_checker = False
    '''
    문제 1-2.
    '''
    # 전처리
    tokens = self.preprocessing(sequences) 
    # 인덱스 설정
    idx = len(self.word_dict)
    for tok in tokens:  
      for word in tok:
        # word_dict에 없을 경우 추가
        if word not in self.word_dict: 
          # 단순 출현 순서에 따른 인덱스 부여
          self.word_dict[word] = idx 
          idx += 1

    self.fit_checker = True
```

**1-3. `transform()`**

어휘 사전을 활용하여 입력 문장을 정수 인덱싱하는 함수입니다.

- input: 여러 영어 문장이 포함된 list입니다. ex) ['I go to school.', 'I LIKE pizza!']
- output: 각 문장의 정수 인덱싱으로, nested list 형태입니다. ex) [[1, 2, 3, 4], [1, 5, 6]]
- 조건 1: 어휘 사전(`self.word_dict`)에 없는 단어는 'oov'의 index로 변환합니다.
```python
  def transform(self, sequences):
    result = []
    tokens = self.preprocessing(sequences) # 전처리

    if self.fit_checker:
      '''
      문제 1-3.
      '''
      for tok in tokens:
        word_to_idx = [] 
        for word in tok:
          if word in self.word_dict: #word_dict에 있으면 word에 해당하는 idx, 없다면 oov 
            word_to_idx.append(self.word_dict[word]) 
          else:
            word_to_idx.append(self.word_dict["oov"])
        result.append(word_to_idx)

      return result
    else:
      raise Exception("Tokenizer instance is not fitted yet.")
```
---
### **TF-idf Vectorizer**
```python
class TfidfVectorizer:
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.fit_checker = False

  def fit(self, sequences):
    '''
    IDF 행렬 만들기
    '''
  def transform(self, sequences):
    if self.fit_checker:
      '''
      TF-IDF 행렬 만들기
      '''
  def fit_transform(self, sequences):
    self.fit(sequences)
    return self.transform(sequences)
```
**2-1. `fit()`**

입력 문장들을 이용해 IDF 행렬을 만드는 함수입니다.

- input: 여러 영어 문장이 포함된 list 입니다. ex) ['I go to school.', 'I LIKE pizza!']
- 조건 1: IDF 행렬은 list 형태입니다.
    - ex) [토큰1에 대한 IDF 값, 토큰2에 대한 IDF 값, .... ]
- 조건 2: IDF 값은 아래 식을 이용해 구합니다.
    
    $$
    idf(d,t)=log_e(\frac{n}{1+df(d,t)})
    $$
    
    - $df(d,t)$ : 단어 t가 포함된 문장 d의 개수
    - $n$ : 입력된 전체 문장 개수
- 조건 3: 입력된 문장의 토큰화에는 문제 1에서 만든 Tokenizer를 사용합니다.
```python
  def fit(self, sequences):
    tokenized = self.tokenizer.fit_transform(sequences)
    '''
    문제 2-1.
    '''
    n = len(tokenized)
    self.idf_matrix = []
    for word in self.tokenizer.word_dict.values():
      # oov 토큰 넘어 가기
      if word == 0: continue
      # 단어별 등장 횟수 확인
      df = 0
      for tok in tokenized:
        if word in tok:
          df += 1
      self.idf_matrix.append(log(n/(df + 1)))
    self.fit_checker = True
```


**2-2. `transform()`**

입력 문장들을 이용해 TF-IDF 행렬을 만드는 함수입니다.

- input: 여러 영어 문장이 포함된 list입니다. ex) ['I go to school.', 'I LIKE pizza!']
- output : nested list 형태입니다.
    
    ex) [[tf-idf(1, 1), tf-idf(1, 2), tf-idf(1, 3)], [tf-idf(2, 1), tf-idf(2, 2), tf-idf(2, 3)]]
    
    |  | 토큰1 | 토큰2 | 토큰3 |
    | --- | --- | --- | --- |
    | 문장1 | tf-idf(1,1) | tf-idf(1,2) | tf-idf(1,3) |
    | 문장2 | tf-idf(2,1) | tf-idf(2,2) | tf-idf(2,3) |
- 조건1 : 입력 문장을 이용해 TF 행렬을 만드세요.
    - $tf(d, t)$ : 문장 d에 단어 t가 나타난 횟수
- 조건2 : 문제 2-1( `fit()`)에서 만든 IDF 행렬과 아래 식을 이용해 TF-IDF 행렬을 만드세요
    
    $$
    tf-idf(d,t) = tf(d,t) \times idf(d,t)
    $$
```python
  def transform(self, sequences):
    if self.fit_checker:
      tokenized = self.tokenizer.transform(sequences)
      '''
      문제 2-2.
      '''
      tf_matrix = []
      self.tfidf_matrix = []
      # 문장별로 확인
      for tok in tokenized:
        tf_temp = []
        # oov 토큰 무시하기 위해서 1부터 시작
        for word in range(1 ,len(self.tokenizer.word_dict)):
          tf = tok.count(word)
          tf_temp.append(tf)
        tf_matrix.append(tf_temp)
      # tf - idf 계산
      tf_matrix_np = np.array(tf_matrix)
      idf_matrix_np = np.array(self.idf_matrix)
      self.tfidf_matrix = (tf_matrix_np * idf_matrix_np).tolist()
      return self.tfidf_matrix
    else:
      raise Exception("TfidfVectorizer instance is not fitted yet.")
```