# Dataset

## 1. classification

[`KLUE-TC`](https://klue-benchmark.com/tasks/66/overview/description) 데이터셋 활용


* 학습 데이터에서 임의로 Train / Valid / Test 데이터를 생성함
* 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함
  
* Data 구조  
  * `data path` : `./dataset/classification`
  * `# train` : 36,571 문장
  * `# valid` : 9,107 문장
  * `# test` : 9,107 문장
  * `labels` : 정치, 경제, 사회, 문화, 세계, IT/과학, 스포츠 (7개)
  * `문장, 레이블 구분자` : \t

| text  | labels |
|-------|--------|
| 문장  |  라벨  |




## 2. named entity recognition (NER)


[`KLUE-NER`](https://klue-benchmark.com/tasks/69/overview/description) 데이터셋 활용


* 학습 데이터에서 임의로 Train / Valid / Test 데이터를 생성함
* 데이터 탐색에 용이하게 tsv 형태로 데이터를 변환함

* Data 구조  
  * `data path` : `./dataset/ner`
  * `# train` : 16,007 문장
  * `# valid` : 4,999 문장
  * `# test` : 4,999 문장
  * `entity types` : person(PS), location(LC), organization(OG), date(DT), time(TI), and quantity(QT)
  * `ne tags` : B-PS, I-PS, B-LC, I-LC, B-OG, I-OG, B-DT, I-DT, B-TI, I-TI, B-QT, I-QT, O
  * `토큰, 레이블 구분자` : \t
  * `토큰 구분자` : \n
  * `문장 구분자` : \n\n

| token     | tags   |
|---------- |--------|
| char 토큰 | NE_TAG |  


