# Semantic-Segmentation

## Requirements

- PyTorch 1.3 to 1.6

## Dataset

### <MF Dataset>
- 도심지에서 낮과 밤 시간대에 촬영된 데이터 셋으로 쌍을 이루는 RGB 영상과 Thermal 영상을 제공함.
- 픽셀 단위의 시맨틱 라벨 정보를 제공하고 있어 시맨틱 정보 추정 연구에 활용 됨.

- [데이터 셋 다운 홈페이지](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) 이곳에서 [Multi-spectral Semantic Segmentation Dataset (link to Google Drive)](https://drive.google.com/drive/folders/1YtEMiUC8sC0iL9rONNv96n5jWuIsWrVY) 이 링크를 통해서 Dataset을  현재 경로 다운 받아야한다.

- MF Dataset은 RGB 영상과 Thermal 영상을 합쳐서 4채널로 제공은 한다.
- 따라서 두 도메인의 영상을 따로 다루기 편하도록 RGB 영상과 Thermal 영상 따로 저장하는 작업이 필요하다.
- ```Make_split.ipynb```을 이용해 RGB 와 Thermal 를 분리해 저장해야한다. 

## Dataloader


- 데이터 폴더 구조 :
```
data
├── ir_seg_dataset
│   ├── images
│   │   ├── 00001D.png
│   │   ├── 00003N.png
│   │   ├── 00006N.png
│   │   └── ...
│   ├── labels
│   │   ├── 00001D.png
│   │   ├── 00003N.png
│   │   ├── 00006N.png
│   │   └── ...
│   └── ...
├── models
├── output

```

## Train && Test 

### 학습 및 평가 방식 
- 학습
   ```
    bash scripts/train.sh
   ``` 
- 평가
   ```
    bash scripts/test.sh
   ``` 

## 정량적 평가
   
  - 추정된 시맨틱 정보의 성능 평가를 위해 성능평가지표(evalutation metric)로 mIOU(mean Intersection Over Union)를 사용 하였음.
  
  - 서로 다른 두 도메인 사이의 차이를 완화하기 위해 칼라 영상에서 열화상 영상으로의 단 방향 전이가 아닌, 상호 학습을 기반으로 양 방향 전이를 수행하는 새로운 ㉠(상호 학습 기법)을 제안하여 기존 베이스라인(MS-UDA)에 적용함.
  - 또한 기존의 합성곱 신경망(CNN)이 중요한 영역에 대한 가중치 부여가 불가능하다는 문제를 해결하기 위해 어텐션 메커니즘을 적용함.
  - 추가적으로 풍부한 정보(e.g. 뚜렷한 가장자리, 질감)를 기반으로 시맨틱 정보 추출에 효과적인 영역에 가중치를 두고 있는 칼라 도메인의 어텐션 맵에서 열화상 도메인의 어텐션 맵으로의 전이를 수행하는 ㉡(어텐션 맵 손실)을 새롭게 설계함.
   
  - 제안하는 ㉠(상호 학습 기법), ㉡(어텐션 맵 손실)을 통해 UTokyo Multispectral Dataset(MF Dataset)에서 열화상 영상을 이용한 시맨틱 정보 추출 성능이 기존 베이스라인 대비 약 10 mIOU 이상 향상한 것을 확인할 수 있음.
  
| model |  입력영상| RMSE <50m |
|:-----: | :-----:|:-----: |
| MS-UDA |   열화상  |  67.8 |
| MS-UDA+㉠ |   열화상 |  70.4 |
| MS-UDA+㉡ |  열화상 |  66.6 |
| MS-UDA+㉠+㉡ |  열화상 |  **78.2** |
   
   
## 정성적 평가



(a) 칼라 영상, (b) 열화상 영상, (c) 정답 시맨틱 정보, (d) 열화상 영상을 이용한 2차년도 베이스라인 방법론(MS-UDA) 결과 (e) 열화상 영상을 이용한 시맨틱 정보 추출 연구(MS-UDA+㉠+㉡) 결과
