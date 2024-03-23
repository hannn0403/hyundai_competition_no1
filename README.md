# hyundai_competition


## 1. 문제 정의 
---
이번 공모전의 첫번째 task는 시스템 효율 향상을 위해 자동으로 적재 유/무를 판단하기 위해 적재함을 촬영한 이미지 데이터를 기반으로 화물 적재 유/무 판단하는 분류기를 개발하는 과제였습니다. 이 task를 진행하면서 고려되는 문제점들은 

**a. Imbalance Data**

**b. 날씨등 다양한 case에서 발생되는 noise 처리가 있었습니다.**

## 2. 데이터 전처리 
---
**a. Crop**

**b. OpenCV의 다양한 기법들 (Averaging, Gaussian Blur, Bilateral filter, Histogram Equalization, CLAHE)**

실험 결과물들을 비교를 하였을 때 CLAHE 방법이 가장 적합하다고 판단이 되어서 데이터 로드 시 아래 코드로 CLAHE를 이미지에 적용


## 3. 데이터 생성
---
**a. Under sampling** : 이 방안은 Data 불균형을 해소시킬 수는 있지만 학습에 사용할 수 있는 전체 데이터 수를 감소시키고 분류에 중요한 데이터를 학습데이터에서 배제하여 성능저하를 야기할 수 있습니다. 또한 데이터의 수가 많지 않은(label0:7700, label1:4400) 상황에서 balance를 맞추기 위해서 label 1에 해당하는 데이터를 under sampling 하는 것은 오히려 학습의 성능이 크게 저하가 될 것이라고 판단이 되어 적용하지 않았습니다.

**b. Over sampling** : 불균형한 데이터 셋에서 낮은 비율을 차지하던 클래스의 데이터 수를 늘림으로써 데이터 불균형 문제를 해결할 수 있게 하지만 새로운 데이터를 생성하는 것이 아니기 때문에 과적합을 야기하고, 모델의 성능을 저하시킬 가능성이 있습니다. 대표적인 기법으로 SMOTE 기법이 있습니다. 이 기법은 KNN 알고리즘 기반으로 되어있으며 단순 무작위 추가 추출로 데이터 수를 늘리는 것이 아닌 특징을 가지는 새로운 데이터를 생성하는 방법입니다. 하지만 Image Data에는 사용을 잘 안 하기 때문에 SMOTE를 적용하지 않았습니다. 

**c. CGAN Over sampling** : 어느 정도의 육안으로 확인이 가능한 만큼 데이터가 생성이 되는 것을 알 수 있었지만 GAN 모델의 특성상 특징점이 많은 현재 학습 데이터에 대해서 좀 이미지 생성 모델을 더 정교하게 만들기 쉽지 않았고(학습시간 평균 10시간) 위와 같은 사진을 넣어서 학습을 진행하면 오히려 성능이 저하될 것을 생각하여 방법론을 철회하였습니다.

![Figure 1](/figure/CGAN.png)

**d. Cut-Mix Over sampling** : 모델이 객체의 차이를 식별할 수 있는 부분에 집중하지 않고, 덜 구별되는 부분 및 이미지의 전체적인 구역을 보고 학습하도록 하여 일반화와 Localization 성능을 높이는 방법

결론적으로 데이터 생성에서 앞서 정의한 Imbalance Data의 문제를 해결하기 위한 방법으로 저희가 실제 구현한 모델에 사용한 기법은 먼저 Cut-Mix 함수를 저희의 case에 맞춘 함수를 구현하여 label0: 1000, label1: 2000 개의 데이터를 새롭게 생성하였습니다. 이 과정을 통해 label 비율이 기존의 약 7:4에서 4:3으로 조정이 되었습니다. 
