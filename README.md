# hyundai_competition


## 1. 문제 정의 (Problem Definition)

이번 공모전의 첫번째 task는 시스템 효율 향상을 위해 자동으로 적재 유/무를 판단하기 위해 적재함을 촬영한 이미지 데이터를 기반으로 화물 적재 유/무 판단하는 분류기를 개발하는 과제였습니다. 이 task를 진행하면서 고려되는 문제점들은 

**a. Imbalance Data**

**b. 날씨등 다양한 case에서 발생되는 noise 처리가 있었습니다.**

## 2. 데이터 전처리 (Data Preprocessing)

**a. Crop**

**b. OpenCV의 다양한 기법들 (Averaging, Gaussian Blur, Bilateral filter, Histogram Equalization, CLAHE)**

실험 결과물들을 비교를 하였을 때 CLAHE 방법이 가장 적합하다고 판단이 되어서 데이터 로드 시 아래 코드로 CLAHE를 이미지에 적용


## 3. 데이터 생성 (Data Sampling)

**a. Under sampling** : 이 방안은 Data 불균형을 해소시킬 수는 있지만 학습에 사용할 수 있는 전체 데이터 수를 감소시키고 분류에 중요한 데이터를 학습데이터에서 배제하여 성능저하를 야기할 수 있습니다. 또한 데이터의 수가 많지 않은(label0:7700, label1:4400) 상황에서 balance를 맞추기 위해서 label 1에 해당하는 데이터를 under sampling 하는 것은 오히려 학습의 성능이 크게 저하가 될 것이라고 판단이 되어 적용하지 않았습니다.

**b. Over sampling** : 불균형한 데이터 셋에서 낮은 비율을 차지하던 클래스의 데이터 수를 늘림으로써 데이터 불균형 문제를 해결할 수 있게 하지만 새로운 데이터를 생성하는 것이 아니기 때문에 과적합을 야기하고, 모델의 성능을 저하시킬 가능성이 있습니다. 대표적인 기법으로 SMOTE 기법이 있습니다. 이 기법은 KNN 알고리즘 기반으로 되어있으며 단순 무작위 추가 추출로 데이터 수를 늘리는 것이 아닌 특징을 가지는 새로운 데이터를 생성하는 방법입니다. 하지만 Image Data에는 사용을 잘 안 하기 때문에 SMOTE를 적용하지 않았습니다. 

**c. CGAN Over sampling** : 어느 정도의 육안으로 확인이 가능한 만큼 데이터가 생성이 되는 것을 알 수 있었지만 GAN 모델의 특성상 특징점이 많은 현재 학습 데이터에 대해서 좀 이미지 생성 모델을 더 정교하게 만들기 쉽지 않았고(학습시간 평균 10시간) 위와 같은 사진을 넣어서 학습을 진행하면 오히려 성능이 저하될 것을 생각하여 방법론을 철회하였습니다.

![Figure 1](/figure/CGAN.png)

**d. Cut-Mix Over sampling** : 모델이 객체의 차이를 식별할 수 있는 부분에 집중하지 않고, 덜 구별되는 부분 및 이미지의 전체적인 구역을 보고 학습하도록 하여 일반화와 Localization 성능을 높이는 방법

![Figure 2](/figure/custom_cutmix.png)

결론적으로 데이터 생성에서 앞서 정의한 Imbalance Data의 문제를 해결하기 위한 방법으로 저희가 실제 구현한 모델에 사용한 기법은 먼저 Cut-Mix 함수를 저희의 case에 맞춘 함수를 구현하여 label0: 1000, label1: 2000 개의 데이터를 새롭게 생성하였습니다. 이 과정을 통해 label 비율이 기존의 약 7:4에서 4:3으로 조정이 되었습니다. 


## 모델

### a. 모델 아키텍처
- 세 가지 사전 학습된 모델을 timm.github에서 불러왔습니다.
    1. vit_base_patch8_224
    2. swin_large_patch4_window7_224
    3. beit_large_patch16_224
- 이 모델들은 ImageNet-"Real Labels" 결과를 기준으로 선택되었습니다. 기준은 다음과 같습니다:
    1. 높은 정확도
    2. 작은 파라미터 수
    3. 입력 이미지 사이즈가 우리의 이미지 사이즈와 일치하는지 확인

### b. 손실 함수 (Loss Function)
- Cross Entropy를 사용했습니다. 이는 예측값과 실제 값이 같을 때 최소값을 갖으며, 예측값과 실제 값의 차이가 클수록 증가합니다. 이를 통해 오류 역전파를 진행하여 학습합니다.

### c. 옵티마이저 알고리즘
- Adam Optimizer를 선택했습니다. Adam은 Adagrad와 RMSProp의 장점을 결합한 것으로, Gradient의 re-scaling에 영향을 받지 않는 장점을 가지고 있습니다. 또한 Hyper-parameter가 직관적이며 튜닝이 거의 필요하지 않습니다.

### d. 학습률 스케줄러 (Learning Rate Scheduler)
- Cosine Annealing Warm Restarts에 추가적으로 warm up start와 max 값의 감소 기능이 추가된 형태의 스케줄러를 사용했습니다. 이는 미리 정해진 학습 일정에 따라 Learning Rate를 조정하여 학습을 진행하는 방법입니다.

### e. 레이블 스무딩 (Label Smoothing)
- 레이블 스무딩은 데이터 정규화 기법 중 하나로, Label을 부드럽게 만들어 모델의 일반화 성능을 향상시키는 방법입니다.


## 데이터 증강 (Data Augmentation)

데이터 증강은 주어진 데이터셋을 다양한 방법으로 보강하여 학습 데이터의 다양성을 증가시키는 기법입니다. 이를 통해 모델의 Robustness를 향상시킬 수 있습니다. 아래는 저희가 사용한 데이터 증강 기법입니다.

a. Resize
- <transforms.Resize(256, 256)>
- 입력 이미지의 크기를 조정하는 기능입니다. 저희 모델은 입력 이미지의 크기가 (1080, 1920)이었기 때문에 최종적으로 crop(224, 224)을 수행하기에 너무 크다고 판단되어 이미지의 크기를 축소했습니다.

b. Random Horizontal Flip
- <transforms.RandomHorizontalFlip(p=0.5)>
- 이미지를 50% 확률로 수평으로 뒤집는 기능입니다.

c. Random Rotation
- <transforms.RandomRotation(degrees=(-10, 10))>
- 이미지를 입력된 범위 내에서 랜덤으로 회전시키는 기능입니다.

d. Random Posterize
- <transforms.RandomPosterize(bits=5, p=0.2)>
- 이미지를 20% 확률로 bits를 5만큼 감소시키는 기능입니다.

e. Normalization
- <transforms.Normalize(mean,std)>
- 이미지들을 정규화하여 범위를 조정하는 기능으로, 학습 과정에서의 안정성과 속도를 향상시키고 local optima 문제를 예방합니다.

f. Random Crop
- <transforms.RandomCrop(224,224)>
- 입력된 크기의 데이터 크기로 랜덤한 위치에서 이미지를 자르는 기능입니다.


## 앙상블 (Ensemble)

### a. TTA (Test Time Augmentation) Ten Crop
- 모델의 학습이 완료된 후 테스트 데이터를 사용하여 예측을 수행할 때, 일부 이미지가 잘못 분류되는 경우가 발생할 수 있습니다. 이를 해결하기 위해 TTA(Ten Crop)을 사용합니다.
- 이 방법은 테스트할 이미지를 여러 부분으로 나누어 각 부분을 개별적으로 입력으로 사용하여 예측을 수행하고, 그 결과를 평균내어 최종 예측값으로 사용하는 방법입니다.

### b. 3 Model Hard-Voting
- 세 개의 사전 학습된 모델을 불러와 데이터를 학습한 후, TTA를 거쳐 각 모델의 출력을 얻습니다.
- 이후 3 Model Hard-Voting 방식을 사용하여 최종 분류를 수행합니다. 이 방식은 Majority Voting이라고도 하며, 각 모델이 예측한 결과 중 가장 많은 표를 얻은 클래스를 선택하는 방법입니다.


## 결론 (Conclusion)

![Figure 3](/figure/result_1.png)
![Figure 4](/figure/result_2.png)
<center>Best 3 Model들의 Train Accuracy & Validation Accuracy</center>

Pretrained model들에 100 epoch만큼 학습을 시키면서 저장된 각 모델들의 train loss, validation loss의 변화를 나타낸 그래프와 최종 앙상블 모델에 넣을 Best Model들의 train, validation Accuracy입니다. 
위 3개의 모델을 앙상블하여 위에서 설명드린 3 hard voting, TTA ten crop 기법들을 사용하여 다시 validation data를 예측하는 작업을 수행하였습니다. 결과는 아래와 같습니다. 

![Figure 4](/figure/experiment_result.png)


## 참고문헌 (Reference) 

[1] GAN을 활용한 오버샘플링M. J. Son, S. W. Jung, and E. J. Hwang, “A Deep Learning Based Over-Sampling Scheme for Imbalanced Data Classification,” KIPS Transactions on Software and Data Engineering, vol. 8, no. 7, pp. 311–316, Jul. 2019.

[2]https://github.com/rwightman/pytorch-image-models/tree/master/timm/models

[3] https://github.com/rwightman/pytorch-image-models/blob/master/results/




