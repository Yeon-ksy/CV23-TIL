# 9/30~10/11 TIL ✏️

## 세연 [[Github](https://github.com/Yeon-ksy)] [[Velog](https://velog.io/@yeon-ksy/)]

# 1. Object Detection Overview
## History

<img src="https://github.com/user-attachments/assets/ad0f92e2-19ba-4825-a7f0-e143be25a73d" width="500"/>

## Evaluation
### mAP
<img src="https://github.com/user-attachments/assets/bf522e6f-161c-4da0-a57f-5f216a041caf" width="500"/>

- 위 같은 표가 있을 때, 이를 confidence 기준으로 내림차순하고, 누적 TP, FP을 계산하고 이 때, 누적된만큼의 Recall과 Precision을 계산
- 이렇게 나온 Recall과 Precision을 통해 그래프를 그리면 그것이 PR Curve이고 오른쪽 기준 최대 Precision 값들을 기준으로 잇고 그것의 면적을 구하면 AP임. 
- 이것을 각 클래스에 대해 평균을 내면 mAP

### FPS (Frames Per Second)
### FLOPs (Floating Point Operations)
- Model이 얼마나 빠르게 동작하는지 측정하는 metric
- 연산량 횟수이므로 FLOPs가 작으면 작을수록 빠름.

# 2 Stage Detectors
- 2 Stage = 객체가 있을법한 위치를 특정짓고, 해당 객체가 무엇인지 예측하는 2 단계로 나눠짐.

## R-CNN
<img src="https://github.com/user-attachments/assets/62a27b84-ffb0-4c2d-9d66-85806ecc3cf4" width="500"/>

- 입력 이미지가 주어졌을 때, Selective Search를 통해 약 2000개의 ROI를 추출
    - Selective Search : 주어진 이미지의 색깔, 질감, shape 등의 특성을 활용하여 이미지를 작은 영역으로 나눈 다음, 이 영역을 통합해나가는 형식으로 후보 영역을 search함.
    	<img src="https://github.com/user-attachments/assets/acefefa4-37cb-4a09-b9c4-97f12dffb3b2" width="300"/>
    
- ROI의 사이즈가 다 다르기 때문에 크기를 조절해 모두 동일한 사이즈로 변경
	- CNN의 FC 레이어의 입력 사이즈가 고정이므로 
- ROI를 CNN에 넣어 feature를 추출

## SPPNet
<img src="https://github.com/user-attachments/assets/fc424a58-d8f2-438f-9a01-b5c3beb39532" width="500"/>

- 이미지를 먼저 CNN에 통과시켜 feature를 얻어 2000개의 region을 뽑고, spatial pyramid pooling을 통해 고정된 사이즈로 변환. 이후 FC 레이어 통과시킴.
    - 다양한 크기의 ROI로부터 얻을 target의 Feature 사이즈를 결정함.
    	<img src="https://github.com/user-attachments/assets/e853493a-0d39-40c2-bfed-0e8be80676fc" width="300"/>
	    
	- ROI에서 input size에 맞게 Bin을 뽑아내고, 
	    - 각 Bin에서 Max Pooling 등을 통해 하나의 특징을 뽑아냄.

## Fast R-CNN
<img src="https://github.com/user-attachments/assets/f111335f-ce5c-4f97-9be1-22fff69ff557" width="500"/>

- 이미지를 CNN(VGG 16 사용)에 넣고, 나온 고정된 Feature Map을 통해 ROI Projection을 통해서 다른 사이즈를 가지는 ROI를 뽑아냄.
    - ROI Projection : 원본 이미지로부터 Selective Search를 통해 2,000개의 ROI를 뽑아내고, 이를 feature map 상에 Projection함.
    - FC layer에 의해 Feature Map의 이미지 크기가 달라지면, 그에 맞게 ROI를 조정
- 이후, ROI Pooling을 통해 고정된 사이즈의 Feature vector를 출력하고, FC layer를 통과시킴
    - ROI Pooling : spatial pyramid pooling와 같은 알고리즘임. 대신 target size가 7x7만 사용함.
- 이것을 Softmax를 통과시켜 Classifier를 학습하고, BBox Regressor를 통해 바운딩 박스를 학습함.
	- 클래스의 개수는 C+1 개 (배경 포함)

## Faster R-CNN
- Selective Search를 제거하고 대신에 RPN (Region Proposal Network)을 도입하여Region Proposal 역시 학습 가능한 형태로 바꿈
    - RPN
    	<img src="https://github.com/user-attachments/assets/7cbbb8ea-4347-4681-b222-c5fe67ee2880" width="300"/>
    
	- feature map의 각 cell마다 다양한 스케일과 비율을 가진 k개의 Anchor box가 존재함.
	- RPN은 이러한 Anchor Box가 객체를 포함하고 있는지 예측하여 Anchor Box를 미세 조정함.
	- 이를 위해 각 픽셀 별로 두 개의 Head를 통과함.
		- cls layer = k개의 Anchor Box에 객체 유무를 판단
		- reg layer = 바운딩 박스의 중심을 얼마나 이동해야 하는지, 가로, 세로는 얼마나 옮겨야 하는지 판단
- cls prediction으로 각 Anchor Box에 대한 score가 나오는데, 이를 통해 Top N 개의 box를 선택
- 이렇게 뽑은 ROI들은 겹치는 영역이 무수히 많음. 따라서 NMS (Non-Maximum Suppression)을 사용
    - NMS
        - BBox와 이에 대한 Score가 있을 때, 점수를 기준으로 내림차순으로 정렬
        - 여기서 가장 높은 점수를 가진 상자를 선택하고 특정 Threshold를 가지는 IoU를 설정
        - 이 IoU Threshold를 넘는 값인 상자들을 제거 (많이 겹치는 상자를 제거)
        - 이를 반복하여 얻은 상자들이 최종 객체 감지의 결과임.
        - 이 NMS는 mAP를 낮추는 문제가 있음. 이를 해결하기 위해 Soft-NMS를 사용하기도 함.
- 위 과정을 제외하고는 Fast R-CNN과 같음.
-  loss

    <img src="https://github.com/user-attachments/assets/f332b147-cb23-4922-965f-acacd6c7b505" width="300"/>

    - $L_{cls}$ = Cross Entropy loss
    - $L_{reg}$ = MSE loss
        - $p_i^*$ = $i$번째 Anchor Box가 객체를 포함하는 지 여부 (포함은 1, 미포함은 0)
