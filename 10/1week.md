# 9/30~10/11 TIL ✏️

## 세연 [[Github](https://github.com/Yeon-ksy)] [[Velog](https://velog.io/@yeon-ksy/)]

# 1. Object Detection Overview
## History

<img src="https://github.com/user-attachments/assets/ad0f92e2-19ba-4825-a7f0-e143be25a73d" width="500"/>

## Evaluation
### mAP
<img src="https://github.com/user-attachments/assets/bf522e6f-161c-4da0-a57f-5f216a041caf" width="500"/>

- 위 같은 표가 있을 때, 이를 confidence 기준으로 내림차순 후 누적 TP, FP을 계산하여 누적된만큼의 Recall과 Precision을 계산
- 이렇게 나온 Recall과 Precision을 통해 그래프를 그리면 그것이 PR Curve이고 오른쪽 기준 최대 Precision 값들을 기준으로 잇고 그것의 면적을 구하면 AP임. 
- 이것을 각 클래스에 대해 평균을 내면 mAP

### FPS (Frames Per Second)
### FLOPs (Floating Point Operations)
- Model이 얼마나 빠르게 동작하는지 측정하는 metric
- 연산량 횟수이므로 FLOPs가 작으면 작을수록 빠름.

# 2. 2 Stage Detectors
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

# 3. Neck
- Neck이란?

    <img src="https://github.com/user-attachments/assets/cb45933b-544f-4f32-ae35-bf459c1146d8" width="500"/>

    - 2 stage model들의 파이프라인을 봐보자. Backbone을 거쳐 나온 마지막 Feature Map을 통해 RPN을 함.
	- 여기서 Neck은 Backbone의 중간 중간의 Feature Map을 추가로 사용.
- 여러 크기의 Feature Map을 사용하게 된다면, ROI가 보는 Feature Map이 풍부해짐. 즉, 다양한 크기의 객체를 더 잘 탐지하기 위해 Neck을 사용

## FPN (Feature Pyramid Network)
- high level에서 low level로 semantic 정보 전달 필요
- 따라서 top-down path way 추가
	- Pyramid 구조를 통해서 high level 정보를 low level에 순차적으로 전달
		- Low level = Early stage = Bottom
		- High level = Late stage = Top

    <img src="https://github.com/user-attachments/assets/549500c7-4c16-4b48-8e23-cddad73c6191" width="300"/>

    - backbone 과정을 Bottom up, FPN 과정을 Top down이라고 함.
    - 각 Top down의 각 stage는 Resnet의 Pooling을 통해 feature map의 w, h가 절반으로 줄어들 때임.
- 전체 파이프라인

    <img src="https://github.com/user-attachments/assets/2077ed70-c3e9-42d0-84ae-e9c332c04c01" width="500"/>

    - P2, P3, P4 ... 가 나오면 이를 RPN (Region Proposal Network)에 입력하여 class score와 bbox regressor을 출력함. 이를 통해 물체가 있을만한 bbox를 조정하고
	 - 이후, bbox regressor을 통해 물체가 있을만한 bbox를 조정하고, class score를 통해 NMS (Non-Maximum Suppression)를 적용하여 최적의 원본 이미지에 대한 ROI를 추출하고 Score가 높은 1000개의 ROI를 Select함.
	 - 이 1000개의 ROI를 Feature Map에 ROI Projection을 해줘야 함.
		 - 근데 어떤 ROI가 어떤 stage (P5, P4, P3 등)에서 나왔는지 알 수 없고, 어떤 feature map에 projection을 해야하는 지 알기 어려움. 
		 - 따라서 k를 통해 맵핑함.
			 $$k = [k_0 + log_2(\sqrt{wh}/224)]$$
			 - $k_0$는 기본으로 4이고 (4번째 Stage가 기본이 됨.), $w$와 $h$는 stage들에서 나온 ROI의 width, height임.
				 - w, h가 작을수록 low level stage가 선택됨.
## Path Aggregation Network (PANet)
- FPN은 Resnet이 backbone이기에 사실은 stage마다 거리가 멈. 따라서 low level feature map이 high level feature map에 전달이 안될 수도 있음.
- PANet은 이것을 해결하기 위해 Bottom-up Path를 추가함.

    <img src="https://github.com/user-attachments/assets/2ab497f4-e030-4924-bb7f-c825570235b6" width="500"/>

- 또한, Adaptive Feature Pooling을 사용. (모든 feature map으로부터 ROI Align을 통해 모든 feature에 대한 정보를 결합.)

## After FPN
### DetectoRS
-  RPN (Region Proposal Network), Cascade R-CNN은 객체 위치를 위해 여러 번 생각을 반복함. 이것에 착안.
- RFP (Recursive Feature Pyramid)라는 방법을 소개함.
    - FPN (Feature Pyramid Network)을 재귀적으로 진행. 즉, neck 정보를 다시 backbone에 전달하여 backbone도 neck 정보를 다시 활용하게 함.
    - Backbone에 FPN이 들어갈 때 ASPP (Atrous Spatial Pyramid Pooling) 연산을 통해 들어감

        <img src="https://github.com/user-attachments/assets/fb45b407-5b8b-4109-964a-d08423b5f55d" width="300"/>
        
        - SPP (Spatial Pyramid Pooling)에 Atrous Convolution을 적용
## Bi-directional Feature Pyramid (BiFPN)

<img src="https://github.com/user-attachments/assets/8a0fc412-8074-45ce-a760-8f7d9629dd32" width="300"/>

- 모델 구조를 다음과 같이 단순화함.- 
    - input이 하나 이거나 위 쪽의 빨강 박스, 없어도 되는 노드들을 삭제
    - input을 output에 연결
    - 이렇게 만들어진 path 각각을 하나의 feature layer로 취급하여 repeated blocks으로 활용
- 또한, Weighted Feature Fusion을 제안함.
	- 단순히 FPN처럼 summation하는 것이 아니라 각 feature별로 가중치를 부여한 뒤 summation

        $$P_6^{td} = \text{Conv} \left( \frac{w_1 \cdot P_6^{in} + w_2 \cdot \text{Resize}(P_7^{in})}{w_1 + w_2 + \epsilon} \right)$$

        $$ P_6^{out} = \text{Conv} \left( \frac{w'_1 \cdot P_6^{in} + w'_2 \cdot P_6^{td} + w'_3 \cdot \text{Resize}(P_5^{out})}{w'_1 + w'_2 + w'_3 + \epsilon} \right)$$

        - $P_6^{td}$ (td = top-down) 의 경우 (위 사진에서 중간 (위에서) 첫 번째 노드), 단순히 더하지 $w_1$, $w_2$ 를 곱하여 더함.
            - $P_6^{out}$의 경우 (위 사진에서 오른쪽 (위에서) 두 번째 노드) 단순히 더하지 $w_1$, $w_2$, $w_3$를 곱하여 더함.
            - $\epsilon$는 분모가 0이 되지 않도록 분모에 더함. 
        - 이 때, 가중치들은 ReLU를 통과한 값으로 항상 0 이상

## NASFPN
- 단순 일방향(top->bottom or bottom ->top) summation 보다 좋은 방법이 있을까?
- 그렇다면 FPN 아키텍처를 NAS (Neural architecture search)를 통해서 찾자!

    <img src="https://github.com/user-attachments/assets/d5eb62ae-0846-4ade-b11b-702db07c5ad9" width="500"/>

- 단점 
	- COCO dataset, ResNet기준으로 찾은 architecture, 범용적이지 못함
		- Parameter가 많이 소요
	- High search cost
		- 다른 Dataset이나 backbone에서 가장 좋은 성능을 내는 architecture를 찾기 위해 새로운 search cost

## AugFPN
FPN의 문제
	- 서로 다른 level의 feature간의 semantic차이
	- Highest feature map의 정보 손실 (Top은 Top에서 정보 전달이 없음.)
	- 1개의 feature map에서 RoI 생성 (PANet은 해결했지만..)
- 이를 해결하기 위해 다음과 같은 구성을 함.
	- Consistent Supervision, Residual Feature Augmentation, Soft RoI Selection
- 여기서 Residual Feature Augmentation, Soft RoI Selection에 대해 중점적으로 알아보자.
    - Residual Feature Augmentation
        FPN (Feature Pyramid Network)에서 high feature map은 정보 손실이 일어나므로 Residual Feature을 high feature map에 넘겨줌

        <img src="https://github.com/user-attachments/assets/46d01dc4-e95a-443b-984f-d757deda39de" width="300"/>

        - $M_6$ 계산 방법
             - $C_5$ feature을 Ratio-invariant Adaptive Pooling을 통해 다양한 스케일의 feature map을 뽑음. (이를 통해 256 채널로 뽑음.)
            - 이 feature map을 합칠 때 Adaptive Spatial Fusion을 사용함.
        - Adaptive Spatial Fusion
            -  $M_6$을 만들기 위해서는 각기 다른 feature map을 같은 사이즈로 맞춰야 함. 이를 위해 UPSAMPLE을 해줌.
            - 이를 Summation을 하기 위해 가중치를 두고 fusion해야 함.
            - UPSAMPLE된 feature map들을 concat 하여 $NC \times h \times w$을 만듦. (위 사진에서는 feature map이 3이므로 $3C \times h \times w$ )
            - 이를 $1 \times 1$ CONV 네트워크에 입력시켜 $C \times h \times w$로 만들고, $3 \times 3$ CONV 네트워크에 입력시켜 $N \times ( C \times h \times w)$ 로 만듦.
            - 이후 channel-wise [[Sigmoid]] 연산을 하게 되면, $N \times (1 \times h \times w)$가 됨.
		        - 즉, 각 픽셀별로 N개의 값을 가지고 있는 것이고, 이 N이 wight가 됨.
	        - 이것과 원래의 feature map을 wighted sum을 해줌.
    - Soft RoI Selection
        - Residual Feature Augmentation을 통해 나온  feature을 $P_5$에 전달 후, neck을 통해 나온 feature들에 대해 Stage 매핑없이 모든 feature map에 대해 ROI Projection을 진행하고, ROI Pooling을 진행.
        - 이후, Channel-wise 가중치 계산 후 가중 합을 사용하여 ROI Pooling 시의 max pooling을 학습 가능한 가중 합으로 대체