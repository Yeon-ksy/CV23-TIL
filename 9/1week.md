# 9/2~9/9 TIL ✏️

## 태욱 [[Github](https://github.com/K-ple)
### Multimodal
- 텍스트, 이미지, 음성, 영상 등 다양한 데이터 양식 (modality)을 함께 처리하는 것을 의미

### CLIP
- ViT(Vision Transformer)와 Transformer 언어 모델(Transformer-based language model)을 결합하여 이미지와 텍스트를 모두 처리할 수 있게 만들어놓은 모델
- 즉, Text와 Image의 관계성을 모델링

![taeuk_image1.png](image%2Ftaeuk_image1.png)
- 4억개 이미지와 해당 이미지에 대한 설명 Text를 pair로 둔 학습데이터셋을 각 인코더(Text,Image)로 임베딩 후 위사진과 같이 pair의 거리에따라 유사도를 계산 함


![taeuk_image2.png](image%2Ftaeuk_image2.png)
- Text의 경우 A photo of a {object}와 같이 단어 형태가 아닌 문장 형태로 인코딩시 성능 향상의 효과를 볼 수 있음
- 


## 상유 [[Github](https://github.com/dhfpswlqkd)]
- 

## 지현 [[Github](https://github.com/jihyun-0611)]
-

## 윤서 [[Github](https://github.com/myooooon)]
 ## [CV 이론] 
 4. Segmentation & Detection


## Semantic segmentation
이미지의 각 픽셀의 카테고리를 분류하는 것 (instance는 고려하지 않고 의미적 카테고리만 고려)
- 적용 분야 : Medical images, Autonomous driving, Computational photography

### 1. Fully Convolutional Networks(FCN)

<img src="https://github.com/user-attachments/assets/5413e4de-4488-4ab6-a350-a4364a493ab0" width=550/>

- semantic segmentation의 첫 end-to-end 구조 모델
- FCN은 전체 사진의 class를 예측하는 이미지 분류 모델을 튜닝해 픽셀 단위의 class를 예측하도록 학습시키는 Transfer Learning으로 구현한다.
    - 이미지 분류의 Fully Connected layer는 고정된 차원의 벡터를 출력하며 공간적 정보를 담지 못한다. FC 레이어 대신 **1x1 conv**와 **up-sampling**을 이용해 공간적 정보를 가지며 input과 크기가 같은 classification map을 출력한다.
        
- Up-sampling & Transposed convolution
    
    Conv layer를 통과하며 pooling이나 stride로 인해 해상도가 낮아진 feature map을 input size와 같아지도록 up-sampling한다.
    
    하지만 위치 정보가 손실된 feature map을 그대로 up-sampling하면 디테일한 classification map을 얻을 수 없다. 더 디테일한 정보를 가지고 있는 중간 level feature map을 최종 feature map와 더하는 skip connection 을 통해 이를 보완할 수 있다.   

    ![FCN_result](https://github.com/user-attachments/assets/3882c62e-1006-43ec-9e58-6722ff273f48)
    ![FCN_skipconnection](https://github.com/user-attachments/assets/2e7661ec-bb9a-4a7f-a7c0-12334f65b2c8)
    
    - FCN-32s
        
        원본 이미지 크기를 (H, W)라 하면, (H/32, W/32) 크기의 pool5를 32배 up-sampling
        
    - FCN-16s
        
        pool5를 2배 up-sampling하여 (H/16, W/16) 크기의 pool4와 더한 다음(=A), 16배 upsampling
        
    - FCN-8s
        
        FCN-16s의 map A를 2배 up-sampling하여 (H/8, W/8) 크기의 pool3과 더한 다음, 8배 up-sampling
        

### 2. U-Net

<img src="https://github.com/user-attachments/assets/8de12164-e03c-4f88-ba36-d4f652461906" width=550>

- U-Net은 down-sampling하는 contracting path와 up-sampling하는 expanding path의 대칭 구조로 이루어져 있다. Contracting path의 feature map을 expanding path의 feature map에 더해주는 skip connection을 통해 localized 정보를 전달한다.
- Contracting path
    - 3 x 3 conv을 두 번 적용 → 2 x 2 max pooling으로 down-sampling
    - down-sampling할 때 채널의 수가 2배로 늘어난다.
- Expanding path
    - 2 x 2 up-conv으로 up-sampling → 대응하는 contracting path의 feature map 더하기 → 3 x 3 conv을 두 번 적용
    - up-sampling할 때 채널의 수가 1/2로 줄어든다.

## Object detection

Classification + Box localization

- 적용 분야 : Autonomous driving, Optical Character Recognition(OCR)

### 1. Two-stage detector : R-CNN

![R_CNN](https://github.com/user-attachments/assets/8785085b-b288-4d65-bbee-436d3484b17a)

- 객체가 있을만한 영역을 제안해주는 region proposal과 객체를 분류하는 classification 단계를 순차적으로 진행하는 2-stage object detection 모델
- R-CNN은 사람이 만든 region proposal 알고리즘(Selective search)을 사용하여 bounding box를 추출한다. 이를 고정된 사이즈로 변형시키고(warping) 미리 학습된 CNN에 넣어 feature를 추출한다. 이 feature를 SVM object classifier에 넣어 객체를 분류하고, Box offset regressor에 넣어 bounding box offset을 조정한다.

### 2. One-stage detector : YOLO

<img src="https://github.com/user-attachments/assets/47f6c3a8-6f88-4e37-9966-9a35b6f471b2" width=550>

- 이미지를 한번만 보고 region proposal과 classification을 동시에 수행하는 1-stage object detection 모델로 속도가 빨라 실시간 객체 탐지에 유용하다.
- 이미지를 S x S grid로 나누고 grid cell마다 B개의 bounding box, 각 box에 대한 confidence(bounding box 선 굵기), C개의 conditional class probability를 예측한다.  전체 예측 결과는 S x S x (B x 5 + C)크기의 tensor로 인코딩된다(B x 5 + C → bounding box의 x, y, w, h, obj score + class probability). 이때 예측된 bounding box들 중 각 객체에 가장 정확한 하나의 박스를 선택하기 위해 Non-Maximum Suppression(NMS) 알고리즘을 사용한다.

### 3. One-stage detector vs. Two-stage detector

- One-stage (YOLO, RetinaNet…)
    - No explicit RoI pooling
    - Class imbalance problem - One-stage는 모든 pixel에 loss를 계산하는데, 객체가 있는 positive anchor box보다 배경 영역에 해당하는 negative anchor box가 더 많아 loss 계산에 어려움이 있다.
        
        → focal loss로 개선 (focal loss는 ground truth class일 확률이 높으면 down-weights, 낮으면 over-weights하는 방법)
        
- Two-stage(R-CNN, Fast R-CNN, Faster R-CNN…)
    - Regioin proposal로 제안된 box 영역을 가져와서 고정된 크기에 맞게 조절하는 RoI pooling이 존재한다.


## Instance segmentation

semantic segmentation + distinguishing instances (배경은 label을 부여하지 않음)

### 1. Mask R-CNN
- Faster R-CNN은 RoI pooling을 통해 크기가 줄어든 feature가 RoI를 반영하지 못하는 misalignment 문제가 있다. Mask R-CNN은 RoI pooling 대신 RoI align을 사용해 floating point까지 고려한 더 정교한 모델이다. 또한, 마지막 예측 단계에서 Mask head를 추가해 segmentation mask를 예측한다.
    - Extensions : DensePose R-CNN, Mesh R-CNN

## Transformer-based methods

### 1. DETR (Detection Transformer)

- End-to-End Object Detection with Transformers(encoder-decoder 구조를 사용)
- Object detection을 direct set prediction problem으로 만들면서 많은 hand-designed component들을 없앨 수 있게 되었다. 기존 object detection 모델에서 RPN이 같은 객체에 여러 가지 bounding box를 만들 때 최적의 box를 선택해주던 non-maximum suppression도 neural network 안으로 들어온 것이다.
- Method
    1. Feature extraction with CNN + position encoding
    2. Transformer encoder
        - CNN에서 추출된 feature map이 더 강화된 inceptive field를 고려하도록 만들기 위해 사용된다.
    3. Transformer decoder
        - Encoder output과 object queries를 input으로 가진다.
        - Object queries는 어떤 object가 어디에 있는지를 물어본다.
        - 출력이 다시 입력으로 나오는 auto-regressive 형태를 사용하지 않고, N개의 query를 병렬적으로 처리해 N개의 feature로 각각 decoding하도록 한다.
    4. Prediction heads(Feed forward network)
        - N개의 feature가 FFN을 거쳐 class와 bounding box의 형태로 출력
        - 예측하는 bounding box의 개수가 실제 object 개수보다 많도록 query를 설정한다.
        - class label ‘None’은 객체가 없음을 뜻한다.
    5. Bipartite matching
        - Loss를 계산하기 위해 예측한 box와 실제 box를 matching
        - 출력결과가 한번에 순서없이 나오기 때문에 어떤 label과 대응되는지 알 수 없다. 따라서 bounding box prediction set과 ground truth set을 matching하여 학습을 진행한다.

### 2. MaskFormer

<img src="https://github.com/user-attachments/assets/028f7c40-28e8-4812-b6ac-b2aaffde4b72" >

- Mask classification으로 semantic & instance segmentation 두 가지 task를 수행하는 하나의 모델을 구성할 수 있다는 insight에서 시작
- Pixel-level module
    - Backbone model이 low-resolution image features을 생성하고 pixel decoder가 image features을 up-sampling하여 per-pixel embedding을 출력한다.
- Transformer module
    - Transformer 모델의 decoder 부분으로 image features와 positional embeddings를 합친 값을 input으로 넣어 N개의 per-segment embedding을 출력한다. DETR과 마찬가지로 N개의 query를 병렬적으로 처리한다.
- Segmentation module
    - Per-segment embedding이 MLP를 거쳐 각 segment에 대한 classification을 수행한다.
    - Per-segment embedding이 MLP를 거쳐 mask embedding으로 변환된다. Mask embedding과 per-pixel embedding을 dot product한 후 sigmoid함수를 적용해 binary mask prediction을 수행한다.
    - Classification loss와 binary mask loss가 DETR처럼 set prediction으로 나오기 때문에 서로 matching하여 학습을 진행한다.

### 3. Uni-DVPS

<img src="https://github.com/user-attachments/assets/9c9d7fa4-f547-42c6-bc52-653b718cb3dc">

- 비디오에서 panoptic segmentation(semantic segmentation + instance segmentation)과 depth prediction을 한번에 수행하는 unified 모델
- MaskFormer와 유사하게 feature extractor와 pixel decoder, transformer decoder를 가지고 있다.
- Unified Transformer decoder with unified query
    - 여러 task를 수행할 수 있는 통합된 query를 사용
    - Segmentation을 위한 embedding과 depth prediction을 위한 embedding으로 분화된다.
- Feature gate
    - Pixel decoder에서 나온 feature map도 feature gate를 거쳐 두 task에 각각 유리한 feature로 decoding된다.
- Video segmentation을 위해서는 시간에 따른 추적이 필요하다. Uni-DVPS에서는 tracking module을 사용하지 않고 query-based tracking을 사용한다.
    - 같은 instance는 여러 frame에 걸쳐 비슷한 query feature가 나타나고 다른 instance는 서로 구분되는 query feature가 나타나는 특징이 있다. Frame 간 query matching을 통해 video tracking을 수행한다.

## 세연 [[Github](https://github.com/Yeon-ksy)] [[Velog](https://velog.io/@yeon-ksy/)]

# [CV 이론] Multimodal

## 1. CLIP

- 텍스트 통해 이미지를 검색하거나 이미지를 통해 텍스트를 추출할 수 있음.
- 이미지와 텍스트 인코더를 통해 각각의 특징 벡터를 비교 대조
    
    <img src="https://github.com/user-attachments/assets/1cec8297-7784-418e-9745-e36de558cfbe" width="500"/>
    
    - 이미지 인코더와 텍스트 인코더의 Joint embedding을 학습하여 관계성을 학습
        - 이미지 인코더 : ViT-B (또는 ResNet50)
        - 텍스트 인코더 : 트랜스포머
- loss (Contrastive learning을 사용)
    - 이미지가 주어졌을 때, cosine similarity을 통해 관련 있는 feature는 당기고, 관련 없는 것은 떨어트림.
    
    <img src="https://github.com/user-attachments/assets/8423e0b6-81af-47b1-ab10-028bc67901df" width="500"/>
    
    - pseudo code
        
        <img src="https://github.com/user-attachments/assets/25d43781-1a4b-4268-8ea4-ee3d13ddff7a" width="500"/>
        
        - logits 계산에 `np.exp(t)`를 사용하는 데, 이것은 temperature로 cross-entropy을 어느 정도 민감도로 학습을 진행할 지 정함.

### 1.1. CLIP 활용 :  ZeroCap

- GPT2와 CLIP을 결합하여 추가적인 training없이 캡션함.

    
- Method
    - 단어에서 나온 특성들과 이미지와 잘 맞는 지 측정 ($l_{CILP}$)
        
        <img src="https://github.com/user-attachments/assets/f04a9a54-ab12-40c6-891d-40e941f63eae" width="500"/>
        
        - 이를 위해 이미지에서 visual feature을 만들어 단어 특성과 비교
            - 만약 차이가 있으면, 역전파를 통해서 Context를 업데이트
    - 다음 단을 예측할 때, 기존에 있던 벡터들이 변하지 않도록 유지하는 cross-entropy loss를 추가
        
        <img src="https://github.com/user-attachments/assets/b5520a17-d163-4bc5-8709-4bab158fda20" width="500"/>
        

### 1.2. CLIP 활용 : ImageBIND

- 텍스트, 이미지, dept, IMU, audio 등 다양한 센서와의 관계를 통합.

### 1.3. CLIP 활용 : DALL-E 2

- Text-to-Image generation임.
- CLIP & diffusion models로 이루어짐.
- 먼저 학습한 CLIP과 diffusion 모델을 별도로 생성해서 연결하는 모듈 방식임.
    
    <img src="https://github.com/user-attachments/assets/f9265e76-c8c9-4d49-97fb-d50e3fad8936" width="500"/>
    

## 2. Visual-language model

### 2.1. Show, attend and tell

- 이미지의 일부 영역을 참조해서 캡션을 생성
    
    <img src="https://github.com/user-attachments/assets/cdf8361b-2fb9-44f7-9f75-ffe0a878bc99" width="500"/>
    
    - 이미지에 ConvNet을 써서 feature을 생성하고, LSTM을 통해 다음 단어가 사진의 어떤 부분을 참고를 해서 나와야하는지 예측하여 attention과 캡셔닝을 수행
- method
    
    <img src="https://github.com/user-attachments/assets/a4a1c3e6-0d56-4737-93f3-b5c81c0cbd46" width="500"/>
    
    - $s_1$ = 어떤 부분을 참조해서 캡션을 시작하는 지 attetion 맵을 추출.
    - 이 attention 맵과 feature을 서로 weighted combination을 통해 feature $z$를 추출.
    - 이 hidden state로부터 어떤 단어가 예측되어야 하는 단어를 디코딩하고 (d$_1$) 다음 단어가 참고해야 하는 영역을 attention을 함.
    - $z_2$을 계산하고 $h_2$를 생성. (이전 단어가 같이 들어가서 $h_2$를 생성)

### 2.2. Flamingo

- Transformer모델인 Chinchilla활용
    
    <img src="https://github.com/user-attachments/assets/8ac9edd7-02bc-48bf-8379-9ce049c67f79" width="500"/>
    
    - pre-training이 된 레이어은 fix하고 (눈꽃 표시), Learnable한 레이어를 (눈꽃 밑 연보라색) 삽입하여 이 부분만 학습을 함.
    - Vision Encoder을 통해 feature을 뽑고, Language Model의 레이어에 연결함.
        - 이 때, perceiver resamper을 사용하는 데, 이는 input 이미지의 사이즈가 다양한 사이즈를 가지고 있을 때, 항상 fixed-sized을 반환해줌
        - perceiver resamper
            
            <img src="https://github.com/user-attachments/assets/97597589-9492-490b-9568-a812163fac12" width="500"/>
            
            - Learned latent quries = 이 네트워크를 학습하기 전에 쿼리를 할당하고 학습하면서 쿼리 부분을 학습하고 fix해놓음. 이 개수만큼만 출력을 뱉으므로서 같은 차원의 벡터를 출력하게 함.
    - vision input은 key와 value 형태로 cross attentiom layer에 입력되고, language input은 query 형태로  cross attentiom layer에 입력됨.
    - tanh gating (cross attention 층 위에 조그만한 부분,  FFM 위의 조그만한 부분)
        - 초기화를 0으로 함. (이를 통해 skip connection 시, key와 query가 0이 되며, language input만 위 레이어로 올라가게 됨.→ 처음 시작시 기본 Language Model 형태로 시작하고 그 이후에 vision입력이 흘러 들어옴 )
- 이를 통해 파라미터는 적으면서 빠르게 학습할 수 있는 visual model이 만들어짐.

### 2.3. LLaVA

- Flamingo처럼 이미지가 주어졌을 때, 그 이미지를 가지고 대화할 수 있는 모델
    
    <img src="https://github.com/user-attachments/assets/67916872-e042-4ba1-b1fa-d379772fbc6d" width="500"/>
    
    - 먼저 Pre-training 된 LLM모델과 Vision Encoder를 가지고 trainable한 layer (projection W)하나만을 배치함.
        - projection W : vision encoder에서 나온 feature을 LLM이 이해할 수 있는 토큰 형태로 converting해줌
        - projection W을 학습하기 위한  visual instruction data를 GPT를 통해 만듦.

### 2.4. InstructBLIP

- LLaVA, Flamingo와 비슷한 모델
    
    <img src="https://github.com/user-attachments/assets/738de91e-7810-43e4-8b86-d2bdbee16f74" width="500"/>
    
    - LLaVA, Flamingo처럼 Pre-training 된 Vision Encoder을 사용
    - Q-Former : vision encoder에서 나온 feature을 Language model이 이해할 수 있는 토큰 형태로 converting해줌 (Flamingo의 perceiver)
- Q-Former
    
    <img src="https://github.com/user-attachments/assets/ae7121a3-9d79-47a2-ab2c-77fe0d3da268" width="500"/>
    
    - image-text contrastive learning = 이미지 feature와 텍스트 feature 사이에 관계를 측정
        - 이는 각각의 self-attention이 share되면서 구현. 이 때, attention masking을 통해 input text와 learned queries가 얼만큼 섞일 지, 어떤 방향으로 섞일 지 결정.

# 3. Other visual reasoning

### 3-1. Visual programming

- 이미지가 주어졌을 때, 이 이미지를 어떤 식으로 분해해서 재합성해야 하는 지 절차를 계획하고 프로그램을 생성하는 형태로 디자인됨.
    
    <img src="https://github.com/user-attachments/assets/06be4f25-6751-4644-b9d0-f02226e509cc" width="500"/>
    

### 3-2, PaLM-E

- LLaVA처럼 이미지가 들어왔을 때, 이미지를 language token으로 바꿔는 부분도 있지만, 텍스트를 생성해서 그 액션을 control foundation model로 입력해서 로봇을 제어하는 시그널로 converting하는 멀티모달 모델임.

# [CV 이론] Generative Models

## 1. Auto-Encoder (AE)

- 아이디어 : 굉장히 많은 training data를 NN에 넣자!
    
    <img src="https://github.com/user-attachments/assets/87710a34-5d7c-41ec-b861-b32f4b5845b8" width="500"/>
    

## 2. Variational Autoencoder(VAE)

- 가정 : 데이터는 잠재 공간에서 생성
    
    <img src="https://github.com/user-attachments/assets/c5410140-5d00-464c-9772-ae457abd5024" width="500"/>
    
- $p_{\theta} (x|z) = p(x|g_{\theta}(z))$
    - 이 $p_{\theta}$는 다음처럼 우도를 최대화 하는 방향으로 정의됨
- x로부터 z를 추정하는 방법은
    
    <img src="https://github.com/user-attachments/assets/5d723c88-5980-4a58-b197-807b68aab6f5" width="300"/>
    
    가 되는데, 이것 역시 분모에 x에 대한 우도가 들어가므로 추정하기 힘듦.
    
- 따라서 $p_{\theta}(z|x)$를 근사하는 인코더 네트워크 $q_Ø(z|x)$을 둠
    
    <img src="https://github.com/user-attachments/assets/761462c7-73ad-49d0-b45f-77cb5e80c66c" width="500"/>
    
    - Reparameterization trick이란 것을 사용함. (μ, σ는 differentiable 하지 않으므로)
        - 가우시안 노이즈 ε와 μ, σ을 통해 z를 만들어 사용. (선형성이 있기에 미분 가능!)

## 3.  Denoising Diffusion Probabilistic Models(DDPM)

- 데이터 $x_0$가 주어졌을 때, 노이즈를 조금씩 입혀서 latent로 매핑하고, 디노이즈를 통해 다시 $x_0$로 감. (Markovian forward and reverse process)
    
    <img src="https://github.com/user-attachments/assets/c9d1bf08-07d2-4081-9784-223f2ae0e442" width="500"/>
    
- loss
    
    <img src="https://github.com/user-attachments/assets/7466dce4-4009-46ee-bcaa-7dd2d99b7f6d" width="500"/>
    

## 4. Latent Diffusion (a.k.a. Stable Diffusion)

<img src="https://github.com/user-attachments/assets/54a30dfc-669e-41b8-a0a5-5c1606cb9775" width="500"/>

## 5. ControlNet

- Stable Diffusion을 foundation으로 사용자가 원하는 input을 하여 사용.
    
    <img src="https://github.com/user-attachments/assets/ded17701-92a7-40ac-9cc1-c069899770ec" width="500"/>
    
- c라는 컨디션이 들어왔을 때, zero-conv layer를 통과해서 0이 나오게 만들고, 그 다음 input을 더해줘서 trainable copy에 넘겨주고, 다시 zero-conv layer를 통과하여 출력에 더함.
    - trainable copy는 network의 파라미터를 copy한 것 (input C도 image기 때문에 이 구조에 대한 이해를 original network를 이용해서 빠르게 적용하기 위함)

## 6.  LoRA : Low-RankAdaptation

- pretrained 된 모델이 있을 때, 추가적인 학습할 때 사용하는 모듈
- 가정  : model이 adaptation이 일어날 때, weights의 변화도 전부 다 low dim 공간에서 일어난다.
    
    <img src="https://github.com/user-attachments/assets/d80664ca-4588-48e4-9770-b1bd5cc7abb4" width="500"/>
    
    - 따라서 weights를 건들여서 fine-turning하는게 아니라 추가적인 path를 줘서 이를 더해주는 모델임.
    - 출력 부분(B)을 0으로 초기화해서 초기에는 출력이 서서히 스며들게 함.

## 7. Generative models 활용사례

### 7-1. Prompt-to-Prompt Image Editing

- 텍스트를 통한 이미지 editing (원래 이미지를 최대한 보존하면서 editing하는 기술

### 7-2. InstructPix2Pix

- 하나의 이미지가 주어지고,  어떻게 변경할 지 text instruction이 주어지면, instruction을 instruction으로 하여 이미지를 바로 변경.
- 이 논문은 Image Editing을 supervised learning problem으로 접근함.
    - 즉, input 조건과 output pair dataset을 잘 만들어 놓고,  그것을 training함.

### 7-3. Marigold

- diffusion을 단순히 fine-turning해서 monocular depth estimation에 사용
    
    <img src="https://github.com/user-attachments/assets/4e9cc821-0c27-49ed-bba1-3a98e8a3ca2a" width="500"/>
    
    - Stable Diffusion에서 기존에 사용했던 Latent Encoder을 그대로 사용하여 이미지를 Latent화하고 ($z^{(x)}$, 4 채널),  depth도 이미지로 보고 Latent Encoder에 넣어 Latent화함 ($z^{(d)}$, 3채널).
    - $z^{(d)}$에 대해서만 denoizing diffusion을 실시. 가우시안 노이즈를 t 스텝에서 $z^{(d)}$에 더하여 perturbation함. (4 채널이 됨.)
    - 이를 이미지의 Latent와 concat하고, (8 채널이 됨.) u-net에 넣는데, 이때, u-net이 8채널을 input으로 사용할 수 있게 zero-conv 구조 등을 사용함.
    - 이를 통해 U-net이 노이즈를 예측하는 형태로 denoizing diffusion을 training함. (U-net이 fine-turning 됨.)

# [CV 이론] 3D Understanding

## 1.  3D reconstruction

### 1-1. NeRF

- x, y, z 포인트하고 그 포인트를 바라보는 방향인 viewing direction을 입력으로 주면, 그 포인트에 대한 색깔과 투명도를 출력.
    
    <img src="https://github.com/user-attachments/assets/fdd5debf-e6e1-47f1-9bf8-282c1c704dab" width="500"/>
    
- 모델 출력과 이미지의 표현 형태가 호환되지 않음 (입력이 3D 포인트에 대한 색깔임.)
- 따라서 volumetric rendering이라는 테크닉을 활용
    - 컴퓨터 그래픽스에서 사용하는 렌더링 방정식을 이용하여 이미지를 렌더링하도록 계산
        
        <img src="https://github.com/user-attachments/assets/7d2b4b04-bd06-4899-9f20-41e55202f45a" width="500"/>
        
        - 하나의 픽셀은 카메라 센터로 들어오는 직선 상의 모든 객체의 색깔의 누적으로 이뤄지는데, (카메라 센서) 이를 모델링한 식임.
        - $σ(r(t))$ = density. 이것이 0이면, 투명하게 취급 (누적하지 않음)하고, 1이면 완벽하게 불투명한 입자이므로 그것을 고려함.
        - $c(r)$ = 랜더링된 plane 상에서의 한 픽셀
        - $T(t)$ = 물체 뒤 쪽의 색깔까지 누적되는 것을 막기 위함. 가려진 순간 전까지 1을 가지다가 가려진 물체를 만나면 0이 됨.

### 1-2. 3D Gaussian Splatting (3DGS)

- 3D Gaussian Splatting은 NeRF에 비해 10배 이상의 도약을 보여주며, 가장 활발하게 사용됨.
    
    <img src="https://github.com/user-attachments/assets/e16afe5c-6747-4e84-8b88-5ec8e5a791f6" width="500"/>
    
- NeRF에서는 전체 장면을 하나의 NN로 표현했다면, 3D Gaussian Splatting은 3D에서 국부적인 공간을 하나의 가우시안으로 표현
- 각 3D 가우시안은 Mean μ (i.e., x, y, z), Covaraince Σ, Opacity σ, Color parameters; (R, G, B) (or spherical harmonics (SH))로 이루어짐.
    - mean하고 covariance은 3D 가우시안 자체를 표현하고, 이 가우시안에 대한 Opacity, Color parameters을 가짐.
    - covariance은 positive semi-definite matrix이므로 이 특성을 유지하면서 학습하는 게 쉽지 않아  Rotation Matrix $R$ 과 diagonal한 scaling matrix $S$로 symmetric하게 결합해서 대신 표현

        <img src="https://github.com/user-attachments/assets/8bdcafdd-abcf-4547-989d-81dd43cfcfa0" width="200"/>
    

## 2. 3D generation

### 2-1. Mesh R-CNN

- Mesh R-CNN=Faster R-CNN+3D surface regression branch
    
    <img src="https://github.com/user-attachments/assets/68b0bb99-2361-4ee4-96ab-dcc52d2ec8e8" width="500"/>
    
    - 이미지가 주어지면 한정된 class에 대해서 mesh output이 나오는 구조이므로 이미지와 이미지에 대응하는 3D가 잘 정렬된 큰 데이터셋이 필요함.

### 2-2. DreamFusion

- 텍스트와 3D pair가 없이도 좋은 퀄리티의 3D를 생성하는 zero-shot 방법
- Diffusion같은 생성 모델은 아니고 training없이 최적화 형태로 3D를 구현
    
    <img src="https://github.com/user-attachments/assets/a7009752-2c56-4560-acac-44e4497ce7a0" width="500"/>
    
- Score Distillation Sampling (SDS) loss
    - 렌더링된 이미지에 노이즈를 첨가하여 diffusion model의 입력으로 넣어주고, 이 이미지가 fitting되었으면 하는 text도 넣어줌
    - 이 노이즈 낀 이미지를 text를 따르는 형태로 denoizing하도록 함 (현재 상태에서 어떻게 바뀌어야 text를 따르는 형태로 되는 지 방향을 알려주는 것)
    - U-Net 구조에 대해서는 역전파를 진행하지 않음.  (시간이 오래 걸리게 되므로)

### 2-3. Paint-it

- 3D가 아닌 3D 텍스쳐를 생성
- SDS loss의 그레이디언트가 노이즈해서 좋은 3D 생성에 방해가 될 수 있음. 이를 해결하기 위해 SDS loss를 사용하면서도 텍스쳐 맵을 convNet으로 parameterization함.
    - 즉, 입력은 랜덤 노이즈를 샘플링해서 fix하고 convNet도 랜덤 초기화를 함. 대신 output이 텍스쳐 맵이 될 수 있도록 resolution을 맞춰놓음.
    - 처음에는 랜덤한 값이 나오지만,  차즘 텍스쳐 맵을 표현하는 CNN weight가 최적화 됨.

# [CV 이론] 10. 3D Human

## 1. SMPL : Body model

- 신체 모델 𝑀(∙)은 작은 수의 포즈, 모양 및 기타 매개변수를 취하고, 3D mesh을 반환함.
    
    <img src="https://github.com/user-attachments/assets/61d8bddc-2e6e-41b3-ad89-1e98e7a6794a" width="500"/>
    
- SMPL은 mesh를 약 7,000개의 3D Vertex로 표현하고 있어 21,000개의 파라미터로 (x,y,z 3차원 인듯) 사람을 표현함.
- body model은 체형과 관련된 pose와 identity 파라미터를 통해 모델링 됨. 이 두 요소를 독립 요소로 생각해서 학습이나 추론을 더 간단하게 함.
- SMPL parameterization
    
    <img src="https://github.com/user-attachments/assets/fed0d10f-7d80-4a8a-a277-cc1a01e51a01" width="500"/>
    
    - $T$ :  Shape training set (Template) = CAESAR 데이터셋을 이용해서 정교한 포즈를 가지는 데이터를 획득. 이 데이터들의 vertex 값들을 각 위치에 맞게 평균내서 Template mesh에 대응하는 평균 mesh를 만듦
    - $S$ : Shape blend shape matrix = 평균 대비 변화를 잘 모델링 하기 위해서 mesh를 벡터로 표현하고 템플릿 mesh를 평균으로 고려를 해서 각 mesh에서 빼주고 이후 PCA를 수행하여 차원을 축소함. ($U$) U를 shape parameters와의 선형 결합을 통해 각각의 편차 벡터를 표현
    - $W$ : Blend weights matrix = Skinning을 통해 자연스러운 움직임을 만듦.
        - Skinning : Rest pose vertices (템플릿 mesh), Joint locations, Weights, Pose parameters가 주어지면, 각 포즈에 맞는 3D mesh를 출력하는 것
    - $p$ : Pose blend shape matrix = 실제 피부는 자세에 따라 밀리기에 모델링에 한계가 있dj 이를 보완하기 위한 행렬. Blend shape을 rest pose에 선형 결합을 통해 더하므로서 자세를 보완
    - $J$ : Joint regressor matrix = 조인트 포인트들을 정의하는 행렬. 미리 정의해두고 사용.
- 최종 SMPL parameterization
    
    <img src="https://github.com/user-attachments/assets/18b31acb-4bcc-4ca0-997a-5e3e066fd6c3" width="500"/>
    

## 2. SMPLify

- 2D Joint feature를 추출하고(Bottom up) 3D 바디 모델을 카메라 메트릭스를 통해서 projection하여 2D 도메인으로 처리한 다음에 앞에서 구한 2D Joint와 projection된 모델의 차이를 구하여  (Top down) 최적화에 사용
    
    <img src="https://github.com/user-attachments/assets/cf2d3c59-e2a1-42ac-bcbc-d1b99264b96b" width="500"/>
    
    - 입력으로는 안정적으로 동작하는 2D Point 추정을 사용해서 2D Point joint를 미리 구해 입력으로 넣음. 카메라 행렬은 포즈 전체에 회전을 고려한 간단한 2*3 행렬을 사용
- Depth Ambiguity는 Pose and Shape Prior을 통해 해결
- Interpenetrations는 Approx. surface with capsules and penalize intersections을 통해 해결
- Objective Function
    
    <img src="https://github.com/user-attachments/assets/91f671fc-34cc-449b-a683-bdca742ff020" width="500"/>
    
    - data term
        - 2D Joint estimation을 통해서 구한 값($J_{est, i}$)과 초기화 값인 $\beta$로부터 추정된 조인트 값 $J(\beta_i)$가 주어지면, 이를 $R_{\theta} (*)$을 적용하여 3D 좌표가 정해지게 되고, 카메라 메트릭스를 통해 projection하여 2D로 만듦.
        - 이 2D로 Projection된 Joint하고 측정된 $J_{est, i}$와 차이를 계산. ρ는 $L_2$, $L_1$과 같은 norm을 지칭
        - $w_i$ : 일부 joint가 가려져서 보이지 않을 때 조인트에 대해서 loss를 측정하지 않게 하는 값. (마스크 역할)
    - Prior term - 조인트가 부자연스럽게 꺾이는 것을 방지하는 term
        - prior = 현재 어떤 파라미터가 주어졌을 때, 그것이 발생하기 쉬운 경우인지 불가능한 경우인지 판단하는 기준
        - unnatural joint bending, prior on pose, prior on shape, prior on interpenetration 등의 term이 있음. (세부 내용은 생략)

## 3. SPIN

- SMPL을 NN과 연동하여 더 강건하고 효율적인 방법 제시
    
    <img src="https://github.com/user-attachments/assets/8e990c2c-8b27-4731-af2c-e30c1feb88fd" width="500"/>
