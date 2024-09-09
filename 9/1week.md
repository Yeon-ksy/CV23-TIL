# 9/2~9/9 TIL ✏️

## 태욱 [[Github](https://github.com/K-ple)
### CLIP

- 

## 상유 [[Github](https://github.com/dhfpswlqkd)]
- 

## 지현 [[Github](https://github.com/jihyun-0611)]
-

## 윤서 [[Github](https://github.com/myooooon)]
- 

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
