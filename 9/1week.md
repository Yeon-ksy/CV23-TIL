# 9/2~9/9 TIL ✏️

## 태욱 [[Github](https://github.com/K-ple)]

-

## 상유 [[Github](https://github.com/dhfpswlqkd)]

-

## 지현 [[Github](https://github.com/jihyun-0611)]

- Translate one modality to a concept
  - 사람은 여러 형태의 정보를 받았을 때 뇌의 뉴런이 활성화됨
  - 이때, 어떤 하나의 정보를 다양한 modality로 받았을 때 모두 같은 뉴런이 활성화됨 (오프라윈프리 뉴런)
  - 따라서 정보가 어떤 modality로 들어오든 하나의 개념으로서 저장된다는 것을 알 수 있다.
- Multisensory fusion for concept learning
  - 멀티 모달은 위의 사람의 인지 방식에 기반해 multisensory fusion을 통해 concept을 학습 시킨다.
- Multimodal Categories
  ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/fa504c93-6e3e-4bc6-9ba7-bdbaf1ad9e33/ab9eb4b8-aabd-4f23-9d43-28cb91c19189/image.png)
- Data representation
  - Image data structure
    - A 2D image is represented by an intensity value of each pixel in 2D array structure
    - color image patch: 3D pathch - HxWxC
  - Video data structure
    - A stack of image frames
  - 3D data structure
    - Various data representation : Multi-view images, Volumetric(voxel), Part assembly, Point cloud, Mesh, Implicit shape
  - Text embedding
    - Characters are hard to use in machine learning
    - Map words (or tokens) to dense vectors
    **Token**
    - Numerical representation of words
    - Pre-fixed vector embedding (like word dictionary)
    **Tokenization**
    - Split the text data into small units – tokens.
    - ‘Numbers’ can be processed easily than ‘text’
    - Token ID to embedding through embedding layer
    - Token, not word
    - Embedding layer(look-up-table)
      - Input: index(sparse)
      - Output: token embed.
    - Embeddings are learned from scratch
    **Word2Vec –skip-gram model**
    - Trained to learn W and W’
    - Rows in W represent word embedding vectors
    - Learning to predict neighboring 𝑁 words for understanding relationships between words
      → 단어 간의 관계를 이해하기 위해 인접한 𝑁 words 예측을 학습
    - Given a model with a window of size 5, the center words depend on 4 words
      → window 크기가 5인 모델에서 중심 단어는 4개의 단어에 따라 달라짐
    - Emerging semantic relationship
  - Sound representation
    - Acoustic feature extraction from waveform to spectrogram
    - Short-time Fourier transform (STFT)
      : Fourier transform (FT) on windowed waveform results in frequency-magnitude graph
      → waveform에 window를 사용하여 시간 별로 자른뒤 해당 부분에 푸리에 변환을 사용하면 frequency-magnitude graph로 변환할 수 있다.
      - FT decomposes an input signal into constituent frequencies : 푸리에변환은 입력 신호를 구성 주파수로 분해함.
    - Spectrogram: A stack of spectrums along the time axis (위의 주파수 도메인으로 분해한 그래프를 스택하면 스펙토그램이 됨)
- Multi-modal alignment(matching )
  Application–Image tagging
  - Can find relevant tags of a given image, or retrieve images by a query keyword
  **CLIP**: Contrastive Language-Image Pre-training, by OpenAI
  - Learn visual concepts from the natural language supervision
  - Train with a wide variety of images and natural language pairs
    - 400 million (image, text) pairs collected from internet
  - Architecture
    – Image encoder : ViT-B (or ResNet50) - 이미지를 처리하고 해당 이미지의 고정 크기 벡터 표현을 생성
        – Text encoder : Transformers

        - 텍스트 설명을 처리하고 텍스트의 고정 크기 벡터 표현을 출력

        → 텍스트 인코더와 이미지 인코더 모두 각각의 입력을 동일한 차원의 벡터 임베딩으로 변환하여 두 양식을 비교

        → CLIP exhibits domain-robust performance
  **Contrastive learning**
  - Pull a target image (anchor) to a matching image (positive)
  - Push an anchor from many non-matching images (negative)
  - Given a batch of 𝑁(image, text) pairs
    - Predict embeddings for each modality : 각 modality에 대한 embedding 예측
    - Compute 𝑁×𝑁cosine similarities : 두 개의 embedding에 대해 코사인 유사도를 계산함
      cosine similarity:
      $$
      \langle I_i, T_i \rangle = \frac{I_iT_i}{\|I_i\|\|T_i\|}
      $$
  **Pre-training method for CLIP**
  - contrasitive learning objective:
    - Maximize the cosine similarities of the N correct embedding pairs
      → N개의 correct embedding pairs의 코사인 유사도를 최대화하는 것이 contrastive learning 목표
    - Minimize the cosine similarities of the ($N^2 - N$) incorrect pair
      → $N^2 - N$ 개의 incorrect pair에 대해서는 코사인 유사도를 최소화
    ⇒ Optimize a symmetric cross-entropy loss over the similarity scores
    - 대조 손실을 사용하여 올바른 이미지-텍스트 쌍이 잘못된 쌍보다 더 높은 유사성 점수를 갖도록 유도함.
    - 손실은 양방향에 대해 계산
      - 이미지-텍스트: 각 이미지에 대해 모델은 배치의 모든 텍스트 설명 중에서 올바른 텍스트를 찾는다.
      - 텍스트-이미지: 각 텍스트 설명에 대해 모델은 배치의 모든 이미지 중에서 올바른 이미지를 찾는다.
  **Diverse applications using pre-trained CLIP**
  - Image captioning
  - Image stylization with language
  - Image/video retrieval with text
  - Text-to-image generation
  - CLIP-guided motion generation
  - CLIP-guided 3D object/mesh generation
  - …
  - (18,000 citations within 3 years)

## 윤서 [[Github](https://github.com/myooooon)]

-

## 세연 [[Github](https://github.com/Yeon-ksy)] [[Velog](https://velog.io/@yeon-ksy/)]
