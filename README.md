# MOFY: MOsaic For You 실시간 불특정 인물 비식별화
<p align="center">
<img src=src/song_gif.gif width="600" />
</p>
- 송중기 클립 출처: 연예가중계 게릴라 데이트 (KBS2 예능)
배우 송중기를 제외한 모든 인물 비식별화 결과 (식별 대상에 박스 처리)


## 멤버
| [김영운](https://github.com/Cronople) | [이승현](https://github.com/sseunghyuns) | [임서현](https://github.com/seohl16) | [전성휴](https://github.com/shhommychon) | [허석용](https://github.com/HeoSeokYong) |  
| :-: | :-: | :-: | :-: | :-: |  
|<img src="https://avatars.githubusercontent.com/u/57852025?v=4" width=100>|<img src="https://avatars.githubusercontent.com/u/63924704?v=4" width=100> |<img src="https://avatars.githubusercontent.com/u/68208055?v=4" width=100> | <img src="https://avatars.githubusercontent.com/u/38153357?v=4" width=100> |<img src="https://avatars.githubusercontent.com/u/67945696?v=4" width=100>


## Project Overview 
### 목표
<p align="center">
<img src=src/MOPY_Logo.png style="zoom:75%;" />
</p>
웹서비스를 기반으로 사진이나 동영상, 실시간 영상에서 유저가 등록, 지정한 특정 인물들 이외의 사람들의 얼굴을 AI로 비식별화 처리해주는 초상권 보호 솔루션(Mosaic for you,  MOFY)을 제공하고자 한다.
실시간 영상에서도 지연이 적고 강력한 보호 솔루션이 프로젝트의 핵심 목표이다.


### 기대효과 
- 유튜브 야외 방송 / 동영상 촬영 등에서 원치않게 노출되는 일반인에 대한 실시간 비식별화로 일반인의 초상권을 보호할 수 있다.
- 실시간 방송에서 불특정 대상에 대한 초상권 보호로 방송 컨텐츠의 다양한 확장을 기대해볼 수 있다.
- 수동적인 동영상 편집 과정 없이 모자이크를 자동으로 처리해주어 편집 노동력을 줄일 수 있다.


## Demo 

1. 축구선수 손흥민을 제외한 모든 인물 비식별화 결과 (식별 대상에 박스 처리)
- [손흥민 클립 출처: 손세이셔널 – 그를 만든 시간 (tvN 시사/교양)]

<img src=src/son.webp width="600" />


2. 뉴스에서 기자와 인터뷰이를 제외한 모든 인물 비식별화 결과
- [뉴스 클립 출처: NEWS A 성혜란 팩트맨 인터뷰(Channel A)]

<img src=src/interview.webp width="600" />

3. 인터넷 방송에서 방송자와 인터뷰이를 제외한 모든 인물 비식별화 결과
- [유튜브 클립 출처: afreeca TV BJ 남순 NS 유튜브 (Youtube)]

<img src=src/inbang.webp width="600" />

## System Architecture
<p align="center">
<img src=src/system_architecture1.png width="1000" />
</p>

1. *Add Face Data* : 식별화 대상의 얼굴 영역을 embedding하여 데이터베이스화
2. *Select Process* Target : 이미지, 비디오, 웹캠 등 target을 결정
3. *Face Detection* : 이미지/영상 내 사람의 얼굴 탐지 
4. *Face Recognition*: 탐지된 얼굴을 식별 인물 데이터와 비교하여 일치도 검사
5. *Face Tracking*: 실시간 처리와 예측 오차를 줄이기 위해 이전 프레임의 정보를 활용하여 인물 식별 기능 강화 
6. *Mosaic* : 등록된 인물과 일치하는 대상을 제외한 인원에 대해 비식별화 처리

## Dataset 
- Detection Model 학습데이터 - **WIDERFACE**
  - 전체 3만장의 얼굴 이미지, 총 39만 명의 얼굴 라벨 
- Recognition Model 학습 데이터 - **VGGFace2**
  - 전체 331만장의 얼굴 이미지, 9131 classes
- 검증 데이터셋 (**AI-Hub 장면 인식·인물 인식을 위한 방송 영상 데이터셋 & AFD**)


## Model 

<p align="center">
<img src=src/model_pipeline.png width="800" />
</p>

> 모델 파이프라인 


<p align="center">
<img src=src/modelflow.jpg width="1000" />
</p>

> 하나의 이미지에 대한 비식별화 과정



## Future Works
- 모바일 앱 환경에서도 서비스를 제공. 
- 자동차 표지판 등 다른 부문에서 개인정보가 침해될 수 있는 정보도 비식별화.


## Using Library
- Pytorch
- OpenCV
- NumPy
- MoviePy


## Environment 
- Python 3.8 
