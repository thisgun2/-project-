# 파일 구조 설명
본 프로젝트는 7개의 ipynd 파일이 있습니다.
최상위 폴더에는 데이터 폴더와 코드 폴더가 있습니다.
![alt text](image-3.png)

archive.zip 만 캐글에서 제공하는 데이터가 나머지 폴더, ipynd, .pt파일, .csv 파일은 모두 제가 구축하였습니다.
## data 폴더
데이터 폴더를 들어간다면 우선 4개의 폴더가 있습니다.

### archive.zip 파일
맨 처음 캐글에서 데이터를 다운받을때 존재하는 zip 파일입니다.

### archive 파일
archive.zip 파일을 해제한 파일입니다.
/code/data_split/data_split.ipynd 에서 해제를 합니다.
들어간다면 4개의 파일이 있고 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.

### original 파일
archive 파일에서 train과 test를 7:3 비율로 나눈 데이터입니다.
/code/data_split/data_split.ipynd 파일에서 해당 코드가 실행이 됩니다.

original 파일에는 train 폴더와 test 폴더가 존재하며 train 과 test 폴더 안에는
 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.

### agumentation 파일
train 과 test 폴더가 들어있습니다.
test와 train폴더 안에는 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.

train 폴더 안에는 pituitary 파일 meningioma 파일 healthy , healthy(noisy0.5), healthy(noisy0.4), healthy(noisy0.3), healthy(noisy0.2), healthy(noisy0.1)파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.<br>
test폴더 안에는 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.
healthy(noisy0.5), healthy(noisy0.4), healthy(noisy0.3), healthy(noisy0.2), healthy(noisy0.1)파일은
/code/data_agumentation.ipynd 에서 생성이 됩니다.




## Code 폴더<br>
추천 실행순서
data_split.ipynd(굳이 실행 안해도됨 이미 생성된 데이터를 함께 제출했기 때문)->data_agumentation.ipynd(굳이 실행 안해도됨 이미 생성된 데이터를 함께 제출했기 때문)->resnet(agumentation).ipynd->densenet(agumentation).ipynd->efficientnet(agumentation).ipynd -> vit(agumentation).ipynd -> efficientnet(original).ipynd
# data_split.ipynd 파일
archive.zip 파일은 unzip 하고 train과 test 셋을 7:3으로 나눈 ipynd 입니다.<br>
필요 라이브러리<br>
import zipfile<br>
import os<br>
import os<br>
import shutil<br>
import random<br>
![alt text](image-4.png)

맨 위에 그림 폴더 구조도를 참고하여
zip_path 의 archive.zip 경로를 넣어주고<br>
extra_dir 의 archive.zip 을 unzip할 경로를 넣어주면 됩니다.

![alt text](image-5.png)
source_dir의 위에서 unzip한 경로를 넣어주고 실행한다면 original 폴더에 7:3 비율의 train , test가 생깁니다.<br><br>
***캐글 다운 직후 원본 데이터를 unzip하고 훈련데이터와 테스트 데이터를 나누어서 생성하는 코드인데 생성된 데이터가*** ***실제로 archive 파일 orignal 파일에 저장되었는데 이 파일(데이터 이미지) 또한 같이 첨부하였기에 굳이 실행을 안 해도 되는 파일입니다.***



# data_agumentation.ipynd 파일
original/train/ 에서 healty 데이터를 가우시안 노이즈 0.1 0.2 0.3 0.4 0.5로 부과하여 복구하는 식에 데이터 증강을 하는 코드입니다.<br>
필요 라이브러리<br>
import matplotlib.pyplot as plt <br>
import torch <br>
from torch import optim <br>
from torch import nn <br>
import torch.nn.functional as F <br>
from torch.utils.data import DataLoader <br>
from torchvision.datasets import MNIST <br>
from torchvision.transforms import ToTensor <br>

import sys <br>
print(sys.version) <br>
from typing import List <br>
from typing import Tuple <br>
from typing import Optional <br>

import torch <br>
import torch.nn as nn <br> 
import torch.nn.functional as F <br>

import os <br>
from typing import List <br>
from PIL import Image <br>
import torch <br>
import torch.nn as nn <br>
import torch.optim as optim <br>
from torch.utils.data import Dataset, DataLoader <br>
import torchvision.transforms as T <br>
import matplotlib.pyplot as plt <br>

![alt text](image-7.png)
이런 식으로 desnoised 된 데이터가 증강이 됩니다.
folder_path = '/project/data/original/train/healthy' 이런식으로 증강할 원본 데이터를 불러오고 <br>
save_dir = f"/project/data/agumentation/train/healthy(noisy{noise_train_sigma})" <br> 이런식으로
서로 다른 노이즈로 생성된 이미지를 저장하므로 경로설정의 유의하시길 바랍니다.<br><br>
***orginal 폴더의 healthy 데이터를 증강하는 파일인데 증강된 파일이***
***agumentation에 저장되어 증강된 데이터 또한 함께 제출하였기에 굳이 실행을 하지 않아도 되는 코드입니다.***

## data_agumenataion 파일 classfication model(vit(agumentation).ipynd , resnet(augmentation).ipynd, resnet(augmentation).ipynd, efficientnet(augmentation).ipynd)

데이터 증강을 적용하여 클래스 불균형을 발생한 데이터를 훈련데이터로 사용하여 모델을 훈련하고 평가하는 파일입니다.<br> 
***데이터 증강을 적용한 파일이므로 /project/data/agumentation/train 이 훈련데이터입니다***<br>
![alt text](image-8.png)<br>
<br>경로 주의<br><br>
![alt text](image-10.png)<br>
<br>클래스 불균형 구축<br>


![alt text](image-12.png)
<br> 뇌는 좌우반전이기에 RandomHorizontalFilp() 으로 데이터의 다양성 보장<br><br>


***기존에 pip install timm 으로 timm 라이브러리를 사전에 설치하시기를 반드시 추천합니다.***

![alt text](image-9.png)
모델이 4진분류 문제로 fine-tuning 합니다.

<br><br><br>

![alt text](image-13.png)
<br> 파라미터 학습 설정 여부 부여

<br>


![alt text](image-16.png)
토탈 파라미터, 훈련 파라미터, 옵티마이저 스케듈러 설정<br><br><br><br>

![<alt text>](image-17.png)
만든 모델 .pt 로 저장
<br><br><br><br>
![alt text](image-18.png)
accuracy 확인<br><br><br><br>


![alt text](image-20.png)
혼동행렬 생성
![alt text](image-21.png)
AUC 확인
<br>
![alt text](image-22.png)
<br>Sensitivity(recall), Specificity, F1_score 확인


## data_balance 파일 efficientnet(orginal).ipynd

해당 파일은 앞서 복원기법으로 데이터를 증강하여 훈련한 모델과 반대하여 증강된 데이터가 사용되지 않고 원본의 균형적인 데이터로 classification을 진행한 파일입니다. 증강된 데이터 모델중 efficientNet이 auc측면에서 가장 우수한 성능을 보여 efficientNet으로 증강된 모델(데이터 불균형)과 증강되지 않은모델(데이터 균형)을 비교합니다.
<br>
![alt text](image-23.png)
<br> 앞선 실험과 달리 데이터가 균형적인 형태입니다.

<br>

![alt text](image-25.png)
<br>Accuracy


<br>

![alt text](image-26.png)
<br> 혼동행렬

<br>

![alt text](image-28.png)

<br> AUC

<br>

![alt text](image-29.png)

<br>해당 모델은 복원기법으로 증강된 데이터가 훈련에 쓰이지 않습니다. 이에 복원기법으로 증강된 데이터가 실제 음성 데이터 분포에 맞게 생성되었는지 간접적으로 판단한 실험으로 현재 증강데이터(음성)이 train 되지 않은 데이터에 test 데이터로 증강데이터(음성) 을 test 한 결과입니다.
<br><br><br><br><br><br><br><br><br>

***!pip install grad-cam 을 통해 xai의 grad-cam을 설치할 것을 반드시 추천합니다.***
<br><br><br><br>
![alt text](image-30.png)

모델이 병변을 분류할때 병변이미지의 구도, 각도, 배경이 아닌 병변 정의에 맞게 병변을 탐지하고 잘 분류했음을 확인할 수 있습니다.


