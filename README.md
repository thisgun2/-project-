# 파일 구조 설명
본 프로젝트는 7개의 ipynd 파일이 있습니다.
최상위 폴더에는 데이터 폴더와 코드 폴더가 있습니다.
<img width="841" alt="image" src="https://github.com/user-attachments/assets/2e641fcb-c018-4616-bbaf-9393d600dd48" />
<br>***반드시 주의하셔야 합니다. 모든 코드에서 경로 설정에서 최상위는 /project 로 시작합니다! /project 이후에는 /project/data 폴더 /project/code 폴더가 있습니다. 만약 /project 폴더가 본인 환경에서 최상위 폴더가 아니라면 /project 앞에 자신의 경로 코드를 덧붙이거나 경로 수정을 꼭 하셔야합니다***
또한 설치 라이브러리 같은경우 !pip 형태를 주석처리하였으니 만약 라이브러리가 호출이 안된다면 직접 라이브러리를 설치하거나 필요한 !pip 의 주석을 해제하여 설치하시길 바랍니다.
archive.zip 만 캐글에서 제공하는 데이터가 나머지 폴더, ipynd, .pt파일, .csv 파일은 모두 제가 구축하였습니다.
# 실행
***반드시 주의하셔야 합니다. 모든 코드에서 경로 설정에서 최상위는 /project 로 시작합니다! /project 이후에는 /project/data 폴더 /project/code 폴더가 있습니다. 만약 /project 폴더가 본인 환경에서 최상위 폴더가 아니라면 /project 앞에 자신의 경로 코드를 덧붙이거나 경로 수정을 꼭 하셔야합니다***
또한 설치 라이브러리 같은경우 !pip 형태를 주석처리하였으니 만약 라이브러리가 호출이 안된다면 직접 라이브러리를 설치하거나 필요한 !pip 의 주석을 해제하여 설치하시길 바랍니다.
**파일 구조 설명의 그림을 반드시 확인하셔야 합니다** <br>
모든 코드파일은 ipynd로 쥬피터 노트북 환경에서 작용하여 control+5 키로 셀마다 실행하면 됩니다.

## data 폴더
데이터 폴더를 들어간다면 우선 4개의 폴더가 있습니다.

### archive.zip 파일
맨 처음 캐글에서 데이터를 다운받을때 존재하는 zip 파일입니다.

### archive 파일
archive.zip 파일을 해제한 파일입니다.
/project/code/data_split/data_split.ipynd 에서 해제를 합니다.
들어간다면 4개의 파일이 있고 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.

### original 파일
archive 파일에서 train과 test를 7:3 비율로 나눈 데이터입니다.
/project/code/data_split/data_split.ipynd 파일에서 해당 코드가 실행이 됩니다.

original 파일에는 train 폴더와 test 폴더가 존재하며 train 과 test 폴더 안에는
 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.

### agumentation 파일
train 과 test 폴더가 들어있습니다.
test와 train폴더 안에는 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.

train 폴더 안에는 pituitary 파일 meningioma 파일 healthy , healthy(noisy0.5), healthy(noisy0.4), healthy(noisy0.3), healthy(noisy0.2), healthy(noisy0.1)파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.<br>
test폴더 안에는 pituitary 파일 meningioma 파일 healthy 파일, glimoa 파일이 있고 파일 내부에 병변 이미지가 존재합니다.
healthy(noisy0.5), healthy(noisy0.4), healthy(noisy0.3), healthy(noisy0.2), healthy(noisy0.1)파일은
/project/code/data_agumentation.ipynd 에서 생성이 됩니다.




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
<img width="82" alt="image" src="https://github.com/user-attachments/assets/487b6816-72e9-4d8c-99ea-407cfae39f79" />
<br>라이브러리 import가 안된다면 해당 라이브러리를 꼭 설치하여야합니다. <br><br><br>
코드에는 주석처리를 하였으니 라이브러리가 호출이 안된다면 주석처리를 해제하시고 설치하시길 바랍니다.<br>
<br><br><img width="342" alt="image" src="https://github.com/user-attachments/assets/043de3ef-6100-4b32-8769-2fcd0045eb9d" /><br><br>


맨 위에 그림 폴더 구조도를 참고하여
zip_path 의 archive.zip 경로를 넣어주고<br>
extra_dir 의 archive.zip 을 unzip할 경로를 넣어주면 됩니다.<br>

<img width="361" alt="image" src="https://github.com/user-attachments/assets/7a8cd0ee-166b-409d-930b-029ac88ea218" /><br>

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
<br> 라이브러리 호출이 안된다면 !pip install torch torchvision matplotlib pillow 를 설치하시길 바랍니다. 주석처리를 하였으니 라이브러리를 호출이 안된다면 주석처리를 해제하고 설치하시길 바랍니다.<br><br><br>
<img width="320" alt="image" src="https://github.com/user-attachments/assets/dc2d514b-7643-44df-b4e7-b2006edbf40e" />

이런 식으로 desnoised 된 데이터가 증강이 됩니다.
folder_path = '/project/data/original/train/healthy' 이런식으로 증강할 원본 데이터를 불러오고 <br>
save_dir = f"/project/data/agumentation/train/healthy(noisy{noise_train_sigma})" <br> 이런식으로
서로 다른 노이즈로 생성된 이미지를 저장하므로 경로설정의 유의하시길 바랍니다.<br><br>
***orginal 폴더의 healthy 데이터를 증강하는 파일인데 증강된 파일이***
***agumentation에 저장되어 증강된 데이터 또한 함께 제출하였기에 굳이 실행을 하지 않아도 되는 코드입니다.***

## data_agumenataion 파일 classfication model(vit(agumentation).ipynd , resnet(augmentation).ipynd, resnet(augmentation).ipynd, efficientnet(augmentation).ipynd)
필요 라이브러리
!pip install torch torchvision timm scikit-learn pandas seaborn matplotlib 를 설치하는 것을 추천합니다.
바로 설치하기보다는 라이브러리가 호출이 안될때마다 해당 라이브러리를 설치하는것을 추천합니다. <br>

데이터 증강을 적용하여 클래스 불균형을 발생한 데이터를 훈련데이터로 사용하여 모델을 훈련하고 평가하는 파일입니다.<br> 
***데이터 증강을 적용한 파일이므로 /project/data/agumentation/train 이 훈련데이터입니다***<br>
<img width="196" alt="image" src="https://github.com/user-attachments/assets/b0a5b357-232c-4b74-99c7-2c58bc1b2e81" />

<br>***경로 주의 /project가 최상단 경로이고 /project/data/augmentation/train의 있는 경로를 꼭 train의 경로로 로드해야합니다.***<br><br>
<img width="152" alt="image" src="https://github.com/user-attachments/assets/e830cf53-830d-4361-bc7b-2457bd6c8a02" />
<img width="315" alt="image" src="https://github.com/user-attachments/assets/fe81baad-5685-4977-87ea-83cbc3649aaa" />

<br>클래스 불균형 구축<br>


<img width="196" alt="image" src="https://github.com/user-attachments/assets/e55af37d-9a1d-48ea-9626-29832019c839" />

<br> 뇌는 좌우반전이기에 RandomHorizontalFilp() 으로 데이터의 다양성 보장<br><br>


***기존에 pip install timm 으로 timm 라이브러리를 사전에 설치하시기를 반드시 추천합니다.***

<img width="410" alt="image" src="https://github.com/user-attachments/assets/bfdb6a3f-8ba8-4be1-97f7-df825c04d13c" />

모델이 4진분류 문제로 fine-tuning 합니다.

<br><br><br>

<img width="194" alt="image" src="https://github.com/user-attachments/assets/cfe8ddbb-f1e0-4e53-ba22-90611b2c7e75" />

<br> 파라미터 학습 설정 여부 부여

<br>


<img width="474" alt="image" src="https://github.com/user-attachments/assets/0c16d0fa-afed-4cff-9e88-4e0add837d86" />

<br>옵티마이저 스케듈러 설정<br><br><br><br>

<img width="520" alt="image" src="https://github.com/user-attachments/assets/c2d862b8-b088-4588-832b-637e5a7c4343" />

만든 모델 .pt 로 저장
<br>***경로를 꼭 주의하고 model 폴더까지 존재해야 .pt 가 저장이 됩니다!***
<br><br><br><br>
<img width="779" alt="image" src="https://github.com/user-attachments/assets/238e466e-2b7f-40a8-87d6-1c864a877a2a" />
accuracy 확인<br><br><br><br>
***경로를 꼭 주의하고 evaluate 폴더까지 존재해야 .csv 가 저장이 됩니다!***


<img width="442" alt="image" src="https://github.com/user-attachments/assets/19b58236-b0a8-4d27-ab38-aa16b9c69369" />
<br>혼동행렬 생성<br>
혼동행렬을 생성할때 위에 경로로 저장한 csv 파일을 불러와서 생성하므로 위에 csv 경로를 꼭 잘 저장하시고 해당 경로에 저장된 csv파일을 불러와야합니다.
맨 위 폴더 구조도 사진에 폴더 구조가 어떻게 되어있는지 반드시 확인이 필요합니다.
<img width="650" alt="image" src="https://github.com/user-attachments/assets/6c4ae800-87e6-4c6f-b4bb-ef9f07d672c0" />
AUC 확인
ROC curve를 생성할때 위에 경로로 저장한 csv 파일을 불러와서 생성하므로 위에 csv 경로를 꼭 잘 저장하시고 해당 경로에 저장된 csv파일을 불러와야합니다.
맨 위 폴더 구조도 사진에 폴더 구조가 어떻게 되어있는지 반드시 확인이 필요합니다.
<br>
<img width="122" alt="image" src="https://github.com/user-attachments/assets/c0a09c63-f8d9-44b7-a96e-11dd4a7f0d59" />

<br>Sensitivity(recall), Specificity, F1_score 확인


## data_balance 파일 efficientnet(orginal).ipynd

해당 파일은 앞서 복원기법으로 데이터를 증강하여 훈련한 모델과 반대하여 증강된 데이터가 사용되지 않고 원본의 균형적인 데이터로 classification을 진행한 파일입니다. 증강된 데이터 모델중 efficientNet이 auc측면에서 가장 우수한 성능을 보여 efficientNet으로 증강된 모델(데이터 불균형)과 증강되지 않은모델(데이터 균형)을 비교합니다.
<br>
<img width="321" alt="image" src="https://github.com/user-attachments/assets/da8be104-da33-4d02-8105-f0e43b0febbc" /> <br>
<br> <img width="135" alt="image" src="https://github.com/user-attachments/assets/dcbbf1cb-be53-4570-8353-abd2f2df5c3f" />
경로는 똑같지만 noisy -> denosied 를 제거하였기 때문에 균형적인 형태입니다.

<br> 앞선 실험과 달리 데이터가 균형적인 형태입니다.

<br>

<img width="683" alt="image" src="https://github.com/user-attachments/assets/effa408b-114b-4e7b-84e7-8044a5e1f64c" />

<br>Accuracy


<br>

<img width="413" alt="image" src="https://github.com/user-attachments/assets/0ea376bb-383e-4ff8-beba-621885aab1f6" />

<br> 혼동행렬

<br>

<img width="651" alt="image" src="https://github.com/user-attachments/assets/05de1ba3-fd71-4a4f-a528-b5032edb18f1" />


<br> AUC

<br>

<img width="435" alt="image" src="https://github.com/user-attachments/assets/dc8aba1d-1582-48a1-9337-04500ccac65a" />


<br>해당 모델은 복원기법으로 증강된 데이터가 훈련에 쓰이지 않습니다. 이에 복원기법으로 증강된 데이터가 실제 음성 데이터 분포에 맞게 생성되었는지 간접적으로 판단한 실험으로 현재 증강데이터(음성)이 train 되지 않은 데이터에 test 데이터로 증강데이터(음성) 을 test 한 결과입니다.
<br><br><br><br><br><br><br><br><br>

***!pip install grad-cam 을 통해 xai의 grad-cam을 설치할 것을 반드시 추천합니다.***
<br><br><br><br>
<img width="725" alt="image" src="https://github.com/user-attachments/assets/d9b848f1-d2f1-40c7-8009-04528887439d" />

모델이 병변을 분류할때 병변이미지의 구도, 각도, 배경이 아닌 병변 정의에 맞게 병변을 탐지하고 잘 분류했음을 확인할 수 있습니다.


