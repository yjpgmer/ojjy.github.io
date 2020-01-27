---
layout: post
title: Prediction of heart disease
subtitle: Prediction by using TensorFlow 2.0
bigimg: /img/path.jpg
tags: [heart, disease]
---

## Prediction of heart disease
Prediction heart disease based on TensorFlow2.0 DNN from patients data
Needed analsis of interaction between complex data (Floowing example is simple plot)
Steps
 1) Data preparation
 2) Look up patients data
 3) Data preprocess
 4) Generate Neural Network model
 5) Model Training
 6) Prediction Heart disease
 
 
- Data Preparation
Download data -> 
heart.csv - 14 attributes, 303 patients data
```r
# Library loading
!pip install tensorflow-gpu==2.0.0-alpha0
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from google.colab import drive
from sklearn.model_selection import train_test_split

%matplotlib inline
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
```

```r
# mount and file path
drive.mount("/content/drive")
heart_csv_path = "/content/drive/My Drive/Colab Notebooks/tensorflow-2/data/heart.csv"
```

```r
# loading and print
data = pd.read_csv(heart_csv_path)
data.describe()
#age -</b> 나이
#sex -</b> (1 = 남성; 0 = 여성)
#cp -</b> 가슴 통증 유형(0, 1, 2, 3, 4)
#trestbps -</b> 안정 혈압(병원 입원시 mm Hg)
#chol -</b> 혈청 콜레스테롤(mg/dl)
#fbs -</b> (공복 혈당 &gt; 120 mg/dl)(1 = true; 0 = false)
#restecg -</b> 안정 심전도 결과(0, 1, 2)
#thalach -</b> 최대 심박동수
#exang -</b> 협심증 유발 운동(1 = yes; 0 = no)
#oldpeak -</b> 비교적 안정되기까지 운동으로 유발되는 ST depression
#slope -</b> 최대 운동 ST segment의 기울기
#ca -</b> 형광 투시된 주요 혈관의 수(0-3)
#thal -</b> (3 = 보통; 6 = 해결된 결함; 7 = 해결가능한 결함)
```

```r
# check data
data.shape
(303, 14)
```

```r
# check values
data.column
>Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'],
      dtype='object')
```
- Look up patients data
Understanding and visualization of data
```r
source_code
```

- Data preprocess
preprocess for modeling to numbers
```r
source code
```

- Generate Neural Network model
```r
source code
```

- Model Training
```r
source code
```

- Prediction heart disease
```r
source code
```


# Reference
