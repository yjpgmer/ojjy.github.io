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
 
 
# Data Preparation
Download data -> https://www.kaggle.com/ronitf/heart-disease-uci
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

# Look up patients data
Understanding and visualization of data
```r
# Check heart disease in 303 patients
f = sns.countplot(x='target', data=data)
f.set_title("Heart disease presence distribution")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
```

```r
# Check heart disease in 303 patients dividing female and male
f = sns.countplot(x='target', data=data, hue='sex')
plt.legend(['Female', 'Male'])
f.set_title("Heart disease presence by gender")
f.set_xticklabels(['No Heart disease', 'Heart Disease'])
plt.xlabel("")
```

```r
# Implementation interaction among feature using heatmap
# The closer to +1, positive correlation
# The closer to -1, negative correlation
heat_map = sns.heatmap(data.corr(method='pearson'), annot=True, fmt='.2f', linewidths=2)
heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=45)
```

# Data preprocess
preprocess for modeling to numbers
```r
# feater column is bridge between raw data and modeling data
feature_columns = []

# pass on Numeric col intactly
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
  feature_columns.append(tf.feature_column.numeric_column(header))

# transform categorical type from numerical data from Bucketized column
age = tf.feature_column.numeric_column("age")
age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# mapping string values from categorical column
data["thal"] = data["thal"].apply(str)
thal = tf.feature_column.categorical_column_with_vocabulary_list(
      'thal', ['3', '6', '7'])
thal_one_hot = tf.feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

data["sex"] = data["sex"].apply(str)
sex = tf.feature_column.categorical_column_with_vocabulary_list(
      'sex', ['0', '1'])
sex_one_hot = tf.feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

data["cp"] = data["cp"].apply(str)
cp = tf.feature_column.categorical_column_with_vocabulary_list(
      'cp', ['0', '1', '2', '3'])
cp_one_hot = tf.feature_column.indicator_column(cp)
feature_columns.append(cp_one_hot)

data["slope"] = data["slope"].apply(str)
slope = tf.feature_column.categorical_column_with_vocabulary_list(
      'slope', ['0', '1', '2'])
slope_one_hot = tf.feature_column.indicator_column(slope)
feature_columns.append(slope_one_hot)


# using Embedding column for multi values
thal_embedding = tf.feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# Crossed column - connection various features
age_thal_crossed = tf.feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
age_thal_crossed = tf.feature_column.indicator_column(age_thal_crossed)
feature_columns.append(age_thal_crossed)

cp_slope_crossed = tf.feature_column.crossed_column([cp, slope], hash_bucket_size=1000)
cp_slope_crossed = tf.feature_column.indicator_column(cp_slope_crossed)
```

```r
# Pandas dataframe - Tensorflow dataset
def create_dataset(dataframe, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  return tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)) \
          .shuffle(buffer_size=len(dataframe)) \
          .batch(batch_size)
```

```r
seperating training set and test set from data
train, test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)
train_ds = create_dataset(train)
test_ds = create_dataset(test)
```

# Generate Neural Network model
```r
Disposition dropout layer among dense layers in order to reduce overfeating
model = tf.keras.models.Sequential([
  tf.keras.layers.DenseFeatures(feature_columns=feature_columns),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dropout(rate=0.2),
  tf.keras.layers.Dense(units=128, activation='relu'),
  tf.keras.layers.Dense(units=1, activation='sigmoid')
])
```

# Model Training
```r
# print precision and loss after compling model 
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=test_ds, epochs=100, use_multiprocessing=True)
```

```r
# evaluate test set precision using model.evaludate function
>model.evaluate(test_ds)
2/2 [==============================] - 0s 23ms/step - loss: 0.3431 - accuracy: 0.8852
[0.3430721387267113, 0.8852459]
```

```r
# Visualization model precision
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim((0, 1))
plt.legend(['train', 'test'], loc='upper left')
```

```r
# Visualization model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

# Prediction heart disease
```r
from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict(test_ds)
bin_predictions = tf.round(predictions).numpy().flatten()
print(classification_report(y_test.values, bin_predictions))
```

```r
# visualization result of confusion matrix
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)

sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="Blues",fmt="d",cbar=False)
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```

# Reference
https://towardsdatascience.com/heart-disease-prediction-in-tensorflow-2-tensorflow-for-hackers-part-ii-378eef0400ee
https://colab.research.google.com/drive/13EThgYKSRwGBJJn_8iAvg-QWUWjCufB1
https://www.kaggle.com/ronitf/heart-disease-uci
https://www.tensorflow.org/tutorials/structured_data/feature_columns
https://locslab.github.io/Tensorflow-feature-columns(1)/
