#!/usr/bin/env python
# coding: utf-8

# # Predicting Thorax Diseases Using the ChestX-Ray14 Dataset and Convolutional Techniques

# ## Introduction

# Dataset provided by the National Institute of Health at: https://nihcc.app.box.com/v/ChestXray-NIHCC
# 
# *Random subset provided [here](https://www.kaggle.com/nih-chest-xrays/sample)*

# ## Setup

# In[1]:


import glob
import gzip
import os
import tarfile
import time
import warnings
from urllib.request import urlretrieve

import pandas as pd

import keras
from keras.applications import DenseNet121, ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.metrics import AUC
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

import efficientnet.keras as efn


# In[2]:


import tensorflow as tf
tf.test.is_gpu_available()


# In[3]:


ROOT_DIR = '.'
DATA_PATH = '/data'
CHECKPOINT_PATH = '/models'

SAMPLE_RATE = 1.00
EPOCHS = 50
BATCH_SIZE = 32
CHECKPOINT_RATE = 2

CLASSES = [
  'Hernia',
  'Pneumonia',
  'Fibrosis',
  'Edema',
  'Emphysema',
  'Cardiomegaly',
  'Pleural_Thickening',
  'Consolidation',
  'Pneumothorax',
  'Mass',
  'Nodule',
  'Atelectasis',
  'Effusion',
  'Infiltration'
]


# ## Data Loading

# In[4]:


def batch_download_and_extract(path='.', first_n=None):

    # URLs for zip files containing ChestX-ray14 dataset from NIH
    links = [
        'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
        'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
        'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
        'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
        'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
        'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
        'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
        'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
        'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
        'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
        'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
        'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
    ]

    first_n = first_n or len(links)

    if first_n > len(links):
        raise('Number of files requested exceeds amount available')

    for idx, link in enumerate(links[:first_n]):
        fn = 'images_{:03d}.tar.gz'.format(idx+1)
        print('downloading', fn, '...')
        urlretrieve(link, fn)  # download the zip file

        tar = tarfile.open(fn, "r:gz")
        tar.extractall(path + '/images_{:03d}'.format(idx+1))
        tar.close()

        os.remove(fn)  # Remove remaining .tar file

    labels_url = 'https://nihcc.app.box.com/index.php?rm=box_download_shared_file&vanity_name=ChestXray-NIHCC&file_id=f_219760887468'
    urlretrieve(labels_url, path + '/Data_Entry_2017.csv')
  
    print("Download complete. Please check the checksums")


# In[5]:


full_dir = "{0}{1}/full".format(ROOT_DIR, DATA_PATH)

if not os.path.isdir(full_dir):
    print('Data not present -- downloading now ...')
    os.makedirs(full_dir)
    batch_download_and_extract(full_dir)
else:
    print('Data directory already exists')


# In[6]:


df = pd.read_csv("{}/Data_Entry_2017.csv".format(full_dir))
df.head()


# In[7]:


df['Finding Labels'] = df['Finding Labels'].apply(lambda s: s.split('|'))
df.head()


# In[8]:


# https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list

mlb = MultiLabelBinarizer()
df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('Finding Labels')),
                          columns=mlb.classes_,
                          index=df.index))
df.head()


# In[9]:


df = df[['Image Index'] + CLASSES]
df.head()


# In[10]:


full_dir


# In[11]:


datagen_file = 'labels.csv'

if os.path.isfile(full_dir + '/' + datagen_file):
    df = df.read_csv(full_dir + '/' + datagen_file)
else:
    img_paths =  glob.glob(full_dir + '/**/*.png', recursive=True)
    df['Image Index'] = df['Image Index'].apply(lambda x: next(p for p in img_paths if x in p))
    df.to_csv(full_dir + '/' + datagen_file, index=False)
    df.head()


# In[12]:


df = df.sample(frac=SAMPLE_RATE)


# ## Data Preparation

# The ChestX-ray14 dataset is too large to fit entirely in memory when training; therefore, it's incrementally loaded via generator to reduce memory overhead. This is achieved using the Keras [Image Proprocessing](https://keras.io/preprocessing/image/) submodule.

# In[13]:


# https://datascience.stackexchange.com/a/17445/91316

train_df, test_df = train_test_split(df, test_size=0.2)

print('Training/Validation Samples:  {}'.format(len(train_df)))
print('Test Samples:  {}'.format(len(test_df)))


# In[14]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25
)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[15]:


train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='Image Index',
    y_col=CLASSES,
    subset='training',
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='raw',
    #classes=CLASSES,
    target_size=(224, 224)
)


# In[16]:


valid_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='Image Index',
    y_col=CLASSES,
    subset='validation',
    batch_size=BATCH_SIZE,
    shuffle=True,
    class_mode='raw',
    #classes=[],
    target_size=(224, 224)
)


# In[17]:


test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='Image Index',
    y_col=CLASSES,
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='raw',
    #classes=[],
    target_size=(224, 224)
)


# ## Modeling

# In[18]:


ROOT_DIR + CHECKPOINT_PATH


# In[19]:


if not os.path.exists(ROOT_DIR + CHECKPOINT_PATH):
    os.makedirs(ROOT_DIR + CHECKPOINT_PATH)


# In[20]:


class TimeHistory(keras.callbacks.Callback):
    """Object used on keras callbacks to measure epoch training time

    Args:
        None

    Params:
        time (list): collection of times in seconds for each epoch's training

    """

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# Three models will be implemented and their results compared:
# 
# 1.   ResNet
# 2.   DenseNet
# 3.   EfficientNet
# 

# ### ResNet

# A pre-built ResNet model from the Keras library is used. Documentation on the model can be found [here](https://keras.io/applications/). Pre-trained weights from the ImageNet dataset are used.

# In[40]:


resnet_base = ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)
output = Dense(14, activation='sigmoid')(resnet_base.output)

resnet = Model(input=resnet_base.input, outputs=output)


# In[41]:


resnet.summary()


# In[42]:


resnet.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)


# In[43]:


resnet_time = TimeHistory()
resnet_stopping = EarlyStopping(patience=5, restore_best_weights=True)
resnet_checkpoint = ModelCheckpoint(filepath=full_dir + CHECKPOINT_PATH + '/resnet-best.hdf5', 
                                    save_best_only=True)

resnet_history = resnet.fit_generator(
    generator=train_generator,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=valid_generator,
    callbacks=[resnet_time, resnet_stopping, resnet_checkpoint]
)


# ### DenseNet

# In[28]:


densenet_base = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)
output = Dense(15, activation='sigmoid')(densenet_base.output)

densenet = Model(input=densenet_base.input, outputs=output)


# In[29]:


densenet.summary()


# In[30]:


densenet.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)


# In[ ]:


densenet_time = TimeHistory()
densenet_stopping = EarlyStopping(patience=5, restore_best_weights=True)
densenet_checkpoint = ModelCheckpoint(filepath=full_dir + CHECKPOINT_PATH + '/densenet-best.hdf5', 
                                      save_best_only=True)

densenet.fit_generator(
    generator=train_generator,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=valid_generator,
    callbacks=[densenet_time, densenet_stopping, densenet_checkpoint]
)


# ### EfficientNet

# EfficientNet is a lightweight CNN architecture that is designed to require significantly less compute than other state of the art architectures on common transfer learning datasets.

# Pre-built EfficientNet models built in Keras are used from the efficientnet library available on [GitHub](https://github.com/qubvel/efficientnet) and installable via PyPI.

# In[24]:


efficientnet_base = efn.EfficientNetB4(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling='avg'
)
output = Dense(15, activation='sigmoid')(efficientnet_base.output)

efficientnet = Model(input=efficientnet_base.input, outputs=output)


# In[25]:


efficientnet.summary()


# In[26]:


efficientnet.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)


# In[27]:


efficientnet_time = TimeHistory()
efficientnet_stopping = EarlyStopping(patience=5, restore_best_weights=True)
efficientnet_checkpoint = ModelCheckpoint(filepath=full_dir + CHECKPOINT_PATH + '/efficientnet-best.hdf5', 
                                          save_best_only=True)

efficientnet_history = efficientnet.fit_generator(
    generator=train_generator,
    epochs=EPOCHS,
    shuffle=True,
    validation_data=valid_generator,
    callbacks=[efficientnet_time, efficientnet_stopping, efficientnet_checkpoint]
)


# ## Results

# ### ResNet

# In[37]:


resnet = load_model(full_dir + CHECKPOINT_PATH + '/resnet-best.hdf5', 
                    compile=False)

resnet_pred = resnet.predict_generator(
    generator=test_generator,
    verbose=1
)


# In[38]:


for idx, cls in enumerate(CLASSES):
    print('{} AUC:  '.format(cls), roc_auc_score(test_df[cls], resnet_pred[:,idx]))


# ### DenseNet

# In[39]:


densenet = load_model(full_dir + CHECKPOINT_PATH + '/densenet-best.hdf5', compile=False)

densenet_pred = densenet.predict_generator(
    generator=test_generator,
    verbose=1
)


# In[40]:


for idx, cls in enumerate(CLASSES):
    print('{} AUC:  '.format(cls), roc_auc_score(test_df[cls], densenet_pred[:,idx]))


# ### EfficientNet

# In[41]:


efficientnet = load_model(full_dir + CHECKPOINT_PATH + '/efficientnet-best.hdf5', compile=False)

efficientnet_pred = efficientnet.predict_generator(
    generator=test_generator,
    verbose=1
)


# In[42]:


for idx, cls in enumerate(CLASSES):
    print('{} AUC:  '.format(cls), roc_auc_score(test_df[cls], efficientnet_pred[:,idx]))


# In[ ]:




