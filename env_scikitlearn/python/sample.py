import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 機械学習のモデル
from sklearn.svm import SVC
# データの分割
from sklearn.model_selection import train_test_split
# 評価
from sklearn.metrics import classification_report

# ----------------------------------------
# ログ出力の定義
# ----------------------------------------
import os.path
import logging
import logging.config
from logging import getLogger

if os.path.isdir("out") == False:
    os.mkdir("out")

logging.config.fileConfig("logging.conf")
logger = getLogger(__name__)

# ----------------------------------------
# データの読み込み
# ----------------------------------------
from sklearn.datasets import load_iris
iris = load_iris()

logger.info(f'データの形状:{iris["data"].shape}')

# datasetの作成
_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
dataset = pd.DataFrame(data = iris['data'], columns = _names)
dataset['species'] = iris['target']
logger.info('\n' + dataset.head().to_string() )

# ----------------------------------------
# データの分割
# ----------------------------------------
# Y:目的変数、X:説明変数
Y = np.array(dataset['species'])
X = np.array(dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])

# データの分割
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=0)

# ----------------------------------------
# 学習
# ----------------------------------------
# SVMによる学習
svm_model = SVC()
svm_model.fit(X_train, Y_train)
Y_pred = svm_model.predict(X_valid)

# ----------------------------------------
# モデルの評価
# ----------------------------------------
logger.info('\n' + classification_report(Y_valid, Y_pred))