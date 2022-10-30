import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
# 学習済モデル取得用
import tensorflow_hub as hub
# 処理時間計測用
import time
# ログファイル名
import datetime

# 全体の処理時間計測用
start = time.perf_counter()

# --------------------
# ログ出力の定義
# --------------------
import os.path
import logging
import logging.config
from logging import getLogger

if os.path.isdir("out") == False:
    os.mkdir("out")

logging.config.fileConfig("logging.conf")
logger = getLogger(__name__)

# --------------------
# データセットの準備
# --------------------
# TensorFlow flowers データセットを使用
data_root = tf.keras.utils.get_file(
  'flower_photos',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

logger.info(f'data_root:{data_root}')

batch_size = 32
img_height = 224
img_width = 224

# 学習データの取得
train_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# 検証データの取得
val_ds = tf.keras.utils.image_dataset_from_directory(
  str(data_root),
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size
)

# flowers データセットには 5 つのクラスがある
class_names = np.array(train_ds.class_names)
print(class_names)

# 値を0~1の範囲に正規化
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y)) # Where x—images, y—labels.

# 効率的な入力パイプラインの作成
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

for image_batch, labels_batch in train_ds:
  logger.info(f'Image batch shape: {image_batch.shape}')
  logger.info(f'Label batch shape: {labels_batch.shape}')
  break

# --------------------
# 事前学習済モデルの準備
# --------------------
# ヘッドレスモデルのダウンロード
mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"
feature_extractor_model = mobilenet_v2

# 事前学習済みモデルを Keras レイヤーとしてラップし、特徴量抽出器を作成
feature_extractor_layer = hub.KerasLayer(
    feature_extractor_model,
    input_shape=(224, 224, 3),
    trainable=False)

# 分類用に全結合層を追加
num_classes = len(class_names)

model = tf.keras.Sequential([
  feature_extractor_layer,
  tf.keras.layers.Dense(num_classes)
])
model.summary(print_fn=logger.info)

# --------------------
# モデルのトレーニング
# --------------------
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

log_dir = "./out/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1) # Enable histogram computation for every epoch.

NUM_EPOCHS = 10

history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=NUM_EPOCHS,
                    callbacks=tensorboard_callback)

# --------------------
# 予測
# --------------------
# 予測を行いクラス番号のリストを取得
predicted_batch = model.predict(image_batch)
predicted_id = tf.math.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]
print(predicted_label_batch)

# イメージとラベルを保存
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")
plt.savefig('./out/predict.png')

# --------------------
# 学習済モデルの保存と再利用
# --------------------
# モデルのエクスポート
t = time.time()

export_path = "./out/saved_models/{}".format(int(t))
model.save(export_path)

export_path

# SavedModel を再読み込みできることと、モデルが同じ結果を出力できることを確認
reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()

# 予測を行いクラス番号のリストを取得
reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
print(reloaded_predicted_label_batch)

# イメージとラベルを保存
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(reloaded_predicted_label_batch[n].title())
  plt.axis('off')
_ = plt.suptitle("Model predictions")
plt.savefig('./out/predict2.png')

# 全体の処理の実行時間を出力
logger.info(f'total time: {time.perf_counter() - start}')
# 1回目 275.6627539410001
# 2回目 292.40264047100027
