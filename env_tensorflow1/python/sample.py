import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
# 学習済モデル取得用
import tensorflow_hub as hub

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
# データの準備
# ----------------------------------------
# データをダウンロードしてinフォルダに格納
_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'

data_root = tf.keras.utils.get_file(
    'flower_photos', _url, cache_dir='./', cache_subdir='in', untar=True)

logger.info(f'data_root:{data_root}')

# データ読み込み用のジェネレーターを定義
IMAGE_SHAPE = (224, 224)

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

for image_batch, label_batch in image_data:
  logger.info(f'Image batch shape: {image_batch.shape}')
  logger.info(f'Label batch shape: {label_batch.shape}')
  break

# ----------------------------------------
# モデルの構築(転移学習)
# ----------------------------------------
# 学習済モデルの取得
_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}
base_layer = hub.KerasLayer(_url, input_shape=(224,224,3))
base_layer.trainable = False

# モデルの構築(全結合層の追加)
model = tf.keras.Sequential([
  base_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary(print_fn=logger.info)

model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])

# 独自で損失を取得するためのコールバック
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

batch_stats_callback = CollectBatchStats()

# 1エポックあたりステップ数
steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

# ----------------------------------------
# 学習
# ----------------------------------------
history = model.fit_generator(image_data, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])

logger.info('batch_losses:')
logger.info(batch_stats_callback.batch_losses)
logger.info('batch_acc:')
logger.info(batch_stats_callback.batch_acc)

# ----------------------------------------
# 結果確認
# ----------------------------------------
# 損失をグラフ化
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)
plt.savefig('./out/fig_loss.png')

# 正解率をグラフ化
plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)
plt.savefig('./out/fig_acc.png')

# ----------------------------------------
# 予測
# ----------------------------------------
class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
logger.info(f'class_names:{class_names}')

predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(label_batch, axis=-1)

# 予測結果(イメージ)を保存
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")
plt.savefig('./out/predict.png')

# ----------------------------------------
# モデルのエクスポート
# ----------------------------------------
import time
t = time.time()

export_path = "./out/{}".format(int(t))
model.save(export_path, save_format='tf')
logger.info(f'export_path:{export_path}')

# リロードできる。エクスポート前と同じ結果が得られる
reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

diff = abs(reloaded_result_batch - result_batch).max()

logger.info(f'diff: {diff}')