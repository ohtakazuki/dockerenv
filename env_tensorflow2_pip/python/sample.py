# -*- coding: utf-8 -*-
# @title ライセンス情報 (Apache License 2.0)
# Copyright 2018 The TensorFlow Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import os
import logging
import logging.config
from logging import getLogger
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import pathlib

# --------------------
# 定数定義
# --------------------
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 10 # エポック数
OUTPUT_DIR = "out"
LOG_CONF_FILE = "logging.conf"
CACHE_DIR = '.' # データセットのキャッシュ先

# 特徴抽出器モデルのURL (MobileNetV2を使用)
FEATURE_EXTRACTOR_MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"


# --------------------
# ログ設定
# --------------------
# 出力ディレクトリを確認/作成
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR) # サブディレクトリも作成できるように makedirs に変更

# logging 設定ファイルを読み込み
try:
    logging.config.fileConfig(LOG_CONF_FILE)
    logger = getLogger(__name__)
    logger.info("ロギング設定ファイルを読み込みました。")
except FileNotFoundError:
    print(f"エラー: {LOG_CONF_FILE} が見つかりません。スクリプトと同じディレクトリに配置してください。")
    # logging.conf がない場合のフォールバック設定
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = getLogger(__name__)
    logger.warning(f"{LOG_CONF_FILE} が見つからないため、基本的なログ設定を使用します。")
except Exception as e:
    print(f"ロギング設定の読み込み中にエラーが発生しました: {e}")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = getLogger(__name__)
    logger.error(f"ロギング設定の読み込みエラー: {e}。基本的なログ設定を使用します。")

# --------------------
# メイン処理
# --------------------
def main():
    """メイン処理関数"""
    logger.info("スクリプト実行開始...")
    start_time = time.perf_counter() # 全体の処理時間計測開始

    # --- データセットの準備 ---
    logger.info("データセットの準備を開始します...")
    try:
        train_ds, val_ds, class_names = prepare_dataset()
        logger.info(f"クラス名: {class_names}")

        # データセットからサンプルバッチを取得 (形状確認と後続処理用)
        for image_batch, labels_batch in train_ds.take(1):
            logger.info(f"画像バッチの形状: {image_batch.shape}")
            logger.info(f"ラベルバッチの形状: {labels_batch.shape}")
            # 後で予測やプロットに使用するために保持
            sample_image_batch, sample_labels_batch = image_batch, labels_batch
            break # 1バッチ取得すれば十分
        logger.info("データセットの準備が完了しました。")

    except Exception as e:
        logger.error(f"データセット準備中にエラーが発生しました: {e}", exc_info=True)
        return # エラーが発生したら処理中断

    # --- モデルの構築 ---
    logger.info("モデルの構築を開始します...")
    try:
        model = build_model(len(class_names))
        # モデルのサマリーをログに出力
        model.summary(print_fn=logger.info)
        logger.info("モデルの構築が完了しました。")
    except Exception as e:
        logger.error(f"モデル構築中にエラーが発生しました: {e}", exc_info=True)
        return

    # --- モデルのトレーニング ---
    logger.info("モデルのトレーニングを開始します...")
    try:
        history = train_model(model, train_ds, val_ds)
        logger.info("モデルのトレーニングが完了しました。")
        logger.info(f"トレーニング履歴のキー: {history.history.keys()}")
    except Exception as e:
        logger.error(f"モデルのトレーニング中にエラーが発生しました: {e}", exc_info=True)
        return

    # --- トレーニング済みモデルでの予測と結果保存 ---
    logger.info("トレーニング済みモデルで予測を実行し、結果をプロット・保存します...")
    try:
        # 検証データセットからサンプルバッチを取得 (トレーニング中に使用したものとは別の場合がある)
        pred_image_batch, _ = next(iter(val_ds))
        predicted_labels = predict_and_plot(model, pred_image_batch, class_names, os.path.join(OUTPUT_DIR, "predictions.png"), "Model Predictions")
        logger.info(f"予測結果のサンプル (最初の5件): {predicted_labels[:5]}")
    except Exception as e:
        logger.error(f"トレーニング済みモデルでの予測中にエラーが発生しました: {e}", exc_info=True)
        # 予測エラーは致命的ではないかもしれないので、続行する

    # --- モデルのエクスポートと検証 ---
    logger.info("モデルのエクスポートと検証を開始します...")
    try:
        export_path = export_model(model)
        if export_path:
            reloaded_model = load_and_verify_model(export_path, pred_image_batch, class_names) # 同じバッチで検証
            if reloaded_model:
                logger.info("再読み込みしたモデルでの予測と結果保存を実行します...")
                predict_and_plot(reloaded_model, pred_image_batch, class_names, os.path.join(OUTPUT_DIR, "reloaded_predictions.png"), "再読み込みモデルの予測結果")
        logger.info("モデルのエクスポートと検証が完了しました。")
    except Exception as e:
        logger.error(f"モデルのエクスポートまたは検証中にエラーが発生しました: {e}", exc_info=True)

    # --- 終了処理 ---
    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f'全体の実行時間: {total_time:.2f} 秒')
    print(f'全体の実行時間: {total_time:.2f} 秒') # コンソールにも出力

# --------------------
# ヘルパー関数
# --------------------

def prepare_dataset():
    """フラワーデータセットをダウンロードし、tf.data.Dataset を準備する"""
    logger.info("フラワーデータセットをダウンロードします...")
    data_file = tf.keras.utils.get_file(
      'flower_photos.tgz',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      cache_dir=CACHE_DIR, # カレントディレクトリにキャッシュ
       extract=True)
    data_root = pathlib.Path(data_file).with_suffix('')
    logger.info(f"データセットを展開しました: {data_root}")

    logger.info("画像データセットを作成します (トレーニング用・検証用分割)...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
      str(data_root),
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=IMAGE_SHAPE,
      batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
      str(data_root),
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=IMAGE_SHAPE,
      batch_size=BATCH_SIZE
    )

    class_names = np.array(train_ds.class_names)

    logger.info("データセットの前処理 (正規化、キャッシュ、プリフェッチ) を適用します...")
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

def build_model(num_classes):
    """事前学習済みモデルをベースに、新しい分類ヘッドを持つ Keras モデルを構築する"""
    logger.info(f"特徴抽出器レイヤーを作成します (モデル: {FEATURE_EXTRACTOR_MODEL_URL})...")
    
    # 低レベルのTensorFlow APIを使用してモデルを構築
    inputs = tf.keras.Input(shape=IMAGE_SHAPE + (3,))
    
    # 入力テンソルを具体的なテンソルに変換
    x = tf.keras.layers.Lambda(lambda x: tf.convert_to_tensor(x))(inputs)
    
    # tf.functionでラップしたカスタム層を作成
    @tf.function
    def apply_feature_extractor(x):
        feature_extractor = hub.load(FEATURE_EXTRACTOR_MODEL_URL)
        return feature_extractor(x)
    
    # カスタム層を通して特徴ベクトルを取得
    features = tf.keras.layers.Lambda(lambda x: apply_feature_extractor(x))(x)
    
    # 分類ヘッドを追加
    outputs = tf.keras.layers.Dense(num_classes)(features)
    
    # モデルを作成
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    logger.info("モデルの構築が完了しました...")
    return model

def train_model(model, train_ds, val_ds):
    """モデルをコンパイルし、指定されたエポック数でトレーニングする"""
    logger.info("モデルをコンパイルします...")
    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # 最終層が活性化関数なしのため True
      metrics=['accuracy']) # 'acc' ではなく 'accuracy' が一般的

    # TensorBoard コールバックの設定
    log_dir = os.path.join(OUTPUT_DIR, "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger.info(f"TensorBoard ログディレクトリ: {log_dir}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1) # エポックごとにヒストグラムを計算

    logger.info(f"{NUM_EPOCHS} エポックのトレーニングを開始します...")
    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=NUM_EPOCHS,
                        callbacks=[tensorboard_callback])
    return history

def predict_and_plot(model, image_batch, class_names, save_path, title):
    """モデルで予測を行い、結果を Matplotlib でプロットしてファイルに保存する"""
    logger.info(f"画像バッチで予測を実行します...")
    predicted_batch = model.predict(image_batch)
    predicted_ids = tf.math.argmax(predicted_batch, axis=-1)
    predicted_labels = class_names[predicted_ids]

    logger.info(f"予測結果を '{save_path}' にプロット・保存します...")
    plt.figure(figsize=(10, 9))
    plt.subplots_adjust(hspace=0.5)
    num_images_to_plot = min(len(image_batch), 30) # 最大30枚プロット

    for n in range(num_images_to_plot):
        plt.subplot(6, 5, n + 1)
        # Rescaling レイヤーの影響を元に戻して表示 (オプション)
        # img_display = image_batch[n] * 255.0
        # plt.imshow(tf.cast(img_display, tf.uint8))
        plt.imshow(image_batch[n]) # 正規化された状態のまま表示
        plt.title(predicted_labels[n].title())
        plt.axis('off')

    plt.suptitle(title)
    plt.savefig(save_path) # ファイルに保存
    plt.close() # メモリ解放のために Figure を閉じる
    logger.info("プロットの保存が完了しました。")
    return predicted_labels # 予測ラベル配列を返す

def export_model(model):
    """モデルを SavedModel 形式でエクスポートする"""
    try:
        timestamp = int(time.time())
        export_dir = os.path.join(OUTPUT_DIR, "saved_models", str(timestamp))
        logger.info(f"モデルを SavedModel 形式でエクスポートします: {export_dir}")
        model.save(export_dir, save_format='tf')
        logger.info("モデルのエクスポートが完了しました。")
        return export_dir
    except Exception as e:
        logger.error(f"モデルのエクスポート中にエラーが発生しました: {e}", exc_info=True)
        return None

def load_and_verify_model(export_path, image_batch_to_verify, class_names):
    """SavedModel を再読み込みし、元のモデルとの予測結果を比較検証する"""
    try:
        logger.info(f"エクスポートされたモデルを再読み込みします: {export_path}")
        reloaded_model = tf.keras.models.load_model(export_path)
        logger.info("モデルの再読み込みが完了しました。")

        logger.info("再読み込みモデルの予測結果を検証します...")
        # 元のモデルでの予測結果 (比較のため、メイン処理から渡すか再計算が必要)
        # この例では再計算せず、再読み込みモデルの予測のみ行う
        reloaded_result_batch = reloaded_model.predict(image_batch_to_verify)

        # 差分の比較 (オプション: 元のモデルの予測結果が必要)
        # original_result_batch = model.predict(image_batch_to_verify) # 仮に model がスコープ内にある場合
        # diff = abs(reloaded_result_batch - original_result_batch).max()
        # logger.info(f"元のモデルと再読み込みモデルの予測結果の最大絶対差分: {diff}")
        # if diff < 1e-6:
        #     logger.info("再読み込みモデルの予測は元のモデルと一致しました。")
        # else:
        #     logger.warning("再読み込みモデルの予測が元のモデルと大きく異なります。")

        reloaded_predicted_ids = tf.math.argmax(reloaded_result_batch, axis=-1)
        reloaded_predicted_labels = class_names[reloaded_predicted_ids]
        logger.info(f"再読み込みモデルによる予測結果のサンプル (最初の5件): {reloaded_predicted_labels[:5]}")
        logger.info("モデル検証が完了しました。")
        return reloaded_model
    except Exception as e:
        logger.error(f"モデルの再読み込みまたは検証中にエラーが発生しました: {e}", exc_info=True)
        return None

# --------------------
# スクリプト実行
# --------------------
if __name__ == "__main__":
    # 日本語環境での Matplotlib の設定 (オプション)
    try:
        # フォントが見つからない場合のエラーを避けるため、存在確認や代替フォント指定を推奨
        # 例: plt.rcParams['font.family'] = 'IPAexGothic' # 事前にインストールが必要
        # plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'sans-serif'] # 代替フォント
        pass # ここでは特定のフォント設定は行わない
    except Exception as e:
        logger.warning(f"Matplotlib の日本語フォント設定中にエラーが発生しました: {e}")

    main()