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
# import tensorflow_hub as hub # <-- 削除
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
# FEATURE_EXTRACTOR_MODEL_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4" # <-- 削除

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
        # for image_batch, labels_batch in train_ds.take(1):
        #     logger.info(f"画像バッチの形状: {image_batch.shape}")
        #     logger.info(f"ラベルバッチの形状: {labels_batch.shape}")
        #     # 後で予測やプロットに使用するために保持
        #     # sample_image_batch, sample_labels_batch = image_batch, labels_batch
        #     break # 1バッチ取得すれば十分
        logger.info("データセットの準備が完了しました。")

    except Exception as e:
        logger.error(f"データセット準備中にエラーが発生しました: {e}", exc_info=True)
        return # エラーが発生したら処理中断

    # --- モデルの構築 ---
    logger.info("モデルの構築を開始します...")
    try:
        model = build_model(len(class_names))
        # モデルのサマリーをログに出力
        model.summary(print_fn=lambda x, **kwargs: logger.info(x))
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
        # 検証データセットからサンプルバッチを取得
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
                predict_and_plot(reloaded_model, pred_image_batch, class_names, os.path.join(OUTPUT_DIR, "reloaded_predictions.png"), "Reloaded Model Predictions")
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
    # 注意: MobileNetV2 は [-1, 1] の範囲の入力を期待することが多いですが、
    # ここでは元コードに合わせて [0, 1] に正規化します。
    # 必要であれば tf.keras.applications.mobilenet_v2.preprocess_input を使用してください。
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names

# --- ここから build_model 関数を修正 ---
def build_model(num_classes):
    """事前学習済みモデル(MobileNetV2)をベースに、新しい分類ヘッドを持つ Keras モデルを構築する"""
    logger.info(f"特徴抽出器として tf.keras.applications.MobileNetV2 を使用します (ImageNet weights)...")

    # 入力層を定義
    inputs = tf.keras.Input(shape=IMAGE_SHAPE + (3,))

    # MobileNetV2 ベースモデルをロード
    # include_top=False: ImageNet用の最終分類層を除外
    # weights='imagenet': ImageNetで事前学習された重みを使用
    # pooling='avg': Global Average Poolingを適用し、出力をベクトル化
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SHAPE + (3,),
        include_top=False,
        weights='imagenet',
        pooling='avg' # 特徴ベクトルを取得
    )

    # ベースモデルの重みをフリーズ (転移学習のため)
    base_model.trainable = False
    logger.info("MobileNetV2 ベースモデルの重みをフリーズしました。")

    # 入力層をベースモデルに接続
    # ベースモデルはトレーニングしないため training=False を指定
    x = base_model(inputs, training=False)

    # オプション: ベースモデルと最終層の間にドロップアウト層を追加 (過学習抑制)
    # x = tf.keras.layers.Dropout(0.2)(x) # 必要に応じてレートを調整

    # 新しい分類ヘッド (Dense レイヤー) を追加
    # 活性化関数は指定せず、損失関数側で from_logits=True を使用
    outputs = tf.keras.layers.Dense(num_classes)(x)

    # モデル全体を定義
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    logger.info("モデルの構築が完了しました。")
    return model
# --- build_model 関数の修正 ここまで ---

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
        # Rescaling レイヤーの影響を元に戻して表示する場合はコメント解除
        # img_display = image_batch[n] * 255.0
        # plt.imshow(tf.cast(img_display, tf.uint8))
        plt.imshow(image_batch[n]) # 正規化された状態のまま表示 ([0, 1] の範囲)
        plt.title(predicted_labels[n].title())
        plt.axis('off')

    plt.suptitle(title)
    plt.savefig(save_path) # ファイルに保存
    plt.close() # メモリ解放のために Figure を閉じる
    logger.info("プロットの保存が完了しました。")
    return predicted_labels # 予測ラベル配列を返す

def export_model(model):
    """モデルを Keras V3 ネイティブ形式 (.keras) でエクスポートする"""
    try:
        timestamp = int(time.time())
        export_filename = f"model_{timestamp}.keras"
        # 保存先ディレクトリ (saved_models) を指定
        save_dir = os.path.join(OUTPUT_DIR, "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        export_path = os.path.join(save_dir, export_filename)

        logger.info(f"モデルを Keras ネイティブ形式 (.keras) でエクスポートします: {export_path}")
        model.save(export_path)
        logger.info("モデルのエクスポートが完了しました。")
        return export_path
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
        reloaded_result_batch = reloaded_model.predict(image_batch_to_verify)

        # オプション: 元のモデルとの差分比較
        # この関数内で元の 'model' にアクセスできないため、
        # 差分比較が必要な場合は、元の予測結果も引数で渡す等の工夫が必要。
        # diff = abs(reloaded_result_batch - original_result_batch).max()
        # logger.info(f"予測結果の最大絶対差分: {diff}")

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
        # 必要に応じて日本語フォントを設定
        # 例: plt.rcParams['font.family'] = 'IPAexGothic'
        pass
    except Exception as e:
        logger.warning(f"Matplotlib の日本語フォント設定中にエラーが発生しました: {e}")

    main()