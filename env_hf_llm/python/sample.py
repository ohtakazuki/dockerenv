import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

# (任意) .env ファイルから環境変数をロード
load_dotenv()

# --- 設定 ---
# 使用するモデル名を指定 (比較的小さな CPU 向けモデルを選択)
# 例: 'gpt2', 'elyza/ELYZA-japanese-Llama-2-7b-instruct', 'rinna/japanese-gpt2-medium'
# ※ モデルサイズが大きいと CPU では非常に遅くなります
model_name = "gpt2"
# Hugging Face Hub トークン (プライベートモデル等で必要)
hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
# キャッシュディレクトリ (Dockerfile/Compose と合わせる)
cache_dir = os.environ.get("HF_HOME", "/home/appuser/.cache/huggingface")

print(f"Using model: {model_name}")
print(f"Cache directory: {cache_dir}")
print(f"PyTorch version: {torch.__version__}")

# --- CPU/GPU 確認 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    # CPU コア数を取得 (参考情報)
    num_cores = os.cpu_count()
    print(f"Number of CPU cores available: {num_cores}")
    # torch.set_num_threads(max(1, num_cores // 2)) # 実行スレッド数を制限する場合

# --- モデルとトークナイザーのロード ---
print("Loading model and tokenizer...")
start_time = time.time()
try:
    # トークナイザーのロード
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=os.path.join(cache_dir, "transformers")
    )
    # モデルのロード
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=os.path.join(cache_dir, "transformers"),
        device_map=device
    )
    # モデルを評価モードに設定
    model.eval()

    load_time = time.time() - start_time
    print(f"Model and tokenizer loaded in {load_time:.2f} seconds.")

    # --- テキスト生成パイプラインの利用 (簡単な方法) ---
    print("\n--- Using text-generation pipeline ---")
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )
    prompt = "Once upon a time,"
    start_time = time.time()
    generated_texts = text_generator(prompt, max_length=50, num_return_sequences=1)
    gen_time = time.time() - start_time
    print(f"Generated text in {gen_time:.2f} seconds:")
    for text in generated_texts:
        print(text['generated_text'])

    # --- 手動でのテキスト生成 (より詳細な制御) ---
    print("\n--- Manual text generation ---")
    input_text = "日本の首都は"
    print(f"Input: {input_text}")
    # 入力をトークン化
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    start_time = time.time()
    # 推論実行 (勾配計算を無効化)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30, # 新しく生成するトークンの最大数
            temperature=0.1,   # 生成のランダム性 (低いほど決定的)
            # top_k=50,        # Top-K サンプリング
            # top_p=0.9,       # Top-P (Nucleus) サンプリング
            do_sample=True, # サンプリングを有効にするか (False だと貪欲法)
            pad_token_id=tokenizer.eos_token_id # パディング用トークン ID
        )
    gen_time = time.time() - start_time
    print(f"Generated output in {gen_time:.2f} seconds.")
    # 出力トークンをデコード
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output:\n{decoded_output}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

print("\nScript finished.")