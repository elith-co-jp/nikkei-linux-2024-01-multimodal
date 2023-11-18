# LLaVAのファインチューニング
動作確認はDGX Station V100 で実施しています。
## 環境構築
LLaVAを動かすための環境を作成します。
```
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
pip install -e .
pip install -e ".[train]"
pip install xformers
cd ..
```

## ファインチューニング
以下のコマンドを実行することでファインチューニングを行えます。
```
sh finetune_task_lora.sh
```

ディレクトリ構造は以下を想定しています。
```
hogehoge
  ├─ nikkei-linux-2024-01-multimodal (このリポジトリのディレクトリ)
  └─ data (instructionデータ)
  ```

## 実行方法
Gradio Web UI でファインチューニングしたモデルを利用する際は、model workerとして以下を実行してください。  
その他の準備は、日経Linuxや[LLaVAリポジトリ](https://github.com/haotian-liu/LLaVA)を参考に実行してください。

```
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path ./checkpoints/llava-v1.5-13b-task-lora --model-base liuhaotian/llava-v1.5-13b
```