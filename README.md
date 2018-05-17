## Neural Chatbot

* to Run

chat 기본 데이타 위치 /tmp/nmt_chat
train 결과가 저장되는 곳이 /tmp/chat_model


```bash
python nmt.py \
    --attention=scaled_luong \
    --src=req --tgt=rep \
    --vocab_prefix=/tmp/nmt_chat/vocab  \
    --train_prefix=/tmp/nmt_chat/train \
    --dev_prefix=/tmp/nmt_chat/test  \
    --test_prefix=/tmp/nmt_chat/test \
    --out_dir=/tmp/chat_model \
    --num_train_steps=12000 \
    --steps_per_stats=100 \
    --num_layers=8 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```

```bash
python nmt.py \
    --out_dir=/tmp/chat_model \
    --inference_input_file=/tmp/my_infer_file.vi \
    --inference_output_file=/tmp/chat_model/output_infer
```
