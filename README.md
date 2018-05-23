## Neural Chatbot

    Neural Network Chatbot from NMT([Neural Machine Translation](https://github.com/tensorflow/nmt))
    
    Using seq2seq(RNN) 

* Running for Training

    To install this tutorial, you need to have TensorFlow installed on your system. This tutorial requires TensorFlow Nightly. To install TensorFlow, follow the [installation instructions here.](https://www.tensorflow.org/install/)

* Make training data

``` bash
    mkdir -p /tmp/nmt_chat
```
Just example!

make traing data for requesting

``` bash
    echo -e "안녕?\n넌 누구니?\n잘 지내?" > train.req
```

make traing data for replying

``` bash
    echo -e "안녕!\n난 코에이?\n응 잘 지내" > train.rep
```

make test data for requesting

``` bash
    echo -e "안녕?\n넌 누구니?" > test.req
```
    
make test data for replying

``` bash
    echo -e "안녕!\n난 코에이" > test.rep
```

make vocab file for training. train.all means a merged file from train.req & train.rep

``` bash
    python $PROJECT_ROOT/bin/generate_vocab < train.all > vocab.req
    cp vocab.req vocab.rep
```

copy all file to '/tmp/nmt_chat'

``` bash
    cp train.req train.rep test.req test.rep vocab.req vocab.rep /tmp/nmt_chat
```
    
* Do training

    Go to Project's ROOT directory and run training 
    
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
    --num_layers=4 \
    --num_units=128 \
    --dropout=0.2 \
    --metrics=bleu
```

* Test chatbot

```bash
python chat.py --out_dir=/tmp/chat_model
```

[한글 설명](http://crazia.tistory.com/ "블로그")
