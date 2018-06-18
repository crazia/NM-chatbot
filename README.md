# Neural Chatbot

    Neural Network Chatbot from Google's Neural Machine Translation
    
    Using seq2seq(RNN) 

* Running for Training
    
    Check original google source [Neural Machine Translation](https://github.com/tensorflow/nmt/ "NMT").
    
    To install this tutorial, you need to have TensorFlow installed on your system. This tutorial requires TensorFlow Nightly. To install TensorFlow, follow the [installation instructions here.](https://www.tensorflow.org/install/)

## Training (django Version)

* Download Source to $PROJECT

```bash
    git clone git@github.com:crazia/NM-chatbot.git
```

* Install MeCab (한글 형태소 분석용)

```bash
    bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
``` 

* Install requirements

```bash
    cd NM-chatbot
    pip install -r requirements.txt    
```

* Make migration

```bash
    python manage.py migrate
```

* Create Super User for admin (id/password) 

```bash
    python manage.py createsuperuser
```

* Make initial data 

```bash
    python manage.py makedata --initial
```

* Run Server

```bash
    python manage.py runserver
```

* Go to Admin site

    go to [http://localhost:8000/admin](http://localhost:8000/admin "admin url")
    and login 

* Make chat data for training

    Go to [http://localhost:8000/admin/chat/chat/add/](http://localhost:8000/admin/chat/chat/add/ "add chat") 
    and input dialog and change the contents of 4'th listbox to '검토 완료' and save.
    
    Repeat saving dialog for several times for training.
    
    
* Make training data

```bash
    python manage.py makedata
```    

## Training (Console Version)
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
    python $PROJECT_ROOT/core/bin/generate_vocab.py < train.all > vocab.req
    cp vocab.req vocab.rep
```

copy all file to '/tmp/nmt_chat'

``` bash
    cp train.req train.rep test.req test.rep vocab.req vocab.rep /tmp/nmt_chat
```
    
* Do training

    Go to Project's ROOT directory and run training 
    
```bash
python -m core.nmt \
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

    Go to Project's ROOT directory and edit constants.py for *out_dir* and run
    
``` bash
export OUT_DIR='DIRECTORY FOR MODEL'
```

``` bash
python manage.py runserver
```


check [http://localhost:8000](http://localhost:8000 "chatbot url")


[한글 설명](https://goo.gl/4s4cy2 "Blog")

