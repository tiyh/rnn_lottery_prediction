# rnn_lottery_prediction
Lottery Prediction using TensorFlow  and LSTM RNN

## Train:

python rnn_lottery.py --data-path=/home/chris/workspace/rnn_lottery/data [--epoch= --batch-size=  --no-use-cudnn-rnn ...]

## Predict:

python rnn_lottery.py  --predictpath=  --logdir=  [--epoch= --batch-size=  --no-use-cudnn-rnn ...]

## Tensorboard:

tensorboard --logdir=/home/chris/workspace/rnn_lottery/savedmodel

http://localhost:6006


## Get train-data

1. run spider

    python tools/spider_rnn_lottery_data.py

2. get txt from Mysql database

    select numbers from rnnlottery  into outfile '/var/lib/mysql-files/source.txt';

3. cut source.txt,get train.txt and valid.txt (default 4:1)

    python tools/cutfile.py

## My development environment

    Ubuntu 18.04 

    tensorflow 1.10.0
    
    Python 2.7
