# rnn_lottery_prediction
Lottery Prediction using TensorFlow  and LSTM RNN

## train:

python rnn_lottery.py --data-path=/home/chris/workspace/rnn_lottery/data [--epoch= --batch-size=  --no-use-cudnn-rnn ...]

## predict:

python rnn_lottery.py --data-path=/home/chris/workspace/rnn_lottery/data --predictpath=   [--epoch= --batch-size=  --no-use-cudnn-rnn ...]

## tensorboard:

tensorboard --logdir=/home/chris/workspace/rnn_lottery/savedmodel

http://localhost:6006
