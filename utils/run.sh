#!/bin/bash

echo "5 files in total"
filedir=../data/preprocessed/trainset/zhidao.train.

for i in {1..3}; do

filepath=${filedir}"$i".json
echo "start deal with   ${filepath} ..."
python preprocess_yhl.py recall f1 a --train_files $filepath -n 8

done
