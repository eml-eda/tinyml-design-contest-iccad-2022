#!/usr/bin/env bash

mkdir -p data
cd data
gdown --fuzzy https://drive.google.com/file/d/11lOiocENt7TVRqwYbUkIgd1_-ZCUNIrH/view?usp=sharing
# tar -xvzf tinyml_contest_data_training.rar
unrar x tinyml_contest_data_training.rar
rm tinyml_contest_data_training.rar
mv tinyml_contest_data_training/* .
rm -rf tinyml_contest_data_training
cd ..