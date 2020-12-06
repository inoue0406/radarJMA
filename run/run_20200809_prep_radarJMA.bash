#!/bin/bash

python ../src_prep/prep_radarJMA.py 2015 ../data/data_kanto_aug/ >& log-01 &
python ../src_prep/prep_radarJMA.py 2016 ../data/data_kanto_aug/ >& log-02 &
python ../src_prep/prep_radarJMA.py 2017 ../data/data_kanto_aug/ >& log-03 &
