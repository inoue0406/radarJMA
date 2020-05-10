#!/bin/bash

python ../src_prep/prep_hrncst_JMA.py 2017-01 >& log-01 &
python ../src_prep/prep_hrncst_JMA.py 2017-02 >& log-02 &
python ../src_prep/prep_hrncst_JMA.py 2017-03 >& log-03 &
python ../src_prep/prep_hrncst_JMA.py 2017-04 >& log-04 &
python ../src_prep/prep_hrncst_JMA.py 2017-05 >& log-05 &
python ../src_prep/prep_hrncst_JMA.py 2017-06 >& log-06 &

