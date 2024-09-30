#!/bin/bash         

nohup python predict_cil.py > nohup_cil5.out &
wait
nohup python pred2label_cil5.py >> nohup_cil5.out &

