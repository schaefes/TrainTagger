#!/bin/bash
if [[ "$1" == "False" ]]; then
    python tagger/train/train.py -n $Name -p 50 
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model 
python tagger/train/train.py --plot-basic -n $Name



    
    
  

  
