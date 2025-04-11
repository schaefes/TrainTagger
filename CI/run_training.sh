#!/bin/bash
if [[ "$1" == "False" ]]; then
    python tagger/train/train.py -n $Name -p 50 
    python tagger/train/train.py --plot-basic -n $Name
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model 
    cd output/baseline
    eos mkdir -p ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
    eos cp model/saved_model.h5 ${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model/saved_model.h5 .
    export MODEL_LOCATION=${EOS_STORAGE_DIR}/${EOS_STORAGE_SUBDIR}/model
else
    mkdir output/baseline/model
    eos cp ${MODEL_LOCATION}/saved_model.h5 output/baseline/model
    eos cp ${MODEL_LOCATION}/extras/* output/baseline/
    python tagger/train/train.py --plot-basic -n $Name
fi 



    
    
  

  
