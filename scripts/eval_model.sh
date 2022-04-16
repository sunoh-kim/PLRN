#! /bin/bash

options=$1	# path to configuration file without ".yml"
m_type=$2	# model type
dataset=$3	# dataset type
gpu_id=$4	# ID of GPU

CUDA_VISIBLE_DEVICES=${gpu_id} python -m src.experiment.eval \
                     	--config pretrained_models/${dataset}/${m_type}/${options}.yml \
                     	--checkpoint pretrained_models/${dataset}/${m_type}/model.pkl \
                     	--method ${m_type} \
                     	--dataset ${dataset}