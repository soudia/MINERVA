#!/usr/bin/env bash

data_input_dir="/mnt/datasets/public/ousmane/dqna/data/umls/"
vocab_dir="/mnt/datasets/public/ousmane/dqna/data/umls/vocab"
total_iterations=2000
path_length=2
hidden_size=50
embedding_size=50
batch_size=4
beta=0.05
Lambda=0.05
use_entity_embeddings=0
train_entity_embeddings=0
train_relation_embeddings=1
base_output_dir="output/umls/"
load_model=1
model_load_dir="/home/ousmane/MINERVA/saved_models/umls/model.ckpt"
nell_evaluation=0
