#!/bin/bash


bio_albert_1_1_large_pubmed_pmc_ckp=gs://cs93-capstone-project/bio_albert_models/bio-albert_1_1_large_pubmed_pmc/bioalbert1.1.1_output_model_2_model.ckpt-best
bio_albert_1_1_large_pubmed_pmc_config=gs://cs93-capstone-project/bio_albert_models/bio-albert_1_1_large_pubmed_pmc/config.json
bio_albert_1_1_large_pubmed_pmc_spm=gs://cs93-capstone-project/bio_albert_models/bio-albert_1_1_large_pubmed_pmc/30k-clean.model
bio_albert_1_1_large_pubmed_pmc_vocab=gs://cs93-capstone-project/bio_albert_models/bio-albert_1_1_large_pubmed_pmc/30k-clean.vocab
DATA_DIR=/home/jupyter/datasets/ChemProt
python -m albert.run_classifier \
  --data_dir=$DATA_DIR \
  --output_dir=gs://cs93-capstone-project//REData/outputs/chemprot/exp01 \
  --init_checkpoint=$bio_albert_1_1_large_pubmed_pmc_ckp \
  --albert_config_file=$bio_albert_1_1_large_pubmed_pmc_config \
  --spm_model_file=$bio_albert_1_1_large_pubmed_pmc_spm \
  --vocab_file=$bio_albert_1_1_large_pubmed_pmc_vocab \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --do_lower_case \
  --max_seq_length=128 \
  --task_name=chemprot \
  --warmup_step=1000 \
  --learning_rate=3e-5 \
  --train_step=10000 \
  --train_batch_size=32  \
  --num_train_epochs=10.0 \
  --save_checkpoints_steps=500 \
 # --tpu_name=alb \
 # --use_tpu=True

