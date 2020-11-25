#!/usr/bin/env bash

source venv35/bin/activate

python3 process_ud_data.py \
--train "./input/fr-ud-train.conllu" \
--dev "./input/fr-ud-dev.conllu" \
--test "./input/fr-ud-test.conllu" \
--output "./output/fr_ud14_pproc" \
--labratio 0.2 \
--unlabratio 0.5 \

python3 vsl_gg.py \
--debug 1 \
--model hier \
--random_seed 0 \
--data_file "./output/fr_ud14_pproc.ud" \
--vocab_file "./output/fr_ud14_vocab" \
--tag_file "./output/ud_tagfile" \
--embed_file "./input/fr.bin" \
--use_unlabel 1 \
--prior_file ".output/test_gg_hier" \
--embed_type ud \
--embed_dim 100 \
--rnn_type gru \
--tie_weights 1 \
--char_embed_dim 50 \
--char_hidden_size 100 \
--latent_z_size 50 \
--latent_y_size 25 \
--rnn_size 100 \
--mlp_hidden_size 100 \
--mlp_layer 2 \
--kl_anneal_rate 1e-4 \
--unlabel_ratio 0.1 \
--update_freq_label 1 \
--update_freq_unlabel 1 \
--opt adam \
--n_iter 30000 \
--batch_size 10 \
--unlabel_batch_size 10 \
--vocab_size 100000 \
--char_vocab_size 300 \
--train_emb 0 \
--save_prior 1 \
--learning_rate 1e-3 \
--l2 0 \
--grad_clip 10. \
--f1_score 0 \
--print_every 5000 \
--eval_every 10000 \
--summarize 1

rm -rf ./output/*
rm events*

deactivate