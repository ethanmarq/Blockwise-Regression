#!/usr/bin/env sh

salloc --mem=256gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
dataset="news20"; load_logistic; logistic_solvers

salloc --mem=256gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
dataset="rcv1_train"; load_logistic; logistic_solvers
