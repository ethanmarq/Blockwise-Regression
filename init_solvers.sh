#!/usr/bin/env sh
# Multinomial Logistic Regression
salloc --mem=128gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
run_mlr_comparison_all("/scratch/marque6/libsvm_data/ledgar.mat", struct('outDir','ledgar'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/dna.mat", struct('outDir','dna'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/mnist8m.mat", struct('outDir', 'mnist'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/aloi.mat", struct('outDir', 'aloi'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/usps.mat", struct('outDir', 'usps'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/news20.mat", struct('outDir','news20'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/rcv1_train.mat", struct('outDir','rcv1_train'))
run_mlr_comparison_all("/scratch/marque6/libsvm_data/mnist.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/letter.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/poker.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/shuttle.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/rcv1_train.mat")

run_mlr_comparison_all("/scratch/marque6/libsvm_data/rcv1_train.mat", struct('outDir', 'svrg-rcv1_train'))

###===###
# Multi-response Linear Regression
salloc --mem=128gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
compare_linear_regression("/scratch/marque6/libsvm_data/cadata.mat", struct('outDir', 'mrlr-cadata'))
compare_linear_regression("/scratch/marque6/libsvm_data/triazines.mat", struct('outDir', 'mrlr-triazines'))
compare_linear_regression("/scratch/marque6/libsvm_data/pyrim.mat", struct('outDir', 'mrlr-pyrim'))
compare_linear_regression("/scratch/marque6/libsvm_data/log1p_E2006.mat", struct('outDir', 'mrlr-log1p'))
compare_linear_regression("/scratch/marque6/libsvm_data/yearPredictionMSD.mat", struct('outDir', 'mrlr-yearPrediction'))
