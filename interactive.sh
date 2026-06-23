#!/usr/bin/env sh
# Single Script
salloc --mem=8gb --cpus-per-task=16 --time=08:00:00
module load matlab
matlab -nodisplay
run_mlr_comparison_all("/scratch/marque6/libsvm_data/letter.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/usps.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/poker.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/shuttle.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/news20.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/rcv1_train.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/mnist8m.mat")
run_mlr_comparison_all("/scratch/marque6/libsvm_data/mnist.mat")

# Sepearte Scripts
salloc --mem=512gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
dataset="mnist8m"; load_logistic; logistic_solvers

salloc --mem=256gb --cpus-per-task=32 --time=08:00:00
module load matlab
matlab -nodisplay
dataset="news20"; load_logistic; logistic_solvers

salloc --mem=512gb --cpus-per-task=128 --time=08:00:00
module load matlab
matlab -nodisplay
dataset="rcv1_train"; load_logistic; logistic_solvers



for l2 = [1, 10, 100, 1000]
    lam2_mean = l2/n;
    Lj_test = full(sum(Z.^2,1))'/(2*n) + lam2_mean;
    mu_H = lam2_mean/max(Lj_test);
    beta = 2*0.1/(1-0.1);
    m_in = floor((1/(0.1*mu_H)+beta)/(1-2*beta))+1;
    fprintf('lambda2=%6g  mu_H=%.3e  m_inner=%d\n', l2, mu_H, m_in);
end
