% load_logistic.m
% === CONFIG
% dataset = 'mnist'
% k = max(y) is read from the labels (number of classes)
% 10 Classes:
% - usps
% - mnist
% 101 Classes:
% - rcv1_topics_train
% 2 Classes:
% - news20
% - rcv1_train
% dataset    = 'news20';
data_path  = sprintf('/scratch/marque6/libsvm_data/%s.mat', dataset);
lambda     = 1;            % L1 + L2 regularization weight  (try 1/sqrt(n))
max_n      = 10000;        % subsample cap on #samples
N          = 500;          % iterations (safety cap)
time_limit = 15;         % time limit (s) per solver
seed       = 0;
x_mode     = 'iter';       % 'iter' or 'time'
% === LOAD
S = load(data_path);
rng(seed);

% Detect label vector and feature matrix
labelnames = {'y','Y','labels','label','classes','class'};
featnames   = {'Z','X','inst','instance_matrix','features','data','A'};
y = []; Z = [];
for c = labelnames, if isfield(S,c{1}), y = S.(c{1}); break; end, end
for c = featnames,  if isfield(S,c{1}), Z = S.(c{1}); break; end, end
if isempty(y) || isempty(Z)
    error('Could not find label/feature fields. File contains: %s', ...
          strjoin(fieldnames(S), ', '));
end

if size(Z,2) < size(y,2)
    [Z, y] = deal(y, Z);
    warning('load_logistic:swap', ...
        'Feature/label fields looked swapped by name; swapped by width.');
end

Z = full(Z);
if ~isvector(y), [~, y] = max(y, [], 2); end  % one-hot matrix -> index vector
y = double(y(:));                              % force column vector
if size(Z,1) ~= numel(y), Z = Z'; end          % fix orientation if transposed
[~, ~, y] = unique(y);                          % relabel to consecutive 1..k

[n, m] = size(Z);
if n > max_n                                % subsample for tractability
    sel = randperm(n, max_n);
    Z = Z(sel,:);  y = y(sel);  n = max_n;
end
y = y(:);                                       % keep column after indexing
k = max(y);

y_b = zeros(n, k);                          % one-hot labels
for c = 1:k, y_b(:,c) = (y == c); end

fprintf('%s: %d samples x %d features, %d classes, nnz=%d\n', ...
        dataset, n, m, k, nnz(Z));

% Standardize features, MNIST overflows
mu_Z = mean(Z, 1);
sd_Z = std(Z, 0, 1);  sd_Z(sd_Z == 0) = 1;
Z = (Z - mu_Z) ./ sd_Z;


% ----- objective:  F(w) = -loglik + lambda*||w||_1 + (lambda/2)*||w||^2 -----
% (last row of w is pinned to 0: the softmax reference class)
% F = @(w) -trace(Z*w(y,:)') + sum(log(1 + sum(exp(w(1:k-1,:)*Z')))) ...
F = @(w) logistic_F(w, Z, y_b, k, lambda);

% ----- gradient-Lipschitz constants (per solver) ---------------------------
L_feat = sum(Z.^2, 1)/2 + lambda;           % 1 x m, per-feature   (F-CBPG)
L_spec = norm(Z, 2)^2/4 + lambda;           % spectral             (C-CBPG)
L_full = norm(Z, 2)^2/2 + lambda;           % spectral             (Whole)
L_samp = max(sum(Z.^2, 2)/2) + lambda/n;    % max row norm         (SVRG/SAGA)

fprintf('L_feat(max)=%.3e, L_spec=%.3e, L_full=%.3e, L_samp=%.3e\n', ...
        max(L_feat), L_spec, L_full, L_samp);

function val = logistic_F(w, Z, y_b, k, lambda)
    S = w(1:k-1,:) * Z';
    M = max(0, max(S, [], 1));                  % include reference logit (=0)
    denom = exp(-M) + sum(exp(S - M), 1);
    lin   = sum(sum(S .* y_b(:,1:k-1)'));       % sum_i s_{i,y_i} (0 if reference)
    val = -lin + sum(M + log(denom)) ...
          + lambda*sum(abs(w(:))) + 0.5*lambda*sum(w(:).^2);
end
