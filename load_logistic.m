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
lambda1 = 1; % L1 + L2 regularization weight
lambda2 = 1;
max_n      = 500000; % subsample cap on #samples
N          = 100; % iterations (safety cap)
time_limit = 20; % time limit (s) per solver
seed       = 0;
x_mode     = 'time'; % 'iter' or 'time'
standardize = true;
add_bias = false;
remove_zero = true;

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
if ~isvector(y), [~, y] = max(y, [], 2); end % one-hot matrix -> index vector
y = double(y(:)); % force column vector
if size(Z,1) ~= numel(y), Z = Z'; end % fix orientation if transposed
[~, ~, y] = unique(y); % relabel to consecutive 1..k

[n, m] = size(Z);
if n > max_n % subsample for tractability
    sel = randperm(n, max_n);
    Z = Z(sel,:);  y = y(sel);  n = max_n;
end
y = y(:); % keep column after indexing
k = max(y);

y_b = zeros(n, k); % one-hot labels
for c = 1:k, y_b(:,c) = (y == c); end

if remove_zero
    nz = any(Z ~= 0, 1); % columns with >= 1 nonzero
    n_dropped = m - nnz(nz);
    Z = Z(:, nz);
    m = size(Z, 2);
    fprintf('Dropped %d all-zero feature columns (%d remain).\n', n_dropped, m);
end

if standardize
    mu_Z = mean(Z, 1);
    sd_Z = std(Z, 0, 1);  sd_Z(sd_Z == 0) = 1;
    Z = (Z - mu_Z) ./ sd_Z;
    % Z = Z - mu_Z;
end

if add_bias
    Z = [Z, ones(n, 1)];
    m = m + 1;
end

fprintf('%s: %d samples x %d features (bias=%d), %d classes, nnz=%d\n', ...
        dataset, n, m, add_bias, k, nnz(Z));

% === Obj
F = @(w) logistic_F(w, Z, y_b, k, lambda1, lambda2, n);

function val = logistic_F(w, Z, y_b, k, lambda1, lambda2, n)
    S = w(1:k-1,:) * Z';
    M = max(0, max(S, [], 1));
    denom = exp(-M) + sum(exp(S - M), 1);
    lin   = sum(sum(S .* y_b(:,1:k-1)'));
    % Summed Form
    % val = (-lin + sum(M + log(denom))) + lambda1*sum(abs(w(:))) + 0.5*lambda2*sum(w(:).^2);
    % Averaged Form
    val = (-lin + sum(M + log(denom)))/n + lambda1*sum(abs(w(:))) + 0.5*lambda2*sum(w(:).^2);
end
