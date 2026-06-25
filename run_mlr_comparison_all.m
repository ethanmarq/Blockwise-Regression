function results = run_mlr_comparison_all(matFile, opts)
%RUN_MLR_COMPARISON_ALL Compare proximal methods for elastic-net MLR.
%
% Minimal usage:
%   results = run_mlr_comparison_all('your_data.mat');
%
% The .mat file should ideally contain:
%   X   % n-by-d data matrix
%   y   % n-by-1 labels
%
% Objective:
%   F(W) = (1/n) sum_i softmax_loss_i(W)
%          + lambda1 * ||W||_1
%          + lambda2/2 * ||W||_F^2
%
% Compared methods:
%   1. Feature-wise cyclic block proximal gradient
%   2. Class-wise cyclic block proximal gradient
%   3. Whole-matrix proximal gradient
%   4. Block-Metric Prox-SVRG, feature-row metric
%   5. Whole-matrix Prox-SVRG
%
% Outputs in opts.outDir:
%   objective_vs_iteration.png
%   objective_vs_time.png
%   one CSV history file per method
%   results.mat
%
% Optional opts fields:
%   opts.lambda1       default 1e-2
%   opts.lambda2       default 1e-1
%   opts.maxEpochs     default 100       % for feature/class BPG
%   opts.maxIterWhole  default 300      % for whole PG
%   opts.maxEpochsSVRG default 100       % for both SVRG methods
%   opts.innerSVRG     default []       % if empty, uses 2*n
%   opts.eta           default 1.0      % PG/BPG damping, step = eta/L
%   opts.etaSVRG       default 0.1      % SVRG damping
%   opts.timeLimit     default inf      % wall-clock seconds per solver
%   opts.maxSamples    default inf      % randomly subsample to this many rows
%   opts.standardize   default false
%   opts.addIntercept  default false
%   opts.seed          default 1
%   opts.outDir        default 'mlr_results_all'
%   opts.evalEvery     default 1
%   opts.verbose       default true
%
% A solver stops at whichever limit it hits first: its epoch/iteration cap
% (maxEpochs / maxIterWhole / maxEpochsSVRG) or opts.timeLimit seconds.
%
% Lipschitz constants for the averaged loss:
%   whole PG:       L = ||X||_2^2 / (2n)
%   class-wise BPG: L = ||X||_2^2 / (4n)
%   feature BPG:    L_j = ||X(:,j)||_2^2 / (2n)
%
% SVRG component-loss constants:
%   whole Prox-SVRG:       L_i <= ||x_i||_2^2 / 2, so L = max_i ||x_i||^2 / 2
%   Block-Metric Prox-SVRG: H_j = max_i x_ij^2 / 2

    if nargin < 2
        opts = struct();
    end
    opts = fill_default_opts(opts);
    rng(opts.seed);

    if ~exist(opts.outDir, 'dir')
        mkdir(opts.outDir);
    end

    [~, datasetName] = fileparts(matFile);

    [X, y] = load_xy_from_mat(matFile);
    [X, y] = preprocess_xy(X, y, opts);

    n = size(X,1);
    d = size(X,2);
    K = max(y);

    fprintf('Loaded data "%s": n = %d, d = %d, K = %d\n', datasetName, n, d, K);
    fprintf('Objective: average softmax loss + %.3e ||W||_1 + %.3e/2 ||W||_F^2\n', ...
        opts.lambda1, opts.lambda2);
    if isfinite(opts.timeLimit)
        fprintf('Per-solver time limit: %.1f s\n', opts.timeLimit);
    end

    W0 = zeros(d, K);

    results = struct();
    results.opts = opts;
    results.matFile = matFile;
    results.datasetName = datasetName;

    fprintf('\n[1/5] Running feature-wise cyclic BPG...\n');
    results.featurewise = featurewise_bpg_mlr(X, y, W0, opts);

    fprintf('\n[2/5] Running class-wise cyclic BPG...\n');
    results.classwise = classwise_bpg_mlr(X, y, W0, opts);

    fprintf('\n[3/5] Running whole-matrix proximal gradient...\n');
    results.whole_pg = whole_prox_gradient_mlr(X, y, W0, opts);

    fprintf('\n[4/5] Running Block-Metric Prox-SVRG...\n');
    results.block_metric_prox_svrg = block_metric_prox_svrg_mlr(X, y, W0, opts);

    fprintf('\n[5/5] Running whole-matrix Prox-SVRG...\n');
    results.whole_prox_svrg = whole_prox_svrg_mlr(X, y, W0, opts);

    save_histories_and_plots(results, opts.outDir);

    resultsSlug = regexprep(datasetName, '[^A-Za-z0-9._-]', '_');
    if isempty(resultsSlug)
        resultsFile = 'results.mat';
    else
        resultsFile = sprintf('results_%s.mat', resultsSlug);
    end
    save(fullfile(opts.outDir, resultsFile), 'results', '-v7.3');
    fprintf('\nDone. Results saved to folder: %s\n', opts.outDir);
end


function opts = fill_default_opts(opts)
    opts = set_default(opts, 'lambda1', 1e-2);
    opts = set_default(opts, 'lambda2', 1e-1);
    opts = set_default(opts, 'maxEpochs', 500);
    opts = set_default(opts, 'maxIterWhole', 500);
    opts = set_default(opts, 'maxEpochsSVRG', 30);
    opts = set_default(opts, 'innerSVRG', []);
    opts = set_default(opts, 'eta', 1.0);
    opts = set_default(opts, 'etaSVRG', 0.1);
    opts = set_default(opts, 'timeLimit', 60);
    opts = set_default(opts, 'maxSamples', 100000);
    opts = set_default(opts, 'standardize', true);
    opts = set_default(opts, 'addIntercept', false);
    opts = set_default(opts, 'seed', 1);
    opts = set_default(opts, 'outDir', 'mlr_results_all');
    opts = set_default(opts, 'evalEvery', 1);
    opts = set_default(opts, 'verbose', true);
end


function opts = set_default(opts, name, value)
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = value;
    end
end


function [X, y] = load_xy_from_mat(matFile)
    S = load(matFile);

    xCandidates = {'Z', 'X', 'data', 'features', 'A', 'x'};
    yCandidates = {'y', 'Y', 'labels', 'label', 'target', 'targets'};

    X = [];
    y = [];

    for i = 1:numel(xCandidates)
        nm = xCandidates{i};
        if isfield(S, nm)
            X = S.(nm);
            break;
        end
    end

    for i = 1:numel(yCandidates)
        nm = yCandidates{i};
        if isfield(S, nm)
            yy = S.(nm);
            if isnumeric(yy) || islogical(yy)
                if isvector(yy)
                    y = yy;
                    break;
                elseif ismatrix(yy) && size(yy,1) == size(X,1) && size(yy,2) > 1
                    [~, y] = max(yy, [], 2);
                    break;
                end
            end
        end
    end

    if isempty(X) || isempty(y)
        names = fieldnames(S);
        numericNames = {};
        for i = 1:numel(names)
            val = S.(names{i});
            if isnumeric(val) || islogical(val)
                numericNames{end+1} = names{i}; %#ok<AGROW>
            end
        end

        for i = 1:numel(numericNames)
            A = S.(numericNames{i});
            if ismatrix(A) && ~isvector(A)
                for j = 1:numel(numericNames)
                    b = S.(numericNames{j});
                    if isvector(b) && numel(b) == size(A,1)
                        X = A;
                        y = b;
                        break;
                    end
                end
            end
            if ~isempty(X) && ~isempty(y)
                break;
            end
        end
    end

    if isempty(X) || isempty(y)
        error(['Could not automatically identify X and y in %s. ', ...
               'Please store variables as X and y, or edit load_xy_from_mat().'], matFile);
    end

    X = double(X);
    y = double(y(:));
end



function [X, y] = preprocess_xy(X, y, opts)
    if size(X,1) ~= numel(y) && size(X,2) == numel(y)
        X = X';
    end
    if size(X,1) ~= numel(y)
        error('X and y dimensions do not match.');
    end

    [~, ~, y] = unique(y);
    y = double(y(:));

    % Optionally subsample rows (e.g. mnist8m: keep a few hundred thousand).
    n = size(X,1);
    if isfinite(opts.maxSamples) && n > opts.maxSamples
        N = round(opts.maxSamples);
        sel = randperm(n, N);
        X = X(sel, :);
        y = y(sel);
        [~, ~, y] = unique(y);   % relabel in case a class disappeared
        y = double(y(:));
        fprintf('Subsampled %d -> %d rows.\n', n, N);
    end

    if opts.standardize
        if issparse(X)
            colScale = sqrt(sum(X.^2, 1) / max(1, size(X,1)));
            colScale = full(colScale);
            colScale(colScale < 1e-12) = 1;
            X = X * spdiags(1 ./ colScale(:), 0, size(X,2), size(X,2));
        else
            mu = mean(X, 1);
            sigma = std(X, 0, 1);
            sigma(sigma < 1e-12) = 1;
            X = bsxfun(@rdivide, bsxfun(@minus, X, mu), sigma);
        end
    end

    if opts.addIntercept
        X = [X, ones(size(X,1), 1)];
    end
end

function out = featurewise_bpg_mlr(X, y, W, opts)
%FEATUREWISE_BPG_MLR  Chunked + screened feature-wise block prox-grad for MLR.
%   Drop-in replacement for the cyclic featurewise_bpg_mlr in
%   run_mlr_comparison_all.m. Same signature, same output struct, same
%   helpers (softmax_rows, one_hot_labels, init_hist, record_hist).
%
%   Idea: Gauss-Seidel ACROSS feature chunks (keeps the per-feature
%   preconditioning that wins on news20), Jacobi WITHIN a chunk so each
%   chunk's gradient + score update are single matmuls (the BLAS-3
%   efficiency Whole PG enjoys, which is what large K rewards). Active-set
%   screening shrinks the effective d, independent of K.
%
%   Layout (matches the file): X n-by-d, W d-by-K (features x classes),
%   full K-class softmax over columns, L_j = ||X(:,j)||^2/(2n)+lambda2.
%
%   Extra opts (all optional, sensible defaults):
%     opts.chunkSize   features per chunk          (default 1024)
%     opts.tau         within-chunk damping >= 1   (default 1.0)
%     opts.screenEvery resync + rescreen period    (default 5; inf = off)
%     opts.screenSlack admit if viol > lam1*slack  (default 1.0)
%     opts.shuffle     permute cols before chunking (default true)

    X = sparse(X);
    n = size(X,1);
    d = size(X,2);
    K = size(W,2);

    lambda1   = opts.lambda1;
    lambda2   = opts.lambda2;
    timeLimit = opts.timeLimit;
    eta       = opts.eta;

    chunkSize   = getf(opts,'chunkSize', 32);
    tau         = getf(opts,'tau',1.0);
    screenEvery = getf(opts,'screenEvery',5);
    screenSlack = getf(opts,'screenSlack',1.0);
    shuffle     = getf(opts,'shuffle',true);

    Y = one_hot_labels(y, K); % n x K
    Z = X * W; % n x K

    L = full(sum(X.^2, 1)).' / (2*n) + lambda2; % d x 1
    L = max(L, 1e-14);
    alphaVec = eta ./ (tau .* L); % d x 1

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);
    iterCount = 0;

    active  = true(d,1);
    chunks  = {}; urows = {};
    rebuild = true;

    for epoch = 1:opts.maxEpochs
        % periodic exact resync of Z + KKT screening of the working set
        if isfinite(screenEvery) && mod(epoch-1, screenEvery) == 0
            Z = X * W; % kill incremental drift
            P = softmax_rows(Z); % n x K
            Gfull = X.' * (P - Y) / n + lambda2 * W; % d x K
            viol  = max(abs(Gfull), [], 2); % d x 1  KKT slack
            active = any(W ~= 0, 2) | (viol > lambda1 * screenSlack);
            rebuild = true;
        end

        % rebuild chunk index sets over active features
        if rebuild
            ac = find(active);
            if shuffle, ac = ac(randperm(numel(ac))); end
            nC = max(1, ceil(numel(ac)/chunkSize));
            chunks = cell(1,nC); urows = cell(1,nC);
            for c = 1:nC
                cols = ac((c-1)*chunkSize+1 : min(c*chunkSize, numel(ac)));
                chunks{c} = cols;
                urows{c}  = find(any(X(:,cols) ~= 0, 2)); % union support
            end
            rebuild = false;
        end

        % Gauss-Sidel sweep over chunks
        stop = false;
        for c = 1:numel(chunks)
            cols = chunks{c};  r = urows{c};
            if isempty(cols) || isempty(r), continue; end

            Zr = Z(r,:); % |r| x K
            Pr = softmax_rows(Zr); % |r| x K
            R  = Pr - Y(r,:); % residual
            Xc = X(r, cols); % |r| x |cols| sparse

            G    = (Xc.' * R) / n + lambda2 * W(cols,:); % |cols| x K, ONE matmul
            a    = alphaVec(cols); % |cols| x 1
            Wold = W(cols,:);
            Tt   = Wold - a .* G; % per-feature step
            Wnew = sign(Tt) .* max(abs(Tt) - a .* lambda1, 0); % soft-threshold

            dW       = Wnew - Wold;
            W(cols,:) = Wnew;
            Z(r,:)    = Zr + Xc * dW; % score update, ONE matmul

            iterCount = iterCount + 1;
            if toc(t0) >= timeLimit, stop = true; break; end
        end

        hist = record_hist(hist, y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
        if isfield(opts,'verbose') && opts.verbose
            fprintf('%-16s epoch = %4d, obj = %.8e, active = %d\n', ...
                    'feat-chunk', epoch, hist.obj(end), nnz(active));
        end
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W     = W;
    out.hist  = hist;
    out.L     = L;
    out.alpha = alphaVec;
    out.info  = struct('active_final', nnz(active), 'nnz_W', nnz(W), ...
                       'epochs', epoch, 'chunkSize', chunkSize, 'tau', tau);
end

function v = getf(s, f, dflt)
    if isstruct(s) && isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = dflt; end
end

function out = featurewise_bpg_mlr_old(X, y, W, opts)
% Softmax/gradient computed on only nonzero rows of each column.
% Per-epoch cost drops from O(d*n*K) to O(K*nnz(X))

    X = sparse(X);
    n = size(X,1);
    d = size(X,2);
    K = size(W,2);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;
    eta = opts.eta;

    Y = one_hot_labels(y, K);
    Z = X * W;
    % L = full(sum(X.^2, 1))' / (2*n);
    L = full(sum(X.^2, 1))' / (2*n) + lambda2;
    L = max(L, 1e-14);
    tau = 2;

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);
    iterCount = 0;
    stop = false;

    for epoch = 1:opts.maxEpochs
        colIdx = cell(d,1);
        colVal = cell(d,1);
        for j = 1:d
            [colIdx{j}, ~, colVal{j}] = find(X(:,j));
        end
        alphaVec = eta ./ L;
        for j = 1:d
            alpha  = alphaVec(j);
            oldRow = W(j,:);
            idx = colIdx{j};
            xv  = colVal{j};
            if isempty(idx)
                % gj = zeros(1, K);
                gj = lambda2 * oldRow;
            else
                Zi = Z(idx,:);
                Pi = softmax_rows(Zi); % softmax on support only
                % gj = (xv' * (Pi - Y(idx,:))) / n;
                gj = (xv' * (Pi - Y(idx,:))) / n + lambda2*oldRow;

            end
            % newRow = elastic_net_prox(oldRow - alpha * gj, alpha, lambda1, lambda2);
            t = oldRow - alpha*gj;
            newRow = sign(t).*max(abs(t)-alpha*lambda1, 0);
            delta  = newRow - oldRow;
            W(j,:) = newRow;
            if ~isempty(idx) && any(delta)
                Z(idx,:) = Zi + xv * delta;
            end
            iterCount = iterCount + 1;
            if mod(j, 256) == 0 && toc(t0) >= timeLimit, stop = true; break; end
        end
        if stop || mod(epoch, opts.evalEvery) == 0 || epoch == opts.maxEpochs
            hist = record_hist(hist, y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
            maybe_print(opts, 'featurewise', epoch, hist.obj(end));
        end
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
end


% function out = featurewise_bpg_mlr(X, y, W, opts)
%     n = size(X,1);
%     d = size(X,2);
%     K = size(W,2);
%     lambda1 = opts.lambda1;
%     lambda2 = opts.lambda2;
%     timeLimit = opts.timeLimit;

%     Y = one_hot_labels(y, K);
%     Z = X * W;

%     L = full(sum(X.^2, 1))' / (2*n);
%     L = max(L, 1e-14);

%     hist = init_hist();
%     t0 = tic;
%     hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

%     iterCount = 0;
%     stop = false;
%     for epoch = 1:opts.maxEpochs
%         for j = 1:d
%             P = softmax_rows(Z);
%             gj = X(:,j)' * (P - Y) / n;

%             alpha = opts.eta / L(j);
%             oldRow = W(j,:);
%             newRow = elastic_net_prox(oldRow - alpha * gj, alpha, lambda1, lambda2);
%             delta = newRow - oldRow;

%             W(j,:) = newRow;
%             Z = Z + X(:,j) * delta;
%             iterCount = iterCount + 1;

%             if mod(j, 256) == 0 && toc(t0) >= timeLimit, stop = true; break; end
%         end

%         if stop || mod(epoch, opts.evalEvery) == 0 || epoch == opts.maxEpochs
%             hist = record_hist(hist, y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
%             maybe_print(opts, 'featurewise', epoch, hist.obj(end));
%         end
%         if stop || toc(t0) >= timeLimit, break; end
%     end

%     out.W = W;
%     out.hist = hist;
%     out.L = L;
% end


function out = classwise_bpg_mlr(X, y, W, opts)
    n = size(X,1);
    K = size(W,2);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;

    Y = one_hot_labels(y, K);
    Z = X * W;

    L = spectral_norm_sq(X) / (4*n);
    L = max(L, 1e-14);
    alpha = opts.eta / L;

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    iterCount = 0;
    stop = false;
    for epoch = 1:opts.maxEpochs
        for k = 1:K
            P = softmax_rows(Z);
            gk = X' * (P(:,k) - Y(:,k)) / n;

            oldCol = W(:,k);
            newCol = elastic_net_prox(oldCol - alpha * gk, alpha, lambda1, lambda2);
            delta = newCol - oldCol;

            W(:,k) = newCol;
            Z(:,k) = Z(:,k) + X * delta;
            iterCount = iterCount + 1;

            if toc(t0) >= timeLimit, stop = true; break; end
        end

        if stop || mod(epoch, opts.evalEvery) == 0 || epoch == opts.maxEpochs
            hist = record_hist(hist, y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
            maybe_print(opts, 'classwise', epoch, hist.obj(end));
        end
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
    out.alpha = alpha;
end


function out = whole_prox_gradient_mlr(X, y, W, opts)
    n = size(X,1);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;

    Z = X * W;
    Y = one_hot_labels(y, size(W,2));

    L = spectral_norm_sq(X) / (2*n);
    L = max(L, 1e-14);
    alpha = opts.eta / L;

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    for it = 1:opts.maxIterWhole
        P = softmax_rows(Z);
        G = X' * (P - Y) / n;

        W = elastic_net_prox(W - alpha * G, alpha, lambda1, lambda2);
        Z = X * W;

        reachedTime = toc(t0) >= timeLimit;
        if reachedTime || mod(it, opts.evalEvery) == 0 || it == opts.maxIterWhole
            hist = record_hist(hist, y, W, Z, lambda1, lambda2, it, toc(t0), it);
            maybe_print(opts, 'whole-pg', it, hist.obj(end));
        end
        if reachedTime, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
    out.alpha = alpha;
end

function out = block_metric_prox_svrg_mlr(X, y, W, opts)
    n = size(X,1);
    d = size(X,2);
    K = size(W,2);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;

    % Feature-row metric for individual losses.
    % H = 0.5 * full(max(X.^2, [], 1))';
    % H = H.*sqrt(d);
    % H= full(sum(X.^2, 1))' / (2*sqrt(n));
    % H= max(H, 1e-14);
    rowL1 = full(sum(abs(X), 2));  % n-by-1
    H = 0.5 * full(max(abs(X) .* rowL1, [], 1))';
    H = max(H, 1e-14);
    alpha = opts.etaSVRG ./ H(:);


    hist = init_hist();
    Z = X * W;
    t0 = tic;
    hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    beta = (2*opts.etaSVRG)/(1-opts.etaSVRG);
    mu_h = opts.lambda2 / max(H);
    m = floor((1/(opts.etaSVRG*mu_h)+beta)/(1-2*beta)) + 1;
    % m = floor(m*max(H)*beta / (d));
    % m = floor(m/(0.5*sqrt(d)));
    % if floor(m/(sqrt(n)*d)) ~= 0
    %     m = floor(m/(sqrt(n)*d));
    % else
    %     m = floor(m/sqrt(n));
    % end


    iterCount = 0;
    stop = false;
    for epoch = 1:opts.maxEpochsSVRG
        Wsnap = W;
        Zsnap = X * Wsnap;
        Psnap = softmax_rows(Zsnap);
        Yall = one_hot_labels(y, K);
        fullGradSnap = X' * (Psnap - Yall) / n;

        for t = 1:m
            i = randi(n);
            xi = X(i,:);

            pi = softmax_rows(xi * W);
            pis = softmax_rows(xi * Wsnap);

            yi = zeros(1, K);
            yi(y(i)) = 1;

            v = xi'*pi - xi'*pis + fullGradSnap;

            % Full block-metric proximal update, decomposed across feature rows.
            % for j = 1:d
            %     alpha_j = opts.etaSVRG / H(j);
            %     W(j,:) = elastic_net_prox(W(j,:) - alpha_j * v(j,:), alpha_j, lambda1, lambda2);
            % end
            % d x K, broadcast over rows
            A = W - alpha .* v;
            W = sign(A) .* max(abs(A) - alpha*lambda1, 0) ./ (1 + alpha*lambda2);

            iterCount = iterCount + 1;

            if mod(t, 1000) == 0 && toc(t0) >= timeLimit, stop = true; break; end
        end

        Z = X * W;
        hist = record_hist(hist, y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
        maybe_print(opts, 'bm-prox-svrg', epoch, hist.obj(end));
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.H = H;
    out.innerSVRG = m;
end


function out = whole_prox_svrg_mlr(X, y, W, opts)
    n = size(X,1);
    K = size(W,2);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;

    if isempty(opts.innerSVRG)
        m = 2*n;
    else
        m = opts.innerSVRG;
    end

    % Uniform component-loss smoothness bound:
    % ||nabla loss_i(W)-nabla loss_i(V)|| <= L_i ||W-V||_F,
    % L_i <= ||x_i||^2 / 2.
    rowNormSq = full(sum(X.^2, 2));
    L = 0.5 * max(rowNormSq);
    L = max(L, 1e-14);
    alpha = opts.etaSVRG / L;

    beta = (2*opts.etaSVRG)/(1-opts.etaSVRG);
    mu_h = opts.lambda2 / max(L);
    m = floor((1/(opts.etaSVRG*mu_h)+beta)/(1-2*beta)) + 1;
    % m = floor(m / sqrt(n));


    hist = init_hist();
    Z = X * W;
    t0 = tic;
    hist = record_hist(hist, y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    iterCount = 0;
    stop = false;
    for epoch = 1:opts.maxEpochsSVRG
        Wsnap = W;
        Zsnap = X * Wsnap;
        Psnap = softmax_rows(Zsnap);
        Yall = one_hot_labels(y, K);
        fullGradSnap = X' * (Psnap - Yall) / n;

        for t = 1:m
            i = randi(n);
            xi = X(i,:);

            pi = softmax_rows(xi * W);
            pis = softmax_rows(xi * Wsnap);

            yi = zeros(1, K);
            yi(y(i)) = 1;

            grad_i_cur = xi' * (pi - yi);
            grad_i_snap = xi' * (pis - yi);

            v = grad_i_cur - grad_i_snap + fullGradSnap;

            % Whole-matrix Prox-SVRG update.
            W = elastic_net_prox(W - alpha * v, alpha, lambda1, lambda2);
            iterCount = iterCount + 1;

            if mod(t, 1000) == 0 && toc(t0) >= timeLimit, stop = true; break; end
        end

        Z = X * W;
        hist = record_hist(hist, y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
        maybe_print(opts, 'whole-prox-svrg', epoch, hist.obj(end));
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
    out.alpha = alpha;
    out.innerSVRG = m;
end


function P = softmax_rows(Z)
    Zmax = max(Z, [], 2);
    Zs = bsxfun(@minus, Z, Zmax);
    E = exp(Zs);
    P = bsxfun(@rdivide, E, sum(E, 2));
end


function Y = one_hot_labels(y, K)
    n = numel(y);
    Y = zeros(n, K);
    Y(sub2ind([n, K], (1:n)', y(:))) = 1;
end


function Wp = elastic_net_prox(A, alpha, lambda1, lambda2)
    Wp = soft_threshold(A, alpha * lambda1) / (1 + alpha * lambda2);
end


function S = soft_threshold(A, tau)
    S = sign(A) .* max(abs(A) - tau, 0);
end


function loss = data_loss_from_scores(Z, y)
    Zmax = max(Z, [], 2);
    logsumexp = log(sum(exp(bsxfun(@minus, Z, Zmax)), 2)) + Zmax;
    n = numel(y);
    loss = mean(logsumexp - Z(sub2ind(size(Z), (1:n)', y(:))));
end


function obj = objective_mlr(W, Z, y, lambda1, lambda2)
    obj = data_loss_from_scores(Z, y) + lambda1 * sum(abs(W(:))) + 0.5 * lambda2 * sum(W(:).^2);
end


function hist = init_hist()
    hist.obj = [];
    hist.data_loss = [];
    hist.nnz = [];
    hist.time = [];
    hist.iter = [];
    hist.epoch_or_iter = [];
end


function hist = record_hist(hist, y, W, Z, lambda1, lambda2, epochOrIter, elapsed, iterCount)
    hist.obj(end+1,1) = objective_mlr(W, Z, y, lambda1, lambda2);
    hist.data_loss(end+1,1) = data_loss_from_scores(Z, y);
    hist.nnz(end+1,1) = nnz(W);
    hist.time(end+1,1) = elapsed;
    hist.iter(end+1,1) = iterCount;
    hist.epoch_or_iter(end+1,1) = epochOrIter;
end


function sn2 = spectral_norm_sq(X)
    try
        sn2 = normest(X)^2;
    catch
        sn2 = norm(X, 2)^2;
    end
end


function maybe_print(opts, name, it, obj)
    if opts.verbose
        fprintf('%-16s step = %5d, obj = %.8e\n', name, it, obj);
    end
end


function save_histories_and_plots(results, outDir)
    methods = {'featurewise', 'classwise', 'whole_pg', 'block_metric_prox_svrg', 'whole_prox_svrg'};
    labels = {'Feature-wise BPG', 'Class-wise BPG', 'Whole PG', ...
              'Block-Metric Prox-SVRG', 'Whole Prox-SVRG'};

    for m = 1:numel(methods)
        method = methods{m};
        H = results.(method).hist;
        T = table(H.iter, H.epoch_or_iter, H.time, H.obj, H.data_loss, H.nnz, ...
            'VariableNames', {'iteration', 'epoch_or_iter', 'time_sec', 'objective', 'data_loss', 'nnz'});
        writetable(T, fullfile(outDir, [method, '_history.csv']));
    end

    fig1 = figure('Visible', 'off');
    hold on;
    for m = 1:numel(methods)
        H = results.(methods{m}).hist;
        plot(H.iter, H.obj, 'LineWidth', 1.8);
    end
    hold off;
    grid on;
    xlabel('Iteration count');
    ylabel('Objective');

    dsName = results.datasetName;
    dsTag  = strrep(dsName, '_', '\_');

    if isempty(dsTag)
        title('Objective vs iteration');
    else
        title(sprintf('%s: objective vs iteration', dsTag));
    end
    legend(labels, 'Location', 'best');
    saveas(fig1, fullfile(outDir, 'objective_vs_iteration.png'));
    saveas(fig1, fullfile(outDir, 'objective_vs_iteration.fig'));
    legend(labels, 'Location', 'best');
    saveas(fig1, fullfile(outDir, 'objective_vs_iteration.png'));
    saveas(fig1, fullfile(outDir, 'objective_vs_iteration.fig'));

    fig2 = figure('Visible', 'off');
    hold on;
    for m = 1:numel(methods)
        H = results.(methods{m}).hist;
        plot(H.time, H.obj, 'LineWidth', 1.8);
    end
    hold off;
    grid on;
    xlabel('Time, seconds');
    ylabel('Objective');
    if isempty(dsTag)
        title('Objective vs time');
    else
        title(sprintf('%s: objective vs time', dsTag));
    end
    legend(labels, 'Location', 'best');
    saveas(fig2, fullfile(outDir, 'objective_vs_time.png'));
    saveas(fig2, fullfile(outDir, 'objective_vs_time.fig'));
    legend(labels, 'Location', 'best');
    saveas(fig2, fullfile(outDir, 'objective_vs_time.png'));
    saveas(fig2, fullfile(outDir, 'objective_vs_time.fig'));

    close(fig1);
    close(fig2);
end
