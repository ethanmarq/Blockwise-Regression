function results = compare_linear_regression(matFile, opts)
% min_X  0.5*||A*X - Y||_F^2 + lambda1*||X||_1 + 0.5*lambda2*||X||_F^2
% A: x n d -> X
% X: d x K -> W
% Y: n x K -> Y

    if nargin < 2
        opts = struct();
    end
    opts = fill_default_opts(opts);
    rng(opts.seed);

    if ~exist(opts.outDir, 'dir')
        mkdir(opts.outDir);
    end

    [~, datasetName] = fileparts(matFile);

    [X, Y] = load_xy_from_mat(matFile);
    [X, Y] = preprocess_xy(X, Y, opts);

    n = size(X,1);
    d = size(X,2);
    K = size(Y,2);

    fprintf('Loaded data "%s": n = %d, d = %d, K = %d\n', datasetName, n, d, K);
    fprintf(['Multi-response linear regression: ', ...
             '0.5||A X - Y||_F^2 + lambda1||X||_1 + 0.5 lambda2||X||_F^2\n']);
    fprintf('lambda1 = %.3g, lambda2 = %.3g\n', opts.lambda1, opts.lambda2);
    if isfinite(opts.timeLimit)
        fprintf('Per-solver time limit: %.1f s\n', opts.timeLimit);
    end

    W0 = zeros(d, K);

    results = struct();
    results.opts = opts;
    results.matFile = matFile;
    results.datasetName = datasetName;

    fprintf('\n[1/5] Running feature-wise cyclic BPG...\n');
    results.featurewise = featurewise_bpg_mlr(X, Y, W0, opts);

    fprintf('\n[2/5] Running class-wise cyclic BPG...\n');
    results.classwise = classwise_bpg_mlr(X, Y, W0, opts);

    fprintf('\n[3/5] Running whole-matrix proximal gradient...\n');
    results.whole_pg = whole_prox_gradient_mlr(X, Y, W0, opts);

    fprintf('\n[4/5] Running block-metric Prox-SVRG...\n');
    results.block_metric_prox_svrg = block_metric_prox_svrg_mlr(X, Y, W0, opts);

    fprintf('\n[5/5] Running whole-matrix Prox-SVRG...\n');
    results.whole_prox_svrg = whole_prox_svrg_mlr(X, Y, W0, opts);

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
    opts = set_default(opts, 'maxEpochsSVRG', 100);
    opts = set_default(opts, 'innerSVRG', []); %[]
    opts = set_default(opts, 'eta', 1.0);
    opts = set_default(opts, 'etaSVRG', 0.1);
    opts = set_default(opts, 'timeLimit', 20);
    opts = set_default(opts, 'maxSamples', 100000);
    opts = set_default(opts, 'standardize', true);
    opts = set_default(opts, 'addIntercept', false);
    opts = set_default(opts, 'seed', 1);
    opts = set_default(opts, 'outDir', 'mrlr_results_all');
    opts = set_default(opts, 'evalEvery', 1);
    opts = set_default(opts, 'verbose', true);
end

function opts = set_default(opts, name, value)
    if ~isfield(opts, name) || isempty(opts.(name))
        opts.(name) = value;
    end
end


function [X, Y] = load_xy_from_mat(matFile)
    S = load(matFile);

    xCandidates = {'A', 'X', 'data', 'features', 'Z', 'x'};
    yCandidates = {'Y', 'B', 'targets', 'responses', 'y', 'labels'};

    X = [];
    Y = [];

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
            Y = S.(nm);
            break;
        end
    end

    if xor(isempty(X), isempty(Y))
        if isempty(X), ref = Y; else, ref = X; end
        names = fieldnames(S);
        for i = 1:numel(names)
            v = S.(names{i});
            if (isnumeric(v) || islogical(v)) && ismatrix(v) && ~isscalar(v) ...
                    && size(v,1) == size(ref,1) && ~isequal(v, ref)
                if isempty(X), X = v; else, Y = v; end
                break;
            end
        end
    end

    if isempty(X) || isempty(Y)
        error(['Could not identify the design matrix A and response matrix Y ', ...
               'in %s. Please store them as variables A (design) and Y ', ...
               '(responses), or edit load_xy_from_mat().'], matFile);
    end

    X = double(X);
    Y = double(Y);
    if isvector(Y)
        Y = Y(:);
    end
end


function [X, Y] = preprocess_xy(X, Y, opts)
    if size(X,1) ~= size(Y,1) && size(X,2) == size(Y,1)
        X = X';
    end
    if size(X,1) ~= size(Y,1)
        error('A and Y must have the same number of rows (samples).');
    end

    % Optionally subsample rows
    n = size(X,1);
    if isfinite(opts.maxSamples) && n > opts.maxSamples
        N = round(opts.maxSamples);
        sel = randperm(n, N);
        X = X(sel, :);
        Y = Y(sel, :);
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


% Gauss-Sidel across feature chunks
% Jacobi within chunk
% Active-set screening
function out = featurewise_bpg_mlr(X, Y, W, opts)
    X = sparse(X);
    n = size(X,1);
    d = size(X,2);

    lambda1   = opts.lambda1;
    lambda2   = opts.lambda2;
    timeLimit = opts.timeLimit;
    eta       = opts.eta;

    density   = nnz(X) / max(1, n*d);
    csDefault = min(d, min(128, max(1, round(1/max(density, eps)))));
    fprintf('Chunksize: %.2e\n', csDefault);

    chunkSize   = getf(opts,'chunkSize', csDefault); % features per chunk
    tau         = getf(opts, 'tau', 1.0); % within-chunk damping >=1
    screenEvery = getf(opts, 'screenEvery', 5); % resync + rescreen period
    screenSlack = getf(opts, 'screenSlack', 1.0); % admit if viol > lam1*slack
    shuffle     = getf(opts, 'shuffle', true); % permute cols before chunking

    Z = X * W;   % n x K

    L = full(sum(X.^2, 1)).' + lambda2; % d x 1
    L = max(L, 1e-14);
    alphaVec = eta ./ (tau .* L);    % d x 1

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, Y, W, Z, lambda1, lambda2, 0, toc(t0), 0);
    iterCount = 0;

    active  = true(d,1);
    chunks  = {}; urows = {};
    rebuild = true;

    for epoch = 1:opts.maxEpochs
        % periodic exact resync of Z + KKT screening of the working set
        if isfinite(screenEvery) && mod(epoch-1, screenEvery) == 0
            Z = X * W; % kill incremental drift
            Gfull = X.' * (Z - Y) + lambda2 * W; % d x K  smooth gradient
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

        % Gauss-Seidel sweep over chunks
        stop = false;
        for c = 1:numel(chunks)
            cols = chunks{c};  r = urows{c};
            if isempty(cols) || isempty(r), continue; end

            Zr = Z(r,:); % |r| x K
            R  = Zr - Y(r,:); % residual (least squares)
            Xc = X(r, cols); % |r| x |cols| sparse

            G    = (Xc.' * R) + lambda2 * W(cols,:); % |cols| x K, ONE matmul
            a    = alphaVec(cols); % |cols| x 1
            Wold = W(cols,:);
            Tt   = Wold - a .* G; % per-feature step
            Wnew = sign(Tt) .* max(abs(Tt) - a .* lambda1, 0); % soft-threshold

            dW        = Wnew - Wold;
            W(cols,:) = Wnew;
            Z(r,:)    = Zr + Xc * dW; % score update, ONE matmul

            iterCount = iterCount + 1;
            if toc(t0) >= timeLimit, stop = true; break; end
        end

        hist = record_hist(hist, Y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
        if opts.verbose
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


% function out = featurewise_bpg_mlr(X, Y, W, opts)
% % O(K*nnz(A))

%     X = sparse(X);
%     d = size(X,2);
%     lambda1 = opts.lambda1;
%     lambda2 = opts.lambda2;
%     timeLimit = opts.timeLimit;
%     eta = opts.eta;

%     Z = X * W;
%     L = full(sum(X.^2, 1))' + lambda2;
%     L = max(L, 1e-14);

%     hist = init_hist();
%     t0 = tic;
%     hist = record_hist(hist, Y, W, Z, lambda1, lambda2, 0, toc(t0), 0);
%     iterCount = 0;
%     stop = false;

%     for epoch = 1:opts.maxEpochs
%         colIdx = cell(d,1);
%         colVal = cell(d,1);
%         for j = 1:d
%             [colIdx{j}, ~, colVal{j}] = find(X(:,j));
%         end
%         alphaVec = eta ./ L;
%         for j = 1:d
%             alpha  = alphaVec(j);
%             oldRow = W(j,:);
%             idx = colIdx{j};
%             xv  = colVal{j};
%             if isempty(idx)
%                 gj = lambda2 * oldRow;
%             else
%                 Zi = Z(idx,:);
%                 Ri = Zi - Y(idx,:);
%                 gj = (xv' * Ri) + lambda2 * oldRow; % = A(:,j)'(AW - Y) + lambda2*Wj
%             end
%             t = oldRow - alpha*gj;
%             newRow = sign(t).*max(abs(t)-alpha*lambda1, 0); % prox
%             delta  = newRow - oldRow;
%             W(j,:) = newRow;
%             if ~isempty(idx) && any(delta)
%                 Z(idx,:) = Zi + xv * delta;
%             end
%             iterCount = iterCount + 1;
%             if mod(j, 256) == 0 && toc(t0) >= timeLimit, stop = true; break; end
%         end
%         if stop || mod(epoch, opts.evalEvery) == 0 || epoch == opts.maxEpochs
%             hist = record_hist(hist, Y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
%             maybe_print(opts, 'featurewise', epoch, hist.obj(end));
%         end
%         if stop || toc(t0) >= timeLimit, break; end
%     end

%     out.W = W;
%     out.hist = hist;
%     out.L = L;
% end


function out = classwise_bpg_mlr(X, Y, W, opts)
    K = size(W,2);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;

    Z = X * W;

    L = spectral_norm_sq(X);
    L = max(L, 1e-14);
    alpha = opts.eta / L;

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, Y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    iterCount = 0;
    stop = false;
    for epoch = 1:opts.maxEpochs
        for k = 1:K
            gk = X' * (Z(:,k) - Y(:,k)); % = A'(A w_k - y_k)

            oldCol = W(:,k);
            newCol = elastic_net_prox(oldCol - alpha * gk, alpha, lambda1, lambda2); % prox
            delta = newCol - oldCol;

            W(:,k) = newCol;
            Z(:,k) = Z(:,k) + X * delta;
            iterCount = iterCount + 1;

            if toc(t0) >= timeLimit, stop = true; break; end
        end

        if stop || mod(epoch, opts.evalEvery) == 0 || epoch == opts.maxEpochs
            hist = record_hist(hist, Y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
            maybe_print(opts, 'classwise', epoch, hist.obj(end));
        end
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
    out.alpha = alpha;
end


function out = whole_prox_gradient_mlr(X, Y, W, opts)
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;

    Z = X * W;

    L = spectral_norm_sq(X);
    L = max(L, 1e-14);
    alpha = opts.eta / L;

    hist = init_hist();
    t0 = tic;
    hist = record_hist(hist, Y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    for it = 1:opts.maxIterWhole
        G = X' * (Z - Y); % = A'(AW - Y)

        W = elastic_net_prox(W - alpha * G, alpha, lambda1, lambda2); % prox
        Z = X * W;

        reachedTime = toc(t0) >= timeLimit;
        if reachedTime || mod(it, opts.evalEvery) == 0 || it == opts.maxIterWhole
            hist = record_hist(hist, Y, W, Z, lambda1, lambda2, it, toc(t0), it);
            maybe_print(opts, 'whole-pg', it, hist.obj(end));
        end
        if reachedTime, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
    out.alpha = alpha;
end

function out = block_metric_prox_svrg_mlr(X, Y, W, opts)
    n = size(X,1);
    d = size(X,2);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;
    etaSVRG = opts.etaSVRG;

    if isempty(opts.innerSVRG), m = d; else, m = opts.innerSVRG; end

    % H = n * full(max(X.^2, [], 1))'; % d-by-1
    % rowL1 = full(sum(abs(X), 2));                % ||x_i||_1
    % H = n * full(max(abs(X) .* rowL1, [], 1))';
    % H = max(H, 1e-14);

    colMaxSq = full(max(X.^2, [], 1)).';
    C        = full(max( (X.^2) * (1 ./ colMaxSq) ));
    H        = (n * C) * colMaxSq;
    H = max(H, 1e-14);
    alpha = etaSVRG ./ H; % d-by-1, broadcasts across the K columns

    hist = init_hist();
    Z = X * W;
    t0 = tic;
    hist = record_hist(hist, Y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    iterCount = 0;
    stop = false;
    for epoch = 1:opts.maxEpochsSVRG
        Wsnap = W;
        fullGradSnap = X' * (X * Wsnap - Y);

        for t = 1:m
            i  = randi(n);
            xi = X(i,:); % 1-by-d
            ri  = xi * W;  % 1-by-K  (current prediction)
            ris = xi * Wsnap; % 1-by-K  (snapshot prediction)
            v = n * (xi' * (ri - ris)) + fullGradSnap; % d-by-K

            A = W - alpha .* v;
            W = sign(A) .* max(abs(A) - alpha*lambda1, 0) ./ (1 + alpha*lambda2);

            iterCount = iterCount + 1;
            if mod(t, 1000) == 0 && toc(t0) >= timeLimit, stop = true; break; end
        end

        Z = X * W;
        hist = record_hist(hist, Y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
        maybe_print(opts, 'bm-prox-svrg', epoch, hist.obj(end));
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.H = H;
    out.innerSVRG = m;
end


function out = whole_prox_svrg_mlr(X, Y, W, opts)
    n = size(X,1);
    lambda1 = opts.lambda1;
    lambda2 = opts.lambda2;
    timeLimit = opts.timeLimit;
    etaSVRG = opts.etaSVRG;

    if isempty(opts.innerSVRG), m = n; else, m = opts.innerSVRG; end

    L = n * max(full(sum(X.^2, 2)));
    L = max(L, 1e-14);
    alpha = etaSVRG / L;

    hist = init_hist();
    Z = X * W;
    t0 = tic;
    hist = record_hist(hist, Y, W, Z, lambda1, lambda2, 0, toc(t0), 0);

    iterCount = 0;
    stop = false;
    for epoch = 1:opts.maxEpochsSVRG
        Wsnap = W;
        fullGradSnap = X' * (X * Wsnap - Y);
 
        for t = 1:m
            i  = randi(n);
            xi = X(i,:);
            ri  = xi * W;
            ris = xi * Wsnap;
            v = n * (xi' * (ri - ris)) + fullGradSnap;

            W = elastic_net_prox(W - alpha * v, alpha, lambda1, lambda2);

            iterCount = iterCount + 1;
            if mod(t, 1000) == 0 && toc(t0) >= timeLimit, stop = true; break; end
        end

        Z = X * W;
        hist = record_hist(hist, Y, W, Z, lambda1, lambda2, epoch, toc(t0), iterCount);
        maybe_print(opts, 'whole-svrg', epoch, hist.obj(end));
        if stop || toc(t0) >= timeLimit, break; end
    end

    out.W = W;
    out.hist = hist;
    out.L = L;
    out.alpha = alpha;
    out.innerSVRG = m;
end


function Wp = elastic_net_prox(A, alpha, lambda1, lambda2)
    Wp = soft_threshold(A, alpha * lambda1) / (1 + alpha * lambda2);
end


function S = soft_threshold(A, tau)
    S = sign(A) .* max(abs(A) - tau, 0);
end


function loss = data_loss_from_scores(Z, Y)
    R = Z - Y;
    loss = 0.5 * sum(R(:).^2); % 0.5*||A*X - Y||_F^2
end


function obj = objective_mlr(W, Z, Y, lambda1, lambda2)
    obj = data_loss_from_scores(Z, Y) + lambda1 * sum(abs(W(:))) + 0.5 * lambda2 * sum(W(:).^2);
end


function hist = init_hist()
    hist.obj = [];
    hist.data_loss = [];
    hist.nnz = [];
    hist.time = [];
    hist.iter = [];
    hist.epoch_or_iter = [];
end


function hist = record_hist(hist, Y, W, Z, lambda1, lambda2, epochOrIter, elapsed, iterCount)
    hist.obj(end+1,1) = objective_mlr(W, Z, Y, lambda1, lambda2);
    hist.data_loss(end+1,1) = data_loss_from_scores(Z, Y);
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
    labels = {'Feature-wise BPG', 'Class-wise BPG', 'Whole PG', 'Block-Metric Prox-SVRG', 'Whole Prox-SVRG'};

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
