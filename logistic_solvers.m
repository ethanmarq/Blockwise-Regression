% logistic_solvers.m
% L1+L2 multi
% Every method here uses the soft-threshold prox, so each genuinely solves the
% L1+L2 problem and produces exact zeros.
% Expects Z, y, y_b, n, m, k, lambda, N, time_limit, x_mode, dataset, F,
% and L_feat, L_spec, L_full, L_samp in the workspace (run load_logistic).

% ===  gradient-Lipschitz constants (per solver)
% NOTE: Remove n from F-CBPG, C-CBPG, Whole when using Summed Form
% = F-CBPG
L_feat = sum(Z.^2, 1)/(2*n) + lambda2;
% = C-CBPG
L_spec = norm(Z, 2)^2/(4*n) + lambda2;
% = Whole
% Without STD
% L_full = norm(Z, 2)^2/(2*n) + lambda2;
% With STD
L_full = 1/n + lambda2;
% = SAGA
L_samp = max(sum(Z.^2, 2)/2) + lambda2;
% = SVRG
c = max(full(sum(Z.^2, 2)))/(2*n);
c = max(c, 1e-12);
% = BM-SVRG
% Incorrect because lam2 in proximal
% Lj = full(max(Z.^2,[],1))'/2 + lam2;

% Working
Lj  = full(sum(Z.^2,1))'/(2*n);
Lj  = max(Lj, 1e-12);

% Working SVRG = BM-SVRG
% Lj = (max(full(sum(Z.^2,2)))/2) * ones(m,1);
% Lj  = max(Lj, 1e-12);

% Experimental
% rl1 = full(sum(abs(Z), 2));
% Lj  = (max(abs(Z) .* rl1, [], 1).') / 2;
% Lj  = max(Lj, 1e-12);
% ===

fprintf('L_feat(max)=%.3e, L_spec=%.3e, L_full=%.3e, L_samp=%.3e\n', ...
        max(L_feat), L_spec, L_full, L_samp);

w_init = randn(k, m); w_init(k,:) = 0; % shared start; reference row = 0
Z  = sparse(Z); % keep sparse for all solvers
Yt = y_b(:,1:k-1)'; % (k-1) x n one-hot tail

% === BM-SVRG  (Block-Metric Prox-SVRG; per-feature metric + elastic-net prox)
fprintf('BM-SVRG...\n');
lam1 = lambda1; lam2 = lambda2;


fprintf(' max Lj=%.3e',max(Lj));

eta = 0.1; % Theory requies eta < 1/5
stepj = (eta ./ Lj).'; % 1 x m
beta    = 2*eta/(1-eta);
mu_H    = lam2/max(Lj);
m_inner = floor((1/(eta*mu_H)+beta)/(1-2*beta)) + 1;
w = w_init;
F_bmsvrg = zeros(1,N); F_bmsvrg(1) = F(w);
T_bmsvrg = zeros(1,N); T_bmsvrg(1) = 0;
tic
stop = false;
for it = 2:N
    Ws    = w;
    Gsnap = (softmax_tail(Ws,Z) - Yt)*Z / n;
    for jj = 1:m_inner
        i  = randi(n);
        zi = Z(i,:);
        sW  = w(1:k-1,:)*zi';   Mw = max(0,max(sW));  pW  = exp(sW-Mw);  pW  = pW /(exp(-Mw)+sum(pW));
        sWs = Ws(1:k-1,:)*zi';  Ms = max(0,max(sWs)); pWs = exp(sWs-Ms); pWs = pWs/(exp(-Ms)+sum(pWs));
        v = (pW - pWs)*zi + Gsnap;
        Wt = w(1:k-1,:) - stepj.*v;
        w(1:k-1,:) = sign(Wt).*max(abs(Wt)-stepj*lam1,0) ./ (1+stepj*lam2);
        if mod(jj,1000)==0 && toc >= time_limit, stop = true; break; end
    end
    F_bmsvrg(it) = F(w); T_bmsvrg(it) = toc;
    if stop || T_bmsvrg(it) >= time_limit, break; end
end
w_bmsvrg = w; iter_bmsvrg = it;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d (m_inner=%d)\n', ...
    T_bmsvrg(it), it, F_bmsvrg(it), nnz(w(1:k-1,:)), (k-1)*m, m_inner);

% === C-CBPG  (cyclic block prox-grad over CLASS rows; spectral step L_spec)
fprintf('C-CBPG...\n');
w = w_init;
F_ccbpg = zeros(1, N); F_ccbpg(1) = F(w);
T_ccbpg = zeros(1, N); T_ccbpg(1) = 0;
tic
for it = 2:N
    for h = 1:k-1
        G = logreg_grad(w, Z, y_b, lambda2);
        t = w(h,:) - G(h,:)/L_spec;
        w(h,:) = sign(t).*max(abs(t) - lambda1/L_spec, 0);
    end
    F_ccbpg(it) = F(w); T_ccbpg(it) = toc;
    if T_ccbpg(it) >= time_limit, break; end
end
iter_ccbpg = it; w_ccbpg = w;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_ccbpg(it), it, F_ccbpg(it), nnz(w(1:k-1,:)), (k-1)*m);

% === F-CBPG  (cyclic block prox-grad over FEATURE columns; per-feature L_feat)
fprintf('F-CBPG...\n');
nz = cell(1, m);
for h = 1:m, nz{h} = find(Z(:,h)); end
w = w_init;
F_fcbpg = zeros(1, N); F_fcbpg(1) = F(w);
T_fcbpg = zeros(1, N); T_fcbpg(1) = 0;
tic
for it = 2:N
    S = w(1:k-1,:) * Z';
    for h = 1:m
        j  = nz{h};  if isempty(j), continue; end
        Sj = S(:,j);
        M  = max(0, max(Sj,[],1));
        E  = exp(Sj - M);
        Pj = E ./ (exp(-M) + sum(E,1));
        % Summed Form
        % dw = (Pj - y_b(j,1:k-1)')*Z(j,h) + lambda2*w(1:k-1,h);
        % Averaged Form
        dw = (Pj - y_b(j,1:k-1)')*Z(j,h)/n + lambda2*w(1:k-1,h);
        wold = w(1:k-1,h);
        t = wold - dw./L_feat(h);
        w(1:k-1,h) = sign(t).*max(abs(t) - lambda1./L_feat(h), 0);
        S(:,j) = Sj + (w(1:k-1,h)-wold) * Z(j,h)';
    end
    F_fcbpg(it) = F(w); T_fcbpg(it) = toc;
    if T_fcbpg(it) >= time_limit, break; end
end
w_fcbpg = w; iter_fcbpg = it;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_fcbpg(it), it, F_fcbpg(it), nnz(w(1:k-1,:)), (k-1)*m);

% === Whole  (full prox-grad / ISTA on the whole matrix; step L_full)
fprintf('Whole...\n');
w = w_init;
F_whole = zeros(1, N); F_whole(1) = F(w);
T_whole = zeros(1, N); T_whole(1) = 0;
tic
for it = 2:N
    G = logreg_grad(w, Z, y_b, lambda2);
    t = w(1:k-1,:) - G/L_full;
    w(1:k-1,:) = sign(t).*max(abs(t) - lambda1/L_full, 0);
    F_whole(it) = F(w); T_whole(it) = toc;
    if T_whole(it) >= time_limit, break; end
end
iter_whole = it; w_whole = w;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_whole(it), it, F_whole(it), nnz(w(1:k-1,:)), (k-1)*m);

% === SVRG  (BM-SVRG with a single global metric H = c·I
fprintf('SVRG...\n');
lam1 = lambda1; lam2 = lambda2;
eta  = 0.1;
step = eta / c; % Only one scalar step
beta    = 2*eta/(1-eta);
mu_H    = lam2 / c;
m_inner = floor((1/(eta*mu_H)+beta)/(1-2*beta)) + 1;
w = w_init;
F_svrg = zeros(1,N); F_svrg(1) = F(w);
T_svrg = zeros(1,N); T_svrg(1) = 0;
tic
stop = false;
for it = 2:N
    Ws    = w;
    Gsnap = (softmax_tail(Ws,Z) - Yt)*Z / n;
    for jj = 1:m_inner
        i  = randi(n);
        zi = Z(i,:);
        sW  = w(1:k-1,:)*zi';   Mw = max(0,max(sW));  pW  = exp(sW-Mw);  pW  = pW /(exp(-Mw)+sum(pW));
        sWs = Ws(1:k-1,:)*zi';  Ms = max(0,max(sWs)); pWs = exp(sWs-Ms); pWs = pWs/(exp(-Ms)+sum(pWs));
        v  = (pW - pWs)*zi + Gsnap;
        Wt = w(1:k-1,:) - step*v; % scalar step, whole matrix at once
        w(1:k-1,:) = sign(Wt).*max(abs(Wt)-step*lam1,0) ./ (1+step*lam2);
        if mod(jj,1000)==0 && toc >= time_limit, stop = true; break; end
    end
    F_svrg(it) = F(w); T_svrg(it) = toc;
    if stop || T_svrg(it) >= time_limit, break; end
end
iter_svrg = it; w_svrg = w;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz=%d/%d (m_inner=%d)\n', ...
    T_svrg(it), it, F_svrg(it), nnz(w(1:k-1,:)), (k-1)*m, m_inner);

% === SAGA  (stored per-sample gradient table; one "iteration" = one epoch)
fprintf('SAGA...\n');
w = w_init;
resid = softmax_tail(w, Z) - y_b(:,1:k-1)';
table = zeros(k-1, m, n);
for i = 1:n, table(:,:,i) = resid(:,i)*Z(i,:); end
avg  = mean(table, 3);
step = 1/(3*L_samp);
F_saga = zeros(1, N); F_saga(1) = F(w);
T_saga = zeros(1, N); T_saga(1) = 0;
tic
for it = 2:N
    stop = false;
    for s = 1:n
        j = randi(n);
        new_entry = (softmax_tail(w, Z(j,:)) - y_b(j,1:k-1)')*Z(j,:);
        dir = (new_entry - table(:,:,j)) + avg + lambda2*w(1:k-1,:);
        avg = avg + (new_entry - table(:,:,j))/n;
        table(:,:,j) = new_entry;
        t = w(1:k-1,:) - step*dir;
        w(1:k-1,:) = sign(t).*max(abs(t) - lambda1*step, 0);
        if toc >= time_limit, stop = true; break; end
    end
    F_saga(it) = F(w); T_saga(it) = toc;
    if stop || T_saga(it) >= time_limit, break; end
end
iter_saga = it; w_saga = w;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_saga(it), it, F_saga(it), nnz(w(1:k-1,:)), (k-1)*m);

% === PLOT
algs = {'BM-SVRG','C-CBPG','F-CBPG','Whole','SVRG','SAGA'};
Tc = {T_bmsvrg(1:iter_bmsvrg), T_ccbpg(1:iter_ccbpg), T_fcbpg(1:iter_fcbpg), ...
      T_whole(1:iter_whole), T_svrg(1:iter_svrg), T_saga(1:iter_saga)};
Fc = {F_bmsvrg(1:iter_bmsvrg), F_ccbpg(1:iter_ccbpg), F_fcbpg(1:iter_fcbpg), ...
      F_whole(1:iter_whole), F_svrg(1:iter_svrg), F_saga(1:iter_saga)};
styles = {'-','-.','-',':','--',':'};

keep   = ~cellfun(@isempty, Fc);
algs   = algs(keep);  Tc = Tc(keep);  Fc = Fc(keep);  styles = styles(keep);

if strcmp(x_mode, 'time')
    Xc = Tc;  xlbl = 'Time (s)';  xtag = 'time';
else
    Xc = cellfun(@(f) 1:numel(f), Fc, 'UniformOutput', false);
    xlbl = 'Iteration'; xtag = 'iter';
end
Fstar = min(cellfun(@min, Fc));

figure('Visible','off'); hold on; grid on; set(gca,'FontSize',16);
for i = 1:numel(Fc)
    plot(Xc{i}, Fc{i}, 'LineStyle', styles{i}, 'LineWidth', 2.5);
end
ylim([Fstar - 5, max(cellfun(@max, Fc)) + 5]);
xlabel(xlbl,'FontSize',20); ylabel('F','FontSize',20);
title(sprintf('Logistic %s (n=%d, k=%d, \\lambda_1=%g, \\lambda_2=%g)', ...
      strrep(dataset,'_','\_'), n, k, lambda1, lambda2), 'FontSize', 20);
fname = sprintf('logistic_%s_n%d_k%d_l1_%.0e_l2_%.0e_%s.png', ...
      dataset, n, k, lambda1, lambda2, xtag);
legend(algs);
exportgraphics(gcf, fname, 'Resolution', 300);
legend(algs);
exportgraphics(gcf, fname, 'Resolution', 300);
fprintf('Saved: %s\n', fname);

% ============================== HELPERS =====================================
function P = softmax_tail(w, Z)
    k = size(w, 1);
    S = w(1:k-1,:) * Z';
    M = max(0, max(S, [], 1));
    E = exp(S - M);
    P = E ./ (exp(-M) + sum(E, 1));
end

function G = logreg_grad(w, Z, y_b, lambda2)
    n  = size(Z, 1);
    k = size(w, 1);
    P = softmax_tail(w, Z);
    G = (P - y_b(:,1:k-1)') * Z/n + lambda2 * w(1:k-1,:);
end
