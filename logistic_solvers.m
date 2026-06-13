% logistic_solvers.m
% Minimal L1+L2 multinomial logistic-regression solver comparison.
% Solvers ported from the trusted-author repo (mlp_*.m).
% Expects Z, y, y_b, n, m, k, lambda, N, time_limit, x_mode, dataset, F,
% and L_feat, L_spec, L_full, L_samp to be in the workspace (run load_logistic).
% time_limit (seconds) caps wall-clock time per algorithm; N is a safety cap.

w_init = randn(k, m); w_init(k,:) = 0;      % shared start; reference row = 0

% === C-CBPG
% Cyclic block proximal gradient over CLASS rows; spectral step L_spec.
fprintf('C-CBPG...\n');
w = w_init;
F_ccbpg = zeros(1, N); F_ccbpg(1) = F(w);
T_ccbpg = zeros(1, N); T_ccbpg(1) = 0;
tic
for it = 2:N
    for h = 1:k-1
        G = logreg_grad(w, Z, y_b, lambda);
        t = w(h,:) - G(h,:)/L_spec;
        w(h,:) = sign(t).*max(abs(t) - lambda/L_spec, 0);
    end
    F_ccbpg(it) = F(w); T_ccbpg(it) = toc;
    if T_ccbpg(it) >= time_limit, break; end
end
iter_ccbpg = it;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_ccbpg(it), it, F_ccbpg(it), nnz(w(1:k-1,:)), (k-1)*m);

% === F-CBPG
% Cyclic block proximal gradient over FEATURE columns; per-feature step L_feat.
fprintf('F-CBPG...\n');
w = w_init;
F_fcbpg = zeros(1, N); F_fcbpg(1) = F(w);
T_fcbpg = zeros(1, N); T_fcbpg(1) = 0;
tic
for it = 2:N
    for h = 1:m
        P  = softmax_tail(w, Z);
        dw = (P - y_b(:,1:k-1)')*Z(:,h) + lambda*w(1:k-1,h);
        t  = w(1:k-1,h) - dw/L_feat(h);
        w(1:k-1,h) = sign(t).*max(abs(t) - lambda/L_feat(h), 0);
    end
    F_fcbpg(it) = F(w); T_fcbpg(it) = toc;
    if T_fcbpg(it) >= time_limit, break; end
end
iter_fcbpg = it;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_fcbpg(it), it, F_fcbpg(it), nnz(w(1:k-1,:)), (k-1)*m);

% === Whole
% Full proximal gradient (ISTA) on the whole matrix; step L_full.
fprintf('Whole...\n');
w = w_init;
F_whole = zeros(1, N); F_whole(1) = F(w);
T_whole = zeros(1, N); T_whole(1) = 0;
tic
for it = 2:N
    G = logreg_grad(w, Z, y_b, lambda);
    t = w(1:k-1,:) - G/L_full;
    w(1:k-1,:) = sign(t).*max(abs(t) - lambda/L_full, 0);
    F_whole(it) = F(w); T_whole(it) = toc;
    if T_whole(it) >= time_limit, break; end
end
iter_whole = it;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_whole(it), it, F_whole(it), nnz(w(1:k-1,:)), (k-1)*m);

% === SVRG
% Prox-SVRG: full-gradient snapshot per epoch + many variance-reduced steps.
fprintf('SVRG...\n');
w = w_init; wt = w_init;
step  = 0.1/L_samp;
inner = ceil(100*L_samp/(lambda/n));
F_svrg = zeros(1, N); F_svrg(1) = F(w);
T_svrg = zeros(1, N); T_svrg(1) = 0;
tic
for it = 2:N
    w  = wt;
    v  = logreg_grad(w, Z, y_b, 0)/n + lambda*w(1:k-1,:)/n;
    wt = w; wavg = zeros(size(w)); stop = false;
    for j = 1:inner
        rt = randi(n);
        f1 = (softmax_tail(w,  Z(rt,:)) - y_b(rt,1:k-1)')*Z(rt,:) + lambda*w(1:k-1,:)/n;
        f2 = (softmax_tail(wt, Z(rt,:)) - y_b(rt,1:k-1)')*Z(rt,:) + lambda*wt(1:k-1,:)/n;
        vk = (f1 - f2) + v;
        wp = w(1:k-1,:) - step*vk;
        w(1:k-1,:) = sign(wp).*max(abs(wp) - lambda*step/n, 0);
        wavg = (wavg*(j-1) + w)/j;
        if toc >= time_limit, stop = true; break; end
    end
    wt(1:k-1,:) = wavg(1:k-1,:);
    F_svrg(it) = F(w); T_svrg(it) = toc;
    if stop || T_svrg(it) >= time_limit, break; end
end
iter_svrg = it;
fprintf('  done in %.1fs at iter %d, F=%.4e (inner=%d)\n', ...
    T_svrg(it), it, F_svrg(it), inner);

% === SAGA
% SAGA with a stored per-sample gradient table; one "iteration" = one epoch.
% table is (k-1) x m x n, may struggle on large datasets
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
        new_entry = (softmax_tail(w, Z(j,:)) - y_b(j,1:k-1)')*Z(j,:);   % (k-1) x m
        dir = (new_entry - table(:,:,j)) + avg + lambda*w(1:k-1,:);
        avg = avg + (new_entry - table(:,:,j))/n;
        table(:,:,j) = new_entry;
        t = w(1:k-1,:) - step*dir;
        w(1:k-1,:) = sign(t).*max(abs(t) - lambda*step, 0);
        if toc >= time_limit, stop = true; break; end
    end
    F_saga(it) = F(w); T_saga(it) = toc;
    if stop || T_saga(it) >= time_limit, break; end
end
iter_saga = it;
fprintf('  done in %.1fs at iter %d, F=%.4e, nnz(w)=%d/%d\n', ...
    T_saga(it), it, F_saga(it), nnz(w(1:k-1,:)), (k-1)*m);

% === PLOT
algs = {'C-CBPG','F-CBPG','Whole','SVRG','SAGA'};
Tc = {T_ccbpg(1:iter_ccbpg), T_fcbpg(1:iter_fcbpg), T_whole(1:iter_whole), ...
      T_svrg(1:iter_svrg),   T_saga(1:iter_saga)};
Fc = {F_ccbpg(1:iter_ccbpg), F_fcbpg(1:iter_fcbpg), F_whole(1:iter_whole), ...
      F_svrg(1:iter_svrg),   F_saga(1:iter_saga)};

if strcmp(x_mode, 'time')
    Xc = Tc;  xlbl = 'Time (s)';  xtag = 'time';
else
    Xc = cellfun(@(f) 1:numel(f), Fc, 'UniformOutput', false);
    xlbl = 'Iteration'; xtag = 'iter';
end

Fstar  = min(cellfun(@min, Fc));
styles = {'-',':','-.','--',':'};

% % START === log(F - F*) by xtag
% figure('Visible','off'); hold on; grid on; set(gca,'FontSize',16);
% for i = 1:numel(Fc)
%     semilogy(Xc{i}, Fc{i} - Fstar, 'LineStyle', styles{i}, 'LineWidth', 2.5);
% end
% xlabel(xlbl,'FontSize',20); ylabel('F - F^\ast','FontSize',20);
% title(sprintf('Logistic %s (n=%d, k=%d, \\lambda=%g)', ...
%       strrep(dataset,'_','\_'), n, k, lambda), 'FontSize', 20);
% legend(algs);
% fname = sprintf('logistic_%s_n%d_k%d_lam%.2f_subopt_%s.png', dataset, n, k, lambda, xtag);
% % END === log(F - F*) by xtag

% START === F by xtag
figure('Visible','off'); hold on; grid on; set(gca,'FontSize',16);
for i = 1:numel(Fc)
    plot(Xc{i}, Fc{i}, 'LineStyle', styles{i}, 'LineWidth', 2.5);
end
ylim([Fstar - 5, max(cellfun(@max, Fc)) + 5]);
xlabel(xlbl,'FontSize',20); ylabel('F','FontSize',20);
title(sprintf('Logistic %s (n=%d, k=%d, \\lambda=%g)', ...
      strrep(dataset,'_','\_'), n, k, lambda), 'FontSize', 20);
legend(algs);
fname = sprintf('logistic_%s_n%d_k%d_lam%.2f_%s.png', dataset, n, k, lambda, xtag);
% END === F by xtag

exportgraphics(gcf, fname, 'Resolution', 300);
legend(algs);
exportgraphics(gcf, fname, 'Resolution', 300);
fprintf('Saved: %s\n', fname);


% ============================== HELPERS =====================================
function P = softmax_tail(w, Z)
    % Softmax probabilities for the first k-1 classes: (k-1) x size(Z,1).
    k = size(w, 1);
    E = exp(w(1:k-1,:) * Z');
    P = E ./ (1 + sum(E, 1));
end

function G = logreg_grad(w, Z, y_b, lambda)
    % Smooth-part gradient (log-likelihood + L2) w.r.t. w(1:k-1,:): (k-1) x m.
    k = size(w, 1);
    P = softmax_tail(w, Z);
    G = (P - y_b(:,1:k-1)') * Z + lambda * w(1:k-1,:);
end
