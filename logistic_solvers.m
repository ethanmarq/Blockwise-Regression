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
for it = 2:N % t = 0 .. T, inplace eval
    for h = 1:m % i = 1 .. d
        P  = softmax_tail(w, Z); % Gradient at W_{t,i-1}
        dw = (P - y_b(:,1:k-1)')*Z(:,h) + lambda*w(1:k-1,h); % ∇_i f(W_{t,i-1})
        t  = w(1:k-1,h) - dw/L_feat(h); % W^i - (1/L_i) * ∇_i f
        w(1:k-1,h) = sign(t).*max(abs(t) - lambda/L_feat(h), 0); % prox_{g_i/L_i}, U_i dispears with inplace indexing
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
% table is (k-1) x m x n, may overflow
fprintf('SAGA...\n');
w = w_init;
resid = softmax_tail(w, Z) - y_b(:,1:k-1)';   % (k-1) x n residuals
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

% --- shared smooth oracle for the second-order methods (Option A) ----------
% They optimize the SAME F as the prox methods, treating the L1 term via its
% subgradient (lambda*sign(w)); the Hessian uses only the smooth (logistic+L2)
% part. So they log the true F but won't drive weights to exact zeros.
objfun = @(wv)    obj(wv, Z, y_b, k, lambda);     % [f, g, P] = objfun(wvec)
hvpfun = @(P, u)  hvp(P, u, Z, lambda);           % Hessian-vector product
x0     = w_init(1:k-1, :); x0 = x0(:);            % optimize the free block only
cgmax  = 100;                                      % inner CG iterations

% === L-BFGS
% Limited-memory BFGS (two-loop recursion, O(d*mem) memory); L1 via subgradient.
fprintf('L-BFGS...\n');
mem = 10; c1 = 1e-4; eta = 2; maxbt = 40; d = numel(x0);
x = x0; [f, g] = objfun(x);
Sm = zeros(d,mem); Ym = zeros(d,mem); rhom = zeros(1,mem); cnt = 0;
F_lbfgs = zeros(1,N); F_lbfgs(1) = f;
T_lbfgs = zeros(1,N); T_lbfgs(1) = 0;
tic
for it = 2:N
    q = g; al = zeros(1,cnt);                          % two-loop recursion
    for i = cnt:-1:1
        al(i) = rhom(i)*(Sm(:,i)'*q);  q = q - al(i)*Ym(:,i);
    end
    if cnt > 0, gamma = (Sm(:,cnt)'*Ym(:,cnt))/(Ym(:,cnt)'*Ym(:,cnt)); else, gamma = 1; end
    r = gamma*q;
    for i = 1:cnt
        be = rhom(i)*(Ym(:,i)'*r);  r = r + Sm(:,i)*(al(i) - be);
    end
    p = -r;
    [xn, fn, gn] = armijo(objfun, x, f, g, p, c1, eta, maxbt);
    s = xn - x; yv = gn - g; sy = s'*yv;
    if sy > 1e-12                                       % keep curvature positive
        if cnt < mem, cnt = cnt + 1;
        else
            Sm = [Sm(:,2:end), zeros(d,1)];  Ym = [Ym(:,2:end), zeros(d,1)];
            rhom = [rhom(2:end), 0];
        end
        Sm(:,cnt) = s; Ym(:,cnt) = yv; rhom(cnt) = 1/sy;
    end
    x = xn; f = fn; g = gn;
    F_lbfgs(it) = f; T_lbfgs(it) = toc;
    if T_lbfgs(it) >= time_limit, break; end
end
iter_lbfgs = it;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_lbfgs(it), iter_lbfgs, F_lbfgs(it));

% === BFGS
% Dense inverse-Hessian BFGS; O(d^2) memory -- small problems only.
fprintf('BFGS...\n');
c1 = 1e-4; eta = 2; maxbt = 40; d = numel(x0);
if d > 4000
    warning('bfgs:mem', 'Dense BFGS uses a %dx%d inverse Hessian (~%.1f GB); prefer L-BFGS.', ...
        d, d, 8*d^2/1e9);
end
x = x0; [f, g] = objfun(x);
Hi = eye(d);
F_bfgs = zeros(1,N); F_bfgs(1) = f;
T_bfgs = zeros(1,N); T_bfgs(1) = 0;
tic
for it = 2:N
    p = -Hi*g;
    [xn, fn, gn] = armijo(objfun, x, f, g, p, c1, eta, maxbt);
    s = xn - x; yv = gn - g; sy = s'*yv;
    if sy > 1e-12
        Hy = Hi*yv;
        Hi = Hi + ((sy + yv'*Hy)/sy^2)*(s*s') - (Hy*s' + s*Hy')/sy;
    end
    x = xn; f = fn; g = gn;
    F_bfgs(it) = f; T_bfgs(it) = toc;
    if T_bfgs(it) >= time_limit, break; end
end
iter_bfgs = it;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_bfgs(it), iter_bfgs, F_bfgs(it));

% === Newton-CG
% Truncated Newton; Newton system solved matrix-free by CG via hvp.
fprintf('Newton-CG...\n');
c1 = 1e-4; eta = 2; maxbt = 40;
x = x0; [f, g, P] = objfun(x);
F_ncg = zeros(1,N); F_ncg(1) = f;
T_ncg = zeros(1,N); T_ncg(1) = 0;
tic
for it = 2:N
    p = newton_cg(hvpfun, P, g, cgmax);                % approx solve Hp = -g
    [xn, fn, gn, Pn] = armijo(objfun, x, f, g, p, c1, eta, maxbt);
    x = xn; f = fn; g = gn; P = Pn;
    F_ncg(it) = f; T_ncg(it) = toc;
    if T_ncg(it) >= time_limit, break; end
end
iter_ncg = it;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_ncg(it), iter_ncg, F_ncg(it));

% === Trust-Region
% Steihaug-CG subproblem with radius adaptation.
fprintf('Trust-Region...\n');
x = x0; [f, g, P] = objfun(x);
Delta = 1; Dmax = 1e3; eta_acc = 0.1;
F_tr = zeros(1,N); F_tr(1) = f;
T_tr = zeros(1,N); T_tr(1) = 0;
tic
for it = 2:N
    p  = steihaug(hvpfun, P, g, Delta, cgmax);
    [ft, gt, Pt] = objfun(x + p);
    Hp   = hvpfun(P, p);
    pred = -(g'*p + 0.5*(p'*Hp));                      % predicted decrease
    ared = f - ft;                                     % actual decrease
    rho  = ared / max(pred, 1e-16);
    if rho < 0.25
        Delta = 0.25*Delta;
    elseif rho > 0.75 && abs(norm(p) - Delta) < 1e-8*Delta
        Delta = min(2*Delta, Dmax);
    end
    if rho > eta_acc                                   % accept step
        x = x + p; f = ft; g = gt; P = Pt;
    end
    F_tr(it) = f; T_tr(it) = toc;
    if T_tr(it) >= time_limit, break; end
end
iter_tr = it;
fprintf('  done in %.1fs at iter %d, F=%.4e\n', T_tr(it), iter_tr, F_tr(it));

% === PLOT
algs = {'C-CBPG','F-CBPG','Whole','SVRG','SAGA','L-BFGS','BFGS','Newton-CG','Trust-Region'};
Tc = {T_ccbpg(1:iter_ccbpg), T_fcbpg(1:iter_fcbpg), T_whole(1:iter_whole), ...
      T_svrg(1:iter_svrg),   T_saga(1:iter_saga), ...
      T_lbfgs(1:iter_lbfgs), T_bfgs(1:iter_bfgs), T_ncg(1:iter_ncg), T_tr(1:iter_tr)};
Fc = {F_ccbpg(1:iter_ccbpg), F_fcbpg(1:iter_fcbpg), F_whole(1:iter_whole), ...
      F_svrg(1:iter_svrg),   F_saga(1:iter_saga), ...
      F_lbfgs(1:iter_lbfgs), F_bfgs(1:iter_bfgs), F_ncg(1:iter_ncg), F_tr(1:iter_tr)};

if strcmp(x_mode, 'time')
    Xc = Tc;  xlbl = 'Time (s)';  xtag = 'time';
else
    Xc = cellfun(@(f) 1:numel(f), Fc, 'UniformOutput', false);
    xlbl = 'Iteration'; xtag = 'iter';
end

Fstar  = min(cellfun(@min, Fc));
styles = {'-',':','-.','--',':','-','--','-.',':'};

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
    % Stable softmax probabilities for the first k-1 classes: (k-1) x size(Z,1).
    k = size(w, 1);
    S = w(1:k-1,:) * Z';
    M = max(0, max(S, [], 1));                 % include reference logit (=0)
    E = exp(S - M);
    P = E ./ (exp(-M) + sum(E, 1));
end

function G = logreg_grad(w, Z, y_b, lambda)
    % Smooth-part gradient (log-likelihood + L2) w.r.t. w(1:k-1,:): (k-1) x m.
    k = size(w, 1);
    P = softmax_tail(w, Z);
    G = (P - y_b(:,1:k-1)') * Z + lambda * w(1:k-1,:);
end

% ----- smooth oracle on the free block (vectorized), matching loader F ------
function [f, g, P] = obj(wv, Z, y_b, k, lambda)
    m = size(Z, 2);
    W = reshape(wv, k-1, m);
    S = W * Z';                                % (k-1) x n logits
    M = max(0, max(S, [], 1));                 % stable log-sum-exp shift
    E = exp(S - M);
    denom = exp(-M) + sum(E, 1);               % = sum over all k classes
    P = E ./ denom;                            % (k-1) x n tail probabilities
    Yt = y_b(:, 1:k-1)';
    lin     = sum(sum(S .* Yt));               % sum_i s_{i,y_i} (0 if reference)
    logpart = sum(M + log(denom));             % stable sum log(1+sum exp S)
    f = -lin + logpart + lambda*sum(abs(wv)) + 0.5*lambda*sum(wv.^2);
    if nargout > 1
        G = (P - Yt)*Z + lambda*W + lambda*sign(W);   % L1 via subgradient
        g = G(:);
    end
end

function Hu = hvp(P, u, Z, lambda)
    % Hessian-vector product of the smooth part (logistic + L2). L1 adds nothing.
    km1 = size(P, 1);
    m   = size(Z, 2);
    U  = reshape(u, km1, m);
    A  = U * Z';                               % (k-1) x n
    PA = P .* A;
    Q  = PA - P .* sum(PA, 1);                 % (diag(p)-pp')a per sample
    H  = Q * Z + lambda * U;
    Hu = H(:);
end

% ----- Armijo backtracking line search (objfun returns [f,g,P]) -------------
function [xn, fn, gn, Pn] = armijo(objfun, x, f, g, p, c1, eta, maxbt)
    gtp = g' * p; step = 1;
    [fn, gn, Pn] = objfun(x + step*p);
    bt = 0;
    while fn > f + c1*step*gtp && bt < maxbt
        step = step/eta;
        [fn, gn, Pn] = objfun(x + step*p);
        bt = bt + 1;
    end
    xn = x + step*p;
end

% ----- inner CG solve for Newton-CG -----------------------------------------
function p = newton_cg(hvpfun, P, g, cgmax)
    p = zeros(numel(g),1); r = g; dvec = -r; rs = r'*r;
    tol = min(0.5, sqrt(norm(g))) * norm(g);    % forcing sequence
    for j = 1:cgmax
        Hd = hvpfun(P, dvec); dHd = dvec'*Hd;
        if dHd <= 1e-12                          % nonpositive curvature
            if j == 1, p = -g; end
            return;
        end
        a = rs/dHd;
        p = p + a*dvec;
        r = r + a*Hd;
        rs_new = r'*r;
        if sqrt(rs_new) <= tol, return; end
        dvec = -r + (rs_new/rs)*dvec;
        rs = rs_new;
    end
end

% ----- Steihaug-CG trust-region subproblem ----------------------------------
function p = steihaug(hvpfun, P, g, Delta, cgmax)
    z = zeros(numel(g),1); r = g; dvec = -r; rs = r'*r;
    g0 = sqrt(rs);
    if g0 < 1e-12, p = z; return; end
    for j = 1:cgmax
        Hd = hvpfun(P, dvec); dHd = dvec'*Hd;
        if dHd <= 0                               % negative curvature -> boundary
            p = z + boundary_tau(z, dvec, Delta)*dvec; return;
        end
        a = rs/dHd;
        znew = z + a*dvec;
        if norm(znew) >= Delta                    % crossed boundary
            p = z + boundary_tau(z, dvec, Delta)*dvec; return;
        end
        r = r + a*Hd; rs_new = r'*r;
        if sqrt(rs_new) <= 1e-6*g0, p = znew; return; end
        dvec = -r + (rs_new/rs)*dvec; z = znew; rs = rs_new;
    end
    p = z;
end

function tau = boundary_tau(z, d, Delta)
    % Largest tau >= 0 with ||z + tau*d|| = Delta.
    a = d'*d; b = 2*(z'*d); c = z'*z - Delta^2;
    tau = (-b + sqrt(b^2 - 4*a*c)) / (2*a);
end
