clear all;close all;
n = 20000;   % # of samples
m = 100;    % # of features
k = 10;     % # of classes
T1 = 200;      % iters of row update
% T2 = 500;       % iters of matrix update
% T3 = 0;
T4 = 100;       % iters of BFGS
T5 = 100;       % iters of LBFGS
T6 = 200;       % (*rate6=) iters of saga
rate6 = 20;     % time per rate6 iters in saga
lambda = 1;

for ss = 1:5
    randn('seed',ss);rand('seed',ss)
    
    X = rand(n,m); % samples
    y = randi(k,[n,1]); % labels
    w_init = randn(k,m);
    w_init(k,:)=0;
    
    obj1 = zeros(T1,1);
    time1 = zeros(T1,1);
    obj11 = zeros(T1,1);
    time11 = zeros(T1,1);
    % obj2 = zeros(T2,1);
    % time2 = zeros(T2,1);
    % obj3 = zeros(T3,1);
    % time3 = zeros(T3,1);
    obj4 = zeros(T4,1);
    time4 = zeros(T4,1);
    
    i=1:k;
    y_b(:,i)=(y==i);
    
    L_all = sum(X.^2,"all")./4;
    L = sum(X.^2)./4;
    L2=norm(X,2)^2/4;
    Ln = max(sum(X.^2,2)/4);
    w = w_init;
    % ours1
    tic
    for i=1:T1*rate6
        % t_total = sum(abs(w),'all');
        if mod(i,rate6)==1
            obj1(floor(i/rate6)+1) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
            time1(floor(i/rate6)+1) = toc;
        end
        tmp = randi(n);
        for h=1:m
            dw = (exp(w(1:k-1,:)*X(tmp,:)')./(1+sum(exp(w(1:k-1,:)*X(tmp,:)')))-y_b(tmp,1:k-1)')*X(tmp,h);
            % P = t_total-abs(w(h,:));
            t = w(1:k-1,h) - dw./L(h);
            % t_abs = abs(t);
            w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
        end
    end
    
    w = w_init;
    % ours2
    tic
    for i=1:T1*rate6
        % t_total = sum(abs(w),'all');
        if mod(i,rate6)==1
            obj11(floor(i/rate6)+1) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
            time11(floor(i/rate6)+1) = toc;
        end
        tmp = randi(n);
        for h=1:m
            dw = (exp(w(1:k-1,:)*X(tmp,:)')./(1+sum(exp(w(1:k-1,:)*X(tmp,:)')))-y_b(tmp,1:k-1)')*X(tmp,h);
            % P = t_total-abs(w(h,:));
            t = w(1:k-1,h) - dw./Ln;
            % t_abs = abs(t);
            w(1:k-1,h) = sign(t).*max(abs(t)-lambda./Ln,0);
        end
    end
    % w = w_init;
    % tic
    % for i=1:T2
    %     obj2(i) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
    %     time2(i) = toc;
    %     dw = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X;
    %     t = w(1:k-1,:) - dw./L2;
    %     w(1:k-1,:) = sign(t).*max(abs(t)-lambda./L2,0);
    % end
    
    opts = struct();
    opts.xtol = 0;
    opts.gtol = 0;
    opts.ftol = 0;
    opts.maxit = T5;
    opts.record  = 0;
    opts.m = 20;
    
    fun = @(w)obj(w,X,y,y_b,lambda);
    hess = @(w,u)hessian(w,X,u);
    x0 = w_init(1:k-1,:);
    x0 =x0(:);
    [x1,~,~,out1] = fminLBFGS_Loop(x0,fun,opts);
    fprintf("lbfgs");
    opts = struct();
    opts.xtol = 1e-8;
    opts.gtol = 1e-6;
    opts.ftol = 1e-16;
    opts.maxit = T5;
    opts.verbose = 0;
    [x2,out2] = fminNewton(x0,fun,hess,opts);
    fprintf("newton")
    opts = struct();
    opts.xtol = 1e-8;
    opts.gtol = 1e-6;
    opts.ftol = 1e-16;
    opts.maxit = T5;
    opts.record  = 0;
    opts.verbose = 0;
    opts.Delta = sqrt(n);
    c1 = 1e-4;
    eta = 1.5;
    c2 = .9;
    [x3,out3] = fminTR(x0,fun,hess,opts);
    fprintf("tr")
    % w = w_init;
    % H = repmat(eye(k-1),1,1,m);
    % t = ones(m,1);
    % eta = 1.5;
    % c1 = 1e-4;
    % c2 = .9;
    % tp = zeros(k,1);
    % flag = false;
    % tic
    % for i=1:T3
    %     obj3(i) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
    %     time3(i) = toc;
    %     for h=1:m
    %         dw = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X+lambda*sign(w(1:k-1,:));
    %         p = -H(:,:,h)*dw(:,h);
    %         tp(1:k-1) = p;
    %         tmp  = p'*dw;
    %         wtmp = w;
    %         step = 1;
    %         wtmp(1:k-1,h) = w(1:k-1,h)+step*p;
    %         while (trace(X*wtmp(y,:)')-sum(log(1+sum(exp(wtmp(1:k-1,:)*X'))))-lambda*sum(abs(wtmp),'all')...
    %                 <trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all')-c1*step*tmp)
    %             step=step/eta;
    %             wtmp(1:k-1,h) = w(1:k-1,h)+step*p;
    %         end
    %         dwtmp = (exp((w(1:k-1,:)+step*p)*X')./(1+sum(exp((w(1:k-1,:)+step*p)*X')))-y_b(:,1:k-1)')*X+lambda*sign(w(1:k-1,:)+step*p);
    %         while (abs(p'*dwtmp)>c2*abs(p'*dw))
    %             step=step/eta;
    %             dwtmp = (exp((w(1:k-1,:)+step*p)*X')./(1+sum(exp((w(1:k-1,:)+step*p)*X')))-y_b(:,1:k-1)')*X+lambda*sign(w(1:k-1,:)+step*p);
    %         end
    %         s = step*p;
    %         w(1:k-1,h)=w(1:k-1,h)+s;
    %         y2 = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X+lambda*sign(w(1:k-1,:));
    %         y2 = y2(:,h)-dw(:,h);
    %         if norm(y2,"fro")==0
    %             flag = true;
    %             break;
    %         end
    %         H(:,:,h) = H(:,:,h)+(s'*y2+y2'*H(:,:,h)*y2)/((s'*y2)^2)*(s*s')-(H(:,:,h)*y2*s'+s*y2'*H(:,:,h))./(s'*y2);
    %     end
    %     if flag
    %         break;
    %     end
    % end
    
    w = w_init;
    % w = w(:);
    H = eye((k-1)*m);
    flag = false;
    tic
    for i=1:T4
        % [f,dw] = obj(w,X,y,y_b,lambda);
        f = -trace(X*w(y,:)')+sum(log(1+sum(exp(w(1:k-1,:)*X'))))+lambda*sum(abs(w),'all');
        time4(i) = toc;
        obj4(i) = -f;
        g = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)')*X+lambda*sign(w(1:k-1,:));
        g = g(:);
        p = -H*g;
        tmp  = p'*g;
        step = 1;
        tmpw = w;
        tmpw(1:k-1,:)=tmpw(1:k-1,:)+step*reshape(p,k-1,m);
        f2 = -trace(X*tmpw(y,:)')+sum(log(1+sum(exp(tmpw(1:k-1,:)*X'))))+lambda*sum(abs(tmpw),'all');
        g2 = (exp(tmpw(1:k-1,:)*X')./(1+sum(exp(tmpw(1:k-1,:)*X')))-y_b(:,1:k-1)')*X+lambda*sign(tmpw(1:k-1,:));
        g2 = g2(:);
        while ((f2>f+tmp*c1*step)||(-p'*g2>-c2*tmp))
            step=step/eta;
            tmpw(1:k-1,:)=w(1:k-1,:)+step*reshape(p,k-1,m);
            f2 = -trace(X*tmpw(y,:)')+sum(log(1+sum(exp(tmpw(1:k-1,:)*X'))))+lambda*sum(abs(tmpw),'all');
            g2 = (exp(tmpw(1:k-1,:)*X')./(1+sum(exp(tmpw(1:k-1,:)*X')))-y_b(:,1:k-1)')*X+lambda*sign(tmpw(1:k-1,:));
            g2 = g2(:);
        end
        s = step*p;
        w = tmpw;
        y2 = g2-g;
        if norm(y2,"fro")==0
            flag = true;
            break;
        end
        H = H+(s'*y2+y2'*H*y2)/((s'*y2)^2)*(s*s')-(H*y2*s'+s*y2'*H)./(s'*y2);
    end
    fprintf("bfgs")
    w = w_init;
    tmp = (exp(w(1:k-1,:)*X')./(1+sum(exp(w(1:k-1,:)*X')))-y_b(:,1:k-1)');
    table = zeros(k-1,m,n);
    obj6 = zeros(T6,1);
    time6 = zeros(T6,1);
    step = 1/(3*Ln);
    for i = 1:n
        table(:,:,i) = tmp(:,i)*X(i,:);
    end
    avg = sum(table,3)./n;
    tic
    for i=1:T6*rate6
        if mod(i,rate6)==1
            obj6(floor(i/rate6)+1) = trace(X*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*X'))))-lambda*sum(abs(w),'all');
            time6(floor(i/rate6)+1) = toc;
        end
        j = randi(n);
        new_entry = (exp(w(1:k-1,:)*X(j,:)')./(1+sum(exp(w(1:k-1,:)*X(j,:)')))-y_b(j,1:k-1)');
        tmp = w(1:k-1,:) - step*(new_entry - table(:,:,j)+avg);
        avg = avg + (new_entry - table(:,:,j))./n;
        w(1:k-1,:) = sign(tmp).*max(abs(tmp)-lambda*step,0);
    end
    datafile = sprintf('data_%d.mat', ss);
    save(datafile);
    
    counter = sum(obj1(1:end-1)>obj1(2:end));
    
    fig1 = figure;
    semilogy(obj1,"LineWidth",1.5);
    hold on;
    semilogy(obj11,"LineWidth",1.5);
    %semilogy(obj2,"LineWidth",1.5);
    semilogy(-out1.f,"LineWidth",1.5);
    % semilogy(obj3,"LineWidth",1.5);
    semilogy(obj4,"LineWidth",1.5);
    semilogy(-out2.f,"LineWidth",1.5);
    semilogy(-out3.f,"LineWidth",1.5);
    semilogy(obj6,"LineWidth",1.5);
    hold off;
    legend("ours","ours2","LBFGS","BFGS","NewtonConjugate","TrustRegion","SAGA");
    title("Iteration")
    iterfig = sprintf('iter_%d.fig', ss);
    savefig(fig1,iterfig);
    close(fig1)
    
    fig2 = figure;
    semilogy(time1,obj1,"LineWidth",1.5);
    hold on;
    semilogy(time11,obj11,"LineWidth",1.5);
    %semilogy(time2,obj2,"LineWidth",1.5);
    semilogy(out1.time,-out1.f,"LineWidth",1.5);
    % semilogy(time3,obj3,"LineWidth",1.5);
    semilogy(time4,obj4,"LineWidth",1.5);
    semilogy(out2.time,-out2.f,"LineWidth",1.5);
    semilogy(out3.time,-out3.f,"LineWidth",1.5);
    semilogy(time6,obj6,"LineWidth",1.5);
    hold off;
    legend("ours","ours2","LBFGS","BFGS","NewtonConjugate","TrustRegion","SAGA");
    title("Time")
    timefig = sprintf('time_%d.fig', ss);
    savefig(fig2,timefig);
    close(fig2);
    
end
function [f,g] = obj(w,X,y,y_b,lambda)
m = size(X,2);
w_m = [reshape(w,[],m);zeros(1,m)];
k = size(w_m,1);
f = -trace(X*w_m(y,:)')+sum(log(1+sum(exp(w_m*X'))))+lambda*sum(abs(w),'all');
g = (exp(w_m(1:k-1,:)*X')./(1+sum(exp(w_m(1:k-1,:)*X')))-y_b(:,1:k-1)')*X;
g = g(:)+lambda*sign(w);
end

function result = hessian(w,X,u)
m = size(X,2);
w_m = [reshape(w,[],m)];
k = size(w_m,1);
exp_sum = (1+sum(exp(w_m*X')));
h = zeros(m*k);
for i = 1:k
    for j = 1:k
        if i==j
            tmp = (exp(w_m(i,:)*X')./exp_sum)';
            h((i-1)*m+1:i*m,(i-1)*m+1:i*m)=X'*(tmp.*(1-tmp).*X);
        else
            h((i-1)*m+1:i*m,(j-1)*m+1:j*m) = X'*((exp(w_m(i,:)*X').*exp(w_m(j,:)*X')./(exp_sum.^2))'.*X);
        end
    end
end
result = h*u;
end
