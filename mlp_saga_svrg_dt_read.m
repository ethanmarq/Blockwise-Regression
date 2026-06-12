clearvars;close all;
% datafolder = "data/mat/";
% resultfolder = "data/results/";
% list = dir(strcat(datafolder,"*.mat"));
% 
% for ss=1:length(list)
    %load(strcat(datafolder,list(ss).name));
    load('/home/kai/Downloads/data/data/mat/segment.mat')
    %Z=(Z-mean(Z))./std(Z);
    [n,m] = size(Z);
    max_num_of_data=15000;
    if n > max_num_of_data
        selected = randperm(n,max_num_of_data);
        Z = Z(selected,:);
        y = y(selected,:);
        n = max_num_of_data;
    end
    if min(y)<1
        y = y+1;
    end
    k = max(y);
    T1 = 200;      % iters of row update
    T6 = 30;       % (*rate6=) iters of saga
    rate6 = 1000;     % time per rate6 iters in saga
    lambda = 1;
    T_svrg = 3;
    T_sarah= 30;
    %start_num = 10;
    
    w_init = randn(k,m);
    w_init(k,:) = 0;
    
    obj1 = zeros(T1,1);
    time1 = zeros(T1,1);
    obj11 = zeros(T1,1);
    time11 = zeros(T1,1);


    
    y_b = zeros(n,k);
    for i=1:k
        y_b(:,i)=(y==i);
    end
    %L_all = sum(Z.^2,"all")./4+lambda;
    L = sum(Z.^2)./2+lambda;
    L2=norm(Z,2)^2/4+lambda;
    %Ln = (norm(Z,'fro')^2/4+lambda)/n;
    Ln = (norm(Z)^2/2+lambda)/n;
    Lf=Ln*n;
    %% stochastic coordinate descent with individual Lipschitz constants
%     w = w_init;
%     tic
%     for i=1:T1*rate6
%         % t_total = sum(abs(w),'all');
%         if mod(i,rate6)==1
%             obj1(floor(i/rate6)+1) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
%             time1(floor(i/rate6)+1) = toc;
%         end
%         tmp = randi(n);
%         for h=1:m
%             dw = (exp(w(1:k-1,:)*Z(tmp,:)')./(1+sum(exp(w(1:k-1,:)*Z(tmp,:)')))-y_b(tmp,1:k-1)')*Z(tmp,h)+lambda*w(1:k-1,h);
%             % P = t_total-abs(w(h,:));
%             t = w(1:k-1,h) - dw./L(h);
%             % t_abs = abs(t);
%             w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
%         end
%     end
    
    %% stochastic coordinate descent with global Lipschitz constant
%     w = w_init;
%     tic
%     for i=1:T1*rate6
%         % t_total = sum(abs(w),'all');
%         if mod(i,rate6)==1
%             obj11(floor(i/rate6)+1) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
%             time11(floor(i/rate6)+1) = toc;
%         end
%         tmp = randi(n);
%         for h=1:m
%             dw = (exp(w(1:k-1,:)*Z(tmp,:)')./(1+sum(exp(w(1:k-1,:)*Z(tmp,:)')))-y_b(tmp,1:k-1)')*Z(tmp,h)+lambda*w(1:k-1,h);
%             % P = t_total-abs(w(h,:));
%             t = w(1:k-1,h) - dw./Ln;
%             % t_abs = abs(t);
%             w(1:k-1,h) = sign(t).*max(abs(t)-lambda./Ln,0);
%         end
%     end
    
    %% class update
    w = w_init;
    time_class = zeros(100*T1,1);
    obj_class = zeros(100*T1,1);
    tic
    for i=1:T1*100
        % t_total = sum(abs(w),'all');
 %obj_svrg(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;

        obj_class(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
        time_class(i) = toc;
        for h=1:k-1
            dd = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)')*Z+lambda*w(1:k-1,:);
            % P = t_total-abs(w(h,:));
            dw=dd(h,:);
            t = w(h,:) - dw./L2;
            % t_abs = abs(t);
            w(h,:) = sign(t).*max(abs(t)-lambda./L2,0);
        end
        
    end
    %% feature update
    w = w_init;
    time_det_indie = zeros(T1,1);
    obj_det_indie = zeros(T1,1);
    tic
    for i=1:T1
        % t_total = sum(abs(w),'all');
 %obj_svrg(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;

        obj_det_indie(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
        time_det_indie(i) = toc;
        for h=1:m
            dw = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)')*Z(:,h)+lambda*w(1:k-1,h);
            % P = t_total-abs(w(h,:));
            t = w(1:k-1,h) - dw./L(h);
            % t_abs = abs(t);
            w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
        end
        
    end
    
    %% whole update
    w = w_init;
    time_whole = zeros(100*T1,1);
    obj_whole = zeros(100*T1,1);
    tic
    for i=1:T1*100
        % t_total = sum(abs(w),'all');
 %obj_svrg(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;

        obj_whole(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
        time_whole(i) = toc;
        %for h=1:m
            dd = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)')*Z+lambda*w(1:k-1,:);
            % P = t_total-abs(w(h,:));
            t = w(1:k-1,:) - dd./Lf;
            % t_abs = abs(t);
            w(1:k-1,:) = sign(t).*max(abs(t)-lambda./Lf,0);
        %end
        
    end
    %% deterministic coordinate descent with global Lipschitz constant
%     w = w_init;
%     time_det_global = zeros(T1,1);
%     obj_det_global = zeros(T1,1);
%     tic
%     for i=1:T1
%         % t_total = sum(abs(w),'all');
%         obj_det_global(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
%         time_det_global(i) = toc;
%         h = randi(m);
%         %for h=1:m
%             dw = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)')*Z(:,h)+lambda*w(1:k-1,h);
%             % P = t_total-abs(w(h,:));
%             t = w(1:k-1,h) - dw./L(h);
%             % t_abs = abs(t);
%             w(1:k-1,h) = sign(t).*max(abs(t)-lambda./L(h),0);
%         %end
%     end
% 
%     %% SAGA
%      LL= max(sum(Z.^2,2)/2)+lambda/n;
%     w = w_init;
%      tmp = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)');
%      table = zeros(k-1,m,n);
%      obj6 = zeros(T6,1);
%      time6 = zeros(T6,1);
%      step = 1/(3*LL);
%      for i = 1:n
%          table(:,:,i) = tmp(:,i)*Z(i,:);
%      end
%      avg = sum(table,3)./n;
%      tic
%      for i=1:T6*rate6
%          if mod(i,rate6)==1
%              obj6(floor(i/rate6)+1) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
%              time6(floor(i/rate6)+1) = toc;
%          end
%          j = randi([1,n]);
%          new_entry = (exp(w(1:k-1,:)*Z(j,:)')./(1+sum(exp(w(1:k-1,:)*Z(j,:)')))-y_b(j,1:k-1)')+lambda*w(1:k-1,:);
%          tmp = w(1:k-1,:) - step*(new_entry - table(:,:,j)+avg);
%          avg = avg + (new_entry - table(:,:,j))./n;
%          w(1:k-1,:) = sign(tmp).*max(abs(tmp)-lambda*step/n,0);
%          w(1:k-1,:) = sign(wprev).*max(abs(wprev)-lambda*step/n,0);

%table(:,:,j)=new_entry;
%      end

    %% SVRG
    Ln = max(sum(Z.^2,2)/2)+lambda/n;
    wt = w_init;
    % step = 0.1/Ln;
    % inner = 1;
    % while 1/(step*(1-4*Ln*step)*inner)+4*Ln*(inner+1)/(1-4*Ln*step)/inner>=1
    %     inner = inner*2;
    % end
    step = 0.1/Ln;
    mu=lambda/n;
    inner = ceil(100*Ln/mu);
    obj_svrg = zeros(T_svrg,1);
    time_svrg = zeros(T_svrg,1);
    tic
    for i=1:T_svrg
        w=wt;
        obj_svrg(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
        time_svrg(i) = toc;
        v = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)')*Z;
        v=v/n+lambda*w(1:k-1,:)/n;
        wt = w;
        wavg=zeros(size(w));
        %wprev = wt;
        %wavg = wt;
        for j=1:inner
            rt = randi(n);
            f1 = ((exp(w(1:k-1,:)*Z(rt,:)')./(1+sum(exp(w(1:k-1,:)*Z(rt,:)')))-y_b(rt,1:k-1)')*Z(rt,:))+lambda*w(1:k-1,:)/n;
            f2 = ((exp(wt(1:k-1,:)*Z(rt,:)')./(1+sum(exp(wt(1:k-1,:)*Z(rt,:)')))-y_b(rt,1:k-1)')*Z(rt,:))+lambda*wt(1:k-1,:)/n;
            vk = (f1-f2)+v;
            wprev=w(1:k-1,:)-step*vk;
            w(1:k-1,:) = sign(wprev).*max(abs(wprev)-lambda*step/n,0);
            %wprev = wk;
            wavg = (wavg*(j-1)+w)/j;
        end
        wt(1:k-1,:) = wavg(1:k-1,:);
        
    end

    %% SARAH
%     w = w_init;
%     gamma = 0.125;
%     inner = ceil(2*L2-1);
%     step = sqrt(2/L2/(inner+1));
%     obj_sarah = zeros(T_sarah,1);
%     time_sarah = zeros(T_sarah,1);
%     tic
%     for i=1:T_sarah
%         obj_sarah(i) = trace(Z*w(y,:)')-sum(log(1+sum(exp(w(1:k-1,:)*Z'))))-lambda*sum(abs(w),'all')-lambda*sum(w.^2,"all")/2;
%         time_sarah(i) = toc;
%         v = (exp(w(1:k-1,:)*Z')./(1+sum(exp(w(1:k-1,:)*Z')))-y_b(:,1:k-1)')*Z+lambda*sign(w(1:k-1,:))+lambda*w(1:k-1,:);
%         thresh =sum(v.^2,'all')*gamma;
%         wprev = w;
%         wt = wprev;
%         wt(1:k-1,:) = wt(1:k-1,:) - step*v;
%         for j=2:inner
%             rt = randi(n);
%             f1 = ((exp(wt(1:k-1,:)*Z(rt,:)')./(1+sum(exp(wt(1:k-1,:)*Z(rt,:)')))-y_b(rt,1:k-1)')*Z(rt,:))*n+lambda*sign(wt(1:k-1,:))+lambda*wt(1:k-1,:);
%             f2 = ((exp(wprev(1:k-1,:)*Z(rt,:)')./(1+sum(exp(wprev(1:k-1,:)*Z(rt,:)')))-y_b(rt,1:k-1)')*Z(rt,:))*n+lambda*sign(wprev(1:k-1,:))+lambda*wprev(1:k-1,:);
%             v = (f1-f2)+v;
%             % wk = sign(wprev).*max(abs(wprev)-lambda*step,0);
%             wprev = wt;
%             wt(1:k-1,:) = wprev(1:k-1,:)-step*v(1:k-1,:);
%             if sum(v.^2,"all")<=thresh
%                 break
%             end
%         end
%         w(1:k-1,:) = wt(1:k-1,:);
%     end


    %% save data & plot
    %save(strcat(resultfolder,list(ss).name));
    
    %counter = sum(obj1(1:end-1)>obj1(2:end));
    
    %fig1 = figure;
%     semilogy(obj1,"LineWidth",1.5,"LineStyle","--","Marker","+",'MarkerIndices', 1:10:length(obj1));
%     hold on;
%     semilogy(obj11,"LineWidth",1.5,"Marker","o",'MarkerIndices', 1:10:length(obj11));
%     semilogy(obj_det_indie,"LineWidth",1.5,"LineStyle","--","Marker","x",'MarkerIndices', 1:10:length(obj_det_indie));
%     semilogy(obj_det_global,"LineWidth",1.5,"Marker","square",'MarkerIndices', 1:10:length(obj_det_global));
%     semilogy(obj6,"LineWidth",1.5,"LineStyle","-.","Marker","*",'MarkerIndices', 1:10:length(obj6));
%     semilogy(obj_svrg,"LineWidth",1.5,"LineStyle","-.","Marker","diamond",'MarkerIndices', 1:10:length(obj_svrg));
%     %semilogy(obj_sarah,"LineWidth",1.5,"LineStyle","--","Marker","v",'MarkerIndices', 1:10:length(obj_sarah));
%     hold off;
%     legend("StoBlock","StoGlobal","DetBlock","DetGlobal","SAGA","SVRG");
%     title("Iteration")
%     iterfig = sprintf('%siter_%s.fig', resultfolder,list(ss).name);
%     savefig(fig1,iterfig);
    
%     fig2 = figure;
%     semilogy(time1,obj1,"LineWidth",1.5,"LineStyle","--","Marker","+",'MarkerIndices', 1:10:length(obj1));
%     hold on;
%     semilogy(time11,obj11,"LineWidth",1.5,"Marker","o","MarkerIndices", 1:10:length(obj11));
%     semilogy(time_det_indie,-obj_det_indie,"LineWidth",1.5,"LineStyle","--","Marker","x",'MarkerIndices', 1:10:length(obj_det_indie));
%     hold on;
%     semilogy(time_det_global,-obj_det_global,"LineWidth",1.5,"Marker","square",'MarkerIndices', 1:10:length(obj_det_global));
%     semilogy(time6,-obj6,"LineWidth",1.5,"LineStyle","-.","Marker","*",'MarkerIndices', 1:10:length(obj6));
%     semilogy(time_svrg,-obj_svrg,"LineWidth",1.5,"LineStyle","-.","Marker","diamond",'MarkerIndices', 1:10:length(obj_svrg));
%     %semilogy(time_sarah,obj_sarah,"LineWidth",1.5,"LineStyle","--","Marker","v",'MarkerIndices', 1:10:length(obj_sarah));
%     grid on;
%     legend("CBPG","RBPG","SAGA","SVRG");
%     title("Time")
%     timefig = sprintf('%stime_%s.fig', resultfolder,list(ss).name);
%     savefig(fig2,timefig); 
% end
% figure
% semilogy(time_det_indie,-obj_det_indie,"LineWidth",1.5,"LineStyle","--","Marker","x",'MarkerIndices', 1:10:length(obj_det_indie));
% hold on;
% semilogy(time_det_global,-obj_det_global,"LineWidth",1.5,"Marker","square",'MarkerIndices', 1:10:length(obj_det_global));
% semilogy(time_svrg,-obj_svrg,"LineWidth",1.5,"LineStyle","-.","Marker","diamond",'MarkerIndices', 1:10:length(obj_svrg));
% semilogy(time6,-obj6,"LineWidth",1.5,"LineStyle","-.","Marker","*",'MarkerIndices', 1:10:length(obj6));
% 
% grid on;
% legend("CBPG","RBPG","SVRG","SAGA");
% title("Time")
% figure
% semilogy(-obj_det_indie,"LineWidth",1.5,"LineStyle","--","Marker","x",'MarkerIndices', 1:10:length(obj_det_indie));
% hold on;
% semilogy(-obj_det_global,"LineWidth",1.5,"Marker","square",'MarkerIndices', 1:10:length(obj_det_global));
% semilogy(-obj_svrg,"LineWidth",1.5,"LineStyle","-.","Marker","diamond",'MarkerIndices', 1:10:length(obj_svrg));
% semilogy(-obj6,"LineWidth",1.5,"LineStyle","-.","Marker","*",'MarkerIndices', 1:10:length(obj6));
% 
% grid on;
% legend("CBPG","RBPG","SVRG","SAGA");
% title("Iteration")

figure
semilogy(time_det_indie,-obj_det_indie,"LineWidth",1.5,"LineStyle","--","Marker","x",'MarkerIndices', 1:10:length(obj_det_indie));
hold on;    
semilogy(time_svrg(1:3),-obj_svrg(1:3),"LineWidth",1.5,"LineStyle","-.","Marker","*",'MarkerIndices', 1:10:30);
hold on;
semilogy(time_class,-obj_class,"LineWidth",1.5,"LineStyle",":","Marker","o",'MarkerIndices', 1:10:length(obj_det_indie));
hold on; 
semilogy(time_whole,-obj_whole,"LineWidth",1.5,"LineStyle","-.","Marker","+",'MarkerIndices', 1:10:length(obj_det_indie));
%hold on; 
%semilogy(time6,-obj6,"LineWidth",1.5,"LineStyle","-.","Marker","+",'MarkerIndices', 1:10:length(obj_det_indie));
grid on; 
legend("F-CBPG","SVRG","C-CBPG","Whole","SAGA")

figure
semilogy(time_det_indie,-obj_det_indie,"LineWidth",1.5,"LineStyle","--","Marker","x",'MarkerIndices', 1:10:length(obj_det_indie));
hold on;    
semilogy(time_class,-obj_class,"LineWidth",1.5,"LineStyle",":","Marker","o",'MarkerIndices', 1:10:length(obj_det_indie));
hold on; 
semilogy(time_whole,-obj_whole,"LineWidth",1.5,"LineStyle","-.","Marker","+",'MarkerIndices', 1:10:length(obj_det_indie));
%hold on; 
%semilogy(time6,-obj6,"LineWidth",1.5,"LineStyle","-.","Marker","+",'MarkerIndices', 1:10:length(obj_det_indie));
grid on; 
legend("F-CBPG","C-CBPG","Whole","SAGA")