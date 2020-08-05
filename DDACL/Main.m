clear all; clc;
addpath('../Datasets/ObjectRecognition');
SCS = 'source_caltech_surf.mat';
TAD = 'target_amazon_decaf.mat';
tag = 1;
%==========================================================================
source_exp = {SCS};
target_exp = {TAD};
%==========================================================================
% parameters:
beta = 0.001;
tau = 0.002;
lambda = 0.001;
d = 100; % the dimension of the subspace
T = 5; % the iteration
fprintf('beta = %.4f, tau = %.4f, lambda = %.4f, d = %.2f\n', beta, tau, lambda, d);
%==========================================================================
len = length(source_exp);
% iter = 20;  % for final
iter = 1;     % for programming
%------------------------------------------------------------------------%
acc_CTF = zeros(iter,len); %Acc for CTF
for expi = 1:len
    acc1 = 0; %Acc for CTF
    disp(['Source Domain:' source_exp{expi}]);
    disp(['Target Domain:' target_exp{expi}]);
    for j = 1:iter
        fprintf('===================itertion[%d]===================\n',j);
        %---------------------------------------------------%
        % loda data
        load(source_exp{expi});
        load(target_exp{expi});
        
        Xl = training_features{j};
        Xl_Label = training_labels{j};
        Xl = normr(Xl);     % get the normalized labeled target data
        if tag == 1
            Xs = source_features;
            Xs_Label = source_labels;
        else
            Xs = source_features{j};
            Xs_Label = source_labels{j};
        end
        Xs = normr(Xs);
        
        Xu = testing_features{j};
        Xu_Label = testing_labels{j};
        Xu = normr(Xu);
        %---------------------------------------------------%
        % learning
        [Wt,bt,Pt,Ps,VectorObj] = DDA(Xs,Xs_Label,Xl,Xl_Label,Xu,Xu_Label,beta,tau,lambda,d,T);        
        %---------------------------------------------------%
        % prediction
        [~,ft_preLabel_u] = Softmax(Xu,Wt,bt,Pt);   % ft(xu);
        [~,ft_preLabel_l] = Softmax(Xl,Wt,bt,Pt);   % ft(xl);
        [~,ft_preLabel_s] = Softmax(Xs,Wt,bt,Ps);   % ft(xs);
        %---------------------------------------------------%
        ft_acc_u = Evaluate(ft_preLabel_u,Xu_Label)*100;
        ft_acc_l = Evaluate(ft_preLabel_l,Xl_Label)*100;
        ft_acc_s = Evaluate(ft_preLabel_s,Xs_Label)*100;
        %---------------------------------------------------%
        fprintf('ft(xu) accuracy is:%f\n',ft_acc_u);
        fprintf('ft(xl) accuracy is:%f\n',ft_acc_l);
        fprintf('ft(xs) accuracy is:%f\n',ft_acc_s);
        acc_CTF(j,expi) = ft_acc_u;        
    end
    fprintf('===========================================================\n');
    fprintf('CTF Total Acc:%f, Average_CTF = %f +/- %f\n',mean(acc_CTF(:,expi)), std(acc_CTF(:,expi))/sqrt(iter));
end