function [Wt,bt,Pt,VectorObj] = Update_parameters_ft(Para_ft,Xl,Yl,lambda,d)
%UPDATE_PARAMETERS Summary of this function goes here
%   Detailed explanation goes here

%e = 1e-5;
%diff = checkgrad(@CalculateGradient_ft, Para_ft, e, Xl, Yl, lambda, d);
%fprintf('the difference is: %f\n', diff);
[~,dt] = size(Xl);
[~,c] = size(Yl);  % the number of labels
[FinalPara, VectorObj, ~] = ...
    minimize(Para_ft, @CalculateGradient_ft, 20, Xl, Yl, lambda, d);
%-----------------------------------------------------%
i_Wt = 0;   % the index of Wt
i_bt = i_Wt+c*d;   % the index of bt
i_Pt = i_bt+c*1;  % the index of Pt
% Get all parameters 
Wt = reshape(FinalPara(i_Wt+1:i_Wt+c*d, 1), c, d);
bt = FinalPara(i_bt+1:i_bt+c*1, 1);
Pt = reshape(FinalPara(i_Pt+1:i_Pt+dt*d, 1), dt, d);
%----------------------------------------------------%

end

