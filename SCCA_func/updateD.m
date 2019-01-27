function [d, struct_out] = updateD(beta, struct_in, CM, lnorm)
% --------------------------------------------------------------------
% Update the diagnoal matrix
% --------------------------------------------------------------------
% Input:
%       - beta, coeffients
%       - struct_in, matrxi regarding group structure
%       - CM, correaltion matrix
% Output:
%       - d, diagonal of matrix D
%       - struct_out, found group structure
%------------------------------------------
% Author: Lei Du, leidu@iu.edu
% Date created: Jan-02-2015
% Date updated: Oct-09-2015
%% Copyright (C) 2013-2015 Li Shen (shenli@iu.edu) and Lei Du
% -----------------------------------------

% group = 0;
% if nargin == 1
%     d = 1 ./ sqrt(beta.^2+eps);
%     group = sum(abs(beta));
% else
%     [nrow,ncol] = size(group_in);
%     for g_i = 1:nrow
%         idx = group_in(g_i,:)~=0;
%         wc1 = beta(idx, :);
%         group = sqrt(sum(wc1.*wc1))+group; % for calculate objective function
%         d_gi = sqrt(sum(wc1.*wc1)+eps);
%         beta_i(idx) = d_gi;
%     end
%     d = 1 ./ beta_i;
% end
% group_out = group;

group = 0;
if nargin == 1
    if length(size(beta))==1
        d = 0.5 ./ sqrt(beta.^2+eps);
    else
        [p,ntask] = size(beta);
        for i = 1:p
            d(i) = 0.5 ./ (sum(beta(i,:).^2)+eps);
        end
    end
    group = sum(abs(beta));
elseif (nargin == 2 && strcmpi(lnorm,'etp')) % exponential-type penalty
    gamma = 1e3;
    numerator = gamma*exp(-gamma*abs(beta));
    denominator = 1-exp(-gamma);
    d = numerator/denominator;
elseif strcmpi(lnorm,'group')
    [nrow,~] = size(struct_in);
    for g_i = 1:nrow
        idx = struct_in(g_i,:)~=0;
        wc1 = beta(idx);
        group = sqrt(sum(wc1.*wc1))+group; % for calculate objective function
        d_gi = sqrt(sum(wc1.*wc1)+eps);
        beta_i(idx) = d_gi;
    end
    d = 1 ./ beta_i;
elseif strcmpi(lnorm,'graph')
    [nrow,ncol] = size(struct_in);
    coef = zeros(nrow,ncol);
    for g_i = 1:nrow
        idx0 = struct_in(g_i,:)==0;
        wc1 = beta;
        wc1(idx0)=0;
        group = sqrt(sum(wc1.*wc1))+group; % for calculate objective function
        d_gi = sqrt(sum(wc1.*wc1)+eps);
        coef(g_i,idx0) = d_gi;
        %         beta_i(idx) = d_gi;
    end
    beta_i = sum(coef,1);
    d = 1 ./ beta_i;
elseif strcmpi(lnorm,'pairL2')
    p = length(beta);
    w = beta.^2;
    Gp = struct_in*w+eps;
    Gp = sqrt(Gp);
    Gp = 1 ./ Gp;
    d = sum(reshape(Gp,p-1,[]));
elseif strcmpi(lnorm,'wpairL2')
    p = length(beta);
    w = beta.^2;
    Gp = struct_in*w+eps;
    Gp = sqrt(Gp);
    Gp = 1 ./ Gp;
%     CM = reshape(CM,length(CM)^2,1); % correlation matrix
    CM(1:(p+1):p^2) = [];
    Gp = CM'.*Gp;
    d = sum(reshape(Gp,p-1,[]));
end
struct_out = group;