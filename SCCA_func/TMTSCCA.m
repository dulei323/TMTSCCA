function [U, V, obj] = TMTSCCA(data, opts)
% --------------------------------------------------------------------
% Temporal Multi-Task group lasso SCCA Algorithm (T-MTSCCA)
% --------------------------------------------------------------------
% Input:
%       - data, data matrix
%       - opts, parameters: (unknown, tuned from other functions)
% Output:
%       - U, weight of X
%       - V, weight of Yt.
%------------------------------------------
% Author: Lei Du
%% Copyright (C) 2016-2018 Li Shen (Li.Shen@pennmedicine.upenn.edu) and Lei Du (dulei@nwpu.edu.cn)
% -----------------------------------------
X = data.X; % SNP data
Y1 = data.Y1; % T1
Y2 = data.Y2; % T2
Y3 = data.Y3; % T3
Y4 = data.Y4; % T4

group_info = opts.X_group;

p = size(X,2);
q1 = size(Y1,2);
q2 = size(Y2,2);
q3 = size(Y3,2);
q4 = size(Y4,2);

u1 = ones(p, 1); % initialize v1 here
u2 = ones(p, 1); % initialize v2 here
u3 = ones(p, 1); % initialize v3 here
u4 = ones(p, 1); % initialize v4 here
v1 = ones(q1, 1); % initialize v1 here
v2 = ones(q2, 1); % initialize v2 here
v3 = ones(q3, 1); % initialize v3 here
v4 = ones(q4, 1); % initialize v4 here
U = [u1 u2 u3 u4]; % initialize U here
V = [v1 v2 v3 v4]; % initialize V here

% set parameters
lambda = opts.lambda;

% set stopping criteria
max_Iter = 100;
t = 0;
tol = 1e-5;
obj = [];
tu = inf;
tv1 = inf;
tv2 = inf;
tv3 = inf;
tv4 = inf;

XX = X'*X;
Yv1 = Y1*v1;
Yv2 = Y2*v2;
Yv3 = Y3*v3;
Yv4 = Y4*v4;
YY1 = Y1'*Y1;
YY2 = Y2'*Y2;
YY3 = Y3'*Y3;
YY4 = Y4'*Y4;
while (t<max_Iter && (tu>tol || tv1>tol || tv2>tol || tv3>tol || tv4>tol)) % default 100 times of iteration
    t = t+1;
    
    % update u
    % -------------------------------------
    U_old = U;
    Yv = [Yv1 Yv2 Yv3 Yv4];    
    
    % update matrics: \tilde{\mathbf{D}}_1, \bar{\mathbf{D}}_1
    ggd = updateD_group_Given2(U,group_info);
    GGD = diag(ggd);
    D1 = updateDs(u1,u2,u3,u4);
    D1 = diag(D1);
    
    % u1
    gd1 = updateD(u1);
    GD1 = diag(gd1);
    F1 = XX+lambda.u3*D1+lambda.u2*GD1+lambda.u1*GGD;
    b1 = X'*Yv1;
    u1 = F1\b1;
    % u2
    gd1 = updateD(u2);
    GD1 = diag(gd1);
    F1 = XX+lambda.u3*D1+lambda.u2*GD1+lambda.u1*GGD;
    b1 = X'*Yv2;
    u2 = F1\b1;
    % u3
    gd1 = updateD(u3);
    GD1 = diag(gd1);
    F1 = XX+lambda.u3*D1+lambda.u2*GD1+lambda.u1*GGD;
    b1 = X'*Yv3;
    u3 = F1\b1;
    % u4
    gd1 = updateD(u4);
    GD1 = diag(gd1);
    F1 = XX+lambda.u3*D1+lambda.u2*GD1+lambda.u1*GGD;
    b1 = X'*Yv4;
    u4 = F1\b1;

    % scale u
    su1 = u1'*XX*u1;
    su2 = u2'*XX*u2;
    su3 = u3'*XX*u3;
    su4 = u4'*XX*u4;
    u1 = u1 / sqrt(su1);
    u2 = u2 / sqrt(su2);
    u3 = u3 / sqrt(su3);
    u4 = u4 / sqrt(su4);
    U = [u1 u2 u3 u4];
    Xu1 = X*u1;
    Xu2 = X*u2;
    Xu3 = X*u3;
    Xu4 = X*u4;
    
    % update v
    % -------------------------------------
    v_old1 = v1;
    v_old2 = v2;
    v_old3 = v3;
    v_old4 = v4;
    % update v
    % -------------------------------------
    di1 = updateD(v1);
    Di1 = lambda.v1*diag(di1);
    di2 = updateD(v2);
    Di2 = lambda.v1*diag(di2);
    di3 = updateD(v3);
    Di3 = lambda.v1*diag(di3);
    di4 = updateD(v4);
    Di4 = lambda.v1*diag(di4);
    dv12 = updateDV_FP(v1,v2);
    dv123 = updateDV_FP(v1,v2,v3);
    dv234 = updateDV_FP(v2,v3,v4);
    dv34 = updateDV_FP(v3,v4);
    Dv12 = lambda.v2*diag(dv12);
    Dv123 = lambda.v2*diag(dv123);
    Dv234 = lambda.v2*diag(dv234);
    Dv34 = lambda.v2*diag(dv34);
    ds = updateDs(v1,v2,v3,v4);
    DS = lambda.v3*diag(ds);
    % ------------------------
    % v1
    F2 = YY1+Di1+Dv12+DS;
    b2 = Y1'*Xu1;
    v1 = F2\b2;
    sv1 = sqrt(v1'*YY1*v1);
    v1 = v1 ./ sv1;
    % v2
    F2 = YY2+Di2+Dv123+DS;
    b2 = Y2'*Xu2;
    v2 = F2\b2;
    sv2 = sqrt(v2'*YY2*v2);
    v2 = v2 ./ sv2;
    % v3
    F2 = YY3+Di3+Dv234+DS;
    b2 = Y3'*Xu3;
    v3 = F2\b2;
    sv3 = sqrt(v3'*YY3*v3);
    v3 = v3 ./ sv3;
    % v4
    F2 = YY4+Di4+Dv34+DS;
    b2 = Y4'*Xu4;
    v4 = F2\b2;
    sv4 = sqrt(v4'*YY4*v4);
    v4 = v4 ./ sv4;
    
    % prepare Y
    Yv1 = Y1*v1;
    Yv2 = Y2*v2;
    Yv3 = Y3*v3;
    Yv4 = Y4*v4;
    
    % ------------------------------
    % stopping condition
    if t > 1
        tu = max(max(abs(U-U_old)));
        tv1 = max(abs(v1-v_old1));
        tv2 = max(abs(v2-v_old2));
        tv3 = max(abs(v3-v_old3));
        tv4 = max(abs(v4-v_old4));
    else
        tu = tol*10;
        tv1 = tol*10;
        tv2 = tol*10;
        tv3 = tol*10;
        tv4 = tol*10;
    end
end
V = [v1 v2 v3 v4];
end
