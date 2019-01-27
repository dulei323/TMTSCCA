function [u, V] = mSCCA(data, opts)
% -----------------------------------------
% Multi-view SCCA Algorithm
%------------------------------------------
% Author: Lei Du, dulei@nwpu.edu.cn
%% Copyright (C) 2016-2019 Li Shen (Li.Shen@pennmedicine.upenn.edu) and Lei Du (dulei@nwpu.edu.cn)
% -----------------------------------------
X = data.X;
Y1 = data.Y1; % T1
Y2 = data.Y2; % T2
Y3 = data.Y3; % T3
Y4 = data.Y4; % T4

p = size(X,2);
q1 = size(Y1,2);
q2 = size(Y2,2);
q3 = size(Y3,2);
q4 = size(Y4,2);

u = ones(p, 1); % initialize u here
v1 = ones(q1, 1); % initialize v1 here
v2 = ones(q2, 1); % initialize v2 here
v3 = ones(q3, 1); % initialize v3 here
v4 = ones(q4, 1); % initialize v4 here

% set parameters
lambda = opts.lambda;

% set stopping criteria
max_Iter = 100;
t = 0;
tol = 1e-5;
tu = inf;
tv1 = inf;
tv2 = inf;
tv3 = inf;
tv4 = inf;

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
    u_old = u;
    Yv = Yv1+Yv2+Yv3+Yv4;    
    XX = X'*X;
    % update DI1
    D1 = updateD(u);
    D1 = diag(D1);
    F1 = XX+lambda.u1*D1;
    b1 = X'*Yv;
    u = F1\b1;
    s1 = u'*XX*u;
    
    % scale u
    u = u./s1;
    Xu = X*u;
    
    % update v
    % -------------------------------------
    v_old1 = v1;
    % update v
    % -------------------------------------
    Dv1 = updateD(v1);
    Dv1 = lambda.v1*diag(Dv1);
    % solve v
    % v1
    F2 = YY1+Dv1;
    b2 = Y1'*(Xu+Yv2+Yv3+Yv4);
    v1 = F2\b2;
    scale = sqrt(v1'*YY1*v1);
    v1 = v1 ./ scale;
    Yv1 = Y1*v1;
    % -----------------------------
    % solve v2
    v_old2 = v2;
    Dv2 = updateD(v2);
    Dv2 = lambda.v1*diag(Dv2);
    F2 = YY2+Dv2;
    %     F2 = D2;
    b2 = Y2'*(Xu+Yv1+Yv3+Yv4);
    v2 = F2\b2;
    scale = sqrt(v2'*YY2*v2);
    v2 = v2 ./ scale;
    Yv2 = Y2*v2;
    % -----------------------------
    % solve v3
    v_old3 = v3;
    Dv3 = updateD(v3);
    Dv3 = lambda.v1*diag(Dv3);
    F2 = YY3+Dv3;
    b2 = Y3'*(Xu+Yv1+Yv2+Yv4);
    v3 = F2\b2;
    scale = sqrt(v3'*YY3*v3);
    v3 = v3 ./ scale;
    Yv3 = Y3*v3;
    % -----------------------------
    % solve v4
    v_old4 = v4;
    Dv4 = updateD(v4);
    Dv4 = lambda.v1*diag(Dv4);
    F2 = YY4+Dv4;
    b2 = Y4'*(Xu+Yv1+Yv2+Yv3);
    v4 = F2\b2;
    scale = sqrt(v4'*YY4*v4);
    v4 = v4 ./ scale;
    Yv4 = Y4*v4;
    
    % ------------------------------
    % stopping condition
    if t > 1
        tu = max(abs(u-u_old));
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