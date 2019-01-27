%function Temporally-constained Group Sparse Canonical Correlation Analysis (TGSCCA)
function [w, V, funVal] = f_TGSCCA(idata, paras)

%% Problem
%  Solution with scaling and constraints in lagrangian form
%  min - sum_i w'X'Y_iv_i + lambda1 * sum_(i-1)|| x_i - x_{i-1}\|_1 + lambda2 * sum_j ||x^j||_2
% S.t. ||w||_2^2=1 ||V||_F^2=1
%
%  x^j denotes the j-th row of x
%  x_i denotes the i-th column of x
%  y_i denotes the i-th column of y
%
%  z=[lambda1, lambda2, lambda3]
% -------------------------------------------------------------------------
% Input:
%       - X: n x p, geno matrix,
%       - Y, n x q*t, pheno matrix
%       - paras, parameters: lambda1(for pheno temporally-constained para),lambda2(geno para),lambda3(pheno para) 
% Output:
%       - w, p x 1, weight of geno
%       - V, q x t, weight of pheno * times
%       - obj: objective function value of each iteration
%--------------------------------------------------------------------------
% Citation:
%    
%--------------------------------------------------------------------------
% Author: Xiaoke Hao, robinhc@163.com
% Date created: 09/23/2016.
% @Nanjing University of Aeronautics and Astronautics
% -------------------------------------------------------------------------

X = [idata.X; idata.X; idata.X; idata.X];
Y = [idata.Y1; idata.Y2; idata.Y3; idata.Y4];

% set parameters
lambdak = paras.lambda.v2;
% lambdak = 0.001;
lambdav = paras.lambda.v1;
% lambdav = 10;
lambdaw = paras.lambda.u1;
opts=[];
n1=size(X,1)/4;
ind=[0  n1 n1*2 n1*3 n1*4];
q=2;
k=length(ind)-1;

% initialize canonical loadings
n_XVar = size(X,2);
[m,n_YVar] = size(Y); 
n_TVar=k;
w = ones(n_XVar, 1)./n_XVar; %Init
V = ones(n_YVar, k)./n_YVar;
x=V;
conA = repmat(eye(n_YVar,n_YVar),n_TVar,1);
n=n_YVar;
Ynew=zeros(n,4);
% stop criteria
% stop_err = 10e-5;
% max_iter = 1000;
stop_err = 10e-4;
max_iter = 100;

for i=1:k    
    ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
    Ynew(:,i)=Y(ind_i,:)'*X(ind_i,:)*w;
end
A=conA;
y=reshape(Ynew,n*4,1);

ATy=zeros(n, k);
ind_new=[0  n n*2 n*3 n*4];


for i=1:k
    ind_new_i=(ind_new(i)+1):ind_new(i+1);     % indices for the i-th group
    tt =A(ind_new_i,:)'*y(ind_new_i,1);
    ATy(:,i)= tt;
end
Ax=zeros(n*4,1);
for i=1:k    
    ind_new_i=(ind_new(i)+1):ind_new(i+1);     % indices for the i-th group
    Ax(ind_new_i,1)=A(ind_new_i,:)* x(:,i);
end

R=zeros(k,k-1);
R(1:(k+1):end)=-1;
R(2:(k+1):end)=1;
Z0=zeros(n,k-1);

L=1;
xp=x; Axp=Ax; xxp=zeros(n,k);    
alphap=0; alpha=1;   

for iter = 1:max_iter
    
    % fix w, get V
    beta=(alphap-1)/alpha;    s=x + beta* xxp;
    As=Ax + beta* (Ax-Axp);       
    for i=1:k    
        ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
        Ynew(:,i)=Y(ind_i,:)'*X(ind_i,:)*w;
    end
    y=reshape(Ynew,n*4,1);
    for i=1:k
        ind_new_i=(ind_new(i)+1):ind_new(i+1);     % indices for the i-th group
        tt =A(ind_new_i,:)'*y(ind_new_i,1);
        ATy(:,i)= tt;
    end   
    for i=1:k
        ind_new_i=(ind_new(i)+1):ind_new(i+1);     % indices for the i-th group           
        tt =A(ind_new_i,:)'*As(ind_new_i,1);           
        ATAs(:,i)= tt;
    end
    g=ATAs-ATy;
    xp=x;    Axp=Ax;

    while (1)
        v=s-g/L;  
        [x, Z, gap]=tesla_proj(v, Z0,...
            0, lambdak/L, n, k,...
            1000, 1e-8, 1, 6);
        Z0=Z;            
        v=x;
        x=eppMatrix(v, n, k, lambdav/ L, q);            
        v=x-s;  
        for i=1:k
            ind_new_i=(ind_new(i)+1):ind_new(i+1);     % indices for the i-th group           
            Ax(ind_new_i,1)=A(ind_new_i,:)* x(:,i);
        end            
        Av=Ax -As;
        r_sum=norm(v,'fro')^2; 
        l_sum=norm(Av, 'fro')^2;
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end            
        if(l_sum <= r_sum * L)
            break;
        else
            L=max(2*L, l_sum/r_sum);
        end
    end       
    alphap=alpha; alpha= (1+ sqrt(4*alpha*alpha +1))/2;       
    ValueL(iter)=L;        
    xxp=x-xp;   Axy=Ax-y; 
    V=x;   
    for i=1:k
        scale1(i,:) = sqrt(V(:,i)'*V(:,i));
        V(:,i) = V(:,i)./ scale1(i,:);
    end
    
        
    % fix V, get w
    for i=1:k    
        ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
        res(:,i)=X(ind_i,:)'*Y(ind_i,:)*V(:,i);
    end
    XY = sum(res,2);
    XX = X(ind_i,:)'*X(ind_i,:); 
    Wi = sqrt(sum(w.*w,2)+eps);
    D1 = diag(1./Wi);
    w = (XX+lambdaw*D1)\XY;
    scale2 = sqrt(w'*XX*w);
    w = w / scale2;
    for i=1:k    
        ind_i=(ind(i)+1):ind(i+1);     % indices for the i-th group
        funValCorr(i,1)=w'*X(ind_i,:)'*Y(ind_i,:)*V(:,i);
    end 
    funVal(iter) = -sum(funValCorr,1) + lambdaw*sum(abs(w))+lambdak* sum(sum(abs(V * R)));
    for i=1:n
        funVal(iter)=funVal(iter)+ lambdav* norm(V(i,:), q);
    end    
    
    if iter > 2 && abs(funVal(iter) - funVal(iter-1)) < stop_err
        break;
    end
    
end
