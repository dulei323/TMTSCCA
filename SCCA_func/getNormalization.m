
% Normalizating data set
function Y = getNormalization(X)

[~,p] = size(X);
Y = X;

for i = 1 : p
    Xv = X(:,i);
    Xv = Xv - mean(Xv);
    Xvn = Xv/norm(Xv);
    Y(:,i) = Xvn;
end

% end function