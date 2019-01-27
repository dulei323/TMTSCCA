function [D, pen_Value] = updateDV_FP(v1, v2, v3)
% --------------------------------------------------------------------
% Update the diagnoal matrix for longitudinal FGL
% --------------------------------------------------------------------
%------------------------------------------
% Author: Lei Du, dulei@nwpu.edu.cn
% Date created: Jan-02-2015
% Date updated: 01-08-2019
%% Copyright (C) 2013-2019 Li Shen (shenli@iu.edu) and Lei Du
% -----------------------------------------

vlen = length(v1);

if nargin < 3
    for i = 1:vlen
        d(i) = sqrt(v1(i).^2+v2(i).^2+eps);
    end
    D = 0.5 ./ d;
else
    for i = 1:vlen
        d(i) = 0.5/sqrt(v1(i).^2+v2(i).^2+eps)+0.5/sqrt(v2(i).^2+v3(i).^2+eps);
%         d(i) = sqrt(v1(i).^2+v2(i).^2+v3(i).^2+eps);
    end
    D = d;
end
pen_Value = sum(d);