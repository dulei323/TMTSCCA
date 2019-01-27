function [D, pen_Value] = updateDs(v1, v2, v3, v4)
% --------------------------------------------------------------------
% Update the diagnoal matrix for three modalities, i.e. v1, v2, v3
% --------------------------------------------------------------------
% Input:
%       - beta, coeffients
%       - struct_in, matrxi regarding group structure
%       - CM, correaltion matrix
% Output:
%       - d, diagonal of matrix D
%       - struct_out, found group structure
%------------------------------------------
% Author: Lei Du, dulei@nwpu.edu.cn
% Date created: Jan-02-2015
% Date updated: 12-20-2017
%% Copyright (C) 2013-2017 Li Shen (shenli@iu.edu) and Lei Du
% -----------------------------------------

vlen = length(v1);

if nargin < 4
    for i = 1:vlen
        d(i) = sqrt(v1(i).^2+v2(i).^2+eps);
    end
else
    for i = 1:vlen
        d(i) = sqrt(v1(i).^2+v2(i).^2+v3(i).^2+v4(i).^2+eps);
    end
end
D = 0.5 ./ d;
pen_Value = sum(d);