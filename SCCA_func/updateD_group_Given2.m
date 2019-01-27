function [d, sub_group_obj_square] = updateD_group_Given2(beta,group_idx)
% --------------------------------------------------------------------
% Update the diagnoal matrix
% --------------------------------------------------------------------
% Input:
%       - beta, coeffients
%       - group_info, matrxi regarding group structure
% Output:
%       - d, diagonal of matrix D
%       - group_obj, group objective value
%------------------------------------------
% Author: Lei Du, dulei@nwpu.edu.cn
% Date created: Jan-02-2015
% Date updated: 03-22-2018
%% Copyright (C) 2013-2018 Li Shen and Lei Du
% -----------------------------------------

snp_group_set = unique(group_idx,'stable');
number_of_group = length(snp_group_set);

obj = 0;
for igroup = 1:number_of_group
    snp_idx = find(group_idx == snp_group_set(igroup));
    wc1 = beta(snp_idx, :);
    di = sqrt(sum(sum(wc1.*wc1))+eps); % for calculate objective function
    snp_wi(snp_idx) = di;
    obj(igroup) = di;
end
d = 0.5 ./ snp_wi;
sub_group_obj_square = sum(obj);