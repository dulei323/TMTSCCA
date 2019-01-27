clc; clear;
% -----------------------------------------
% example code for ISMB/ECCB 2019 (longitudinal multi-task SCCA)
%------------------------------------------
% Author: Lei Du, dulei@nwpu.edu.cn
% Date created:12-08-2018
% @Northwestern Ploytechnical University.
% -----------------------------------------

% load data
addpath('./SCCA_func/');
addpath('./flsa/');
addpath('./q1/');
addpath('./synthetic_data_sets/');
load example_data.mat;

% set parameters
% T-MTSCCA
opts.lambda.u1 = 1; % group L21-norm
opts.lambda.u2 = 0.001; % L1-norm, individual sparsity
opts.lambda.u3 = 1; % L21-norm, individual across tasks
opts.lambda.v1 = 0.001; % L1-norm, individual sparsity
opts.lambda.v2 = 0.1; % time-consistent norm
opts.lambda.v3 = 0.0001; % L21-norm, individual across tasks
opts.X_group = group_idx_x;

% mSCCA
mopts.lambda.u1 = 0.01;
mopts.lambda.v1 = 0.01;

% TGSCCA
tgopts.lambda.u1 = 0.1;
tgopts.lambda.v1 = 0.15;
tgopts.lambda.v2 = 0.1;

% Cross-Validation
Kfold = 10;
[n_sbj, ~] = size(X);
indices = crossvalind('Kfold',n_sbj,Kfold);
% disp('=====================================');
disp('Begin cross validition ...');
disp('===========================');
for k = 1:Kfold
    fprintf('current fold: %d\n',k);
    
    test_idx = indices==k;
    train_idx = ~test_idx;
    
    % set training and testing sets
    % training set
    itrain_set.X = getNormalization(X(train_idx,:));
    itrain_set.Y1 = getNormalization(Y1(train_idx,:));
    itrain_set.Y2 = getNormalization(Y2(train_idx,:));
    itrain_set.Y3 = getNormalization(Y3(train_idx,:));
    itrain_set.Y4 = getNormalization(Y4(train_idx,:));
    
    % testing set
    itest_set.X = getNormalization(X(test_idx,:));
    itest_set.Y1 = getNormalization(Y1(test_idx,:));
    itest_set.Y2 = getNormalization(Y2(test_idx,:));
    itest_set.Y3 = getNormalization(Y3(test_idx,:));
    itest_set.Y4 = getNormalization(Y4(test_idx,:));
    
%     % multi-view SCCA
    [u_mscca, v_mscca] = mSCCA(itrain_set, mopts);    
    % CC
    % X-Y1
    corr_train.mscca1(k) = corr(itrain_set.X*u_mscca, itrain_set.Y1*v_mscca(:,1));
    corr_test.mscca1(k) = corr(itest_set.X*u_mscca, itest_set.Y1*v_mscca(:,1));
    % X-Y2
    corr_train.mscca2(k) = corr(itrain_set.X*u_mscca, itrain_set.Y2*v_mscca(:,2));
    corr_test.mscca2(k) = corr(itest_set.X*u_mscca, itest_set.Y2*v_mscca(:,2));
    % X-Y3
    corr_train.mscca3(k) = corr(itrain_set.X*u_mscca, itrain_set.Y3*v_mscca(:,3));
    corr_test.mscca3(k) = corr(itest_set.X*u_mscca, itest_set.Y3*v_mscca(:,3));
    % X-Y4    
    corr_train.mscca4(k) = corr(itrain_set.X*u_mscca, itrain_set.Y4*v_mscca(:,4));
    corr_test.mscca4(k) = corr(itest_set.X*u_mscca, itest_set.Y4*v_mscca(:,4));
    % u, v
    u1.mscca(:,k) = u_mscca;
    v1.mscca(:,k) = v_mscca(:,1);
    v2.mscca(:,k) = v_mscca(:,2);
    v3.mscca(:,k) = v_mscca(:,3);
    v4.mscca(:,k) = v_mscca(:,4);
    
    % TGSCCA
    [u_tgscca, v_tgscca] = f_TGSCCA(itrain_set,tgopts);    
    % CC
    % X-Y1
    corr_train.tgscca1(k) = corr(itrain_set.X*u_tgscca, itrain_set.Y1*v_tgscca(:,1));
    corr_test.tgscca1(k) = corr(itest_set.X*u_tgscca, itest_set.Y1*v_tgscca(:,1));
    % X-Y2
    corr_train.tgscca2(k) = corr(itrain_set.X*u_tgscca, itrain_set.Y2*v_tgscca(:,2));
    corr_test.tgscca2(k) = corr(itest_set.X*u_tgscca, itest_set.Y2*v_tgscca(:,2));
    % X-Y3
    corr_train.tgscca3(k) = corr(itrain_set.X*u_tgscca, itrain_set.Y3*v_tgscca(:,3));
    corr_test.tgscca3(k) = corr(itest_set.X*u_tgscca, itest_set.Y3*v_tgscca(:,3));
    % X-Y4    
    corr_train.tgscca4(k) = corr(itrain_set.X*u_tgscca, itrain_set.Y4*v_tgscca(:,4));
    corr_test.tgscca4(k) = corr(itest_set.X*u_tgscca, itest_set.Y4*v_tgscca(:,4));
    % u, v
    u1.tgscca(:,k) = u_tgscca;
    v1.tgscca(:,k) = v_tgscca(:,1);
    v2.tgscca(:,k) = v_tgscca(:,2);
    v3.tgscca(:,k) = v_tgscca(:,3);
    v4.tgscca(:,k) = v_tgscca(:,4);
    
    % temporal multi-task SCCA
    [u_mtscca, v_mtscca] = TMTSCCA(itrain_set, opts);
    % CC
    % X-Y1
    corr_train.mtscca1(k) = corr(itrain_set.X*u_mtscca(:,1), itrain_set.Y1*v_mtscca(:,1));
    corr_test.mtscca1(k) = corr(itest_set.X*u_mtscca(:,1), itest_set.Y1*v_mtscca(:,1));
    % X-Y2
    corr_train.mtscca2(k) = corr(itrain_set.X*u_mtscca(:,2), itrain_set.Y2*v_mtscca(:,2));
    corr_test.mtscca2(k) = corr(itest_set.X*u_mtscca(:,2), itest_set.Y2*v_mtscca(:,2));
    % X-Y3
    corr_train.mtscca3(k) = corr(itrain_set.X*u_mtscca(:,3), itrain_set.Y3*v_mtscca(:,3));
    corr_test.mtscca3(k) = corr(itest_set.X*u_mtscca(:,3), itest_set.Y3*v_mtscca(:,3));
    % X-Y4
    corr_train.mtscca4(k) = corr(itrain_set.X*u_mtscca(:,4), itrain_set.Y4*v_mtscca(:,4));
    corr_test.mtscca4(k) = corr(itest_set.X*u_mtscca(:,4), itest_set.Y4*v_mtscca(:,4));
    % u, v
    u1.mtscca(:,k) = u_mtscca(:,1);
    v1.mtscca(:,k) = v_mtscca(:,1);
    u2.mtscca(:,k) = u_mtscca(:,2);
    v2.mtscca(:,k) = v_mtscca(:,2);
    u3.mtscca(:,k) = u_mtscca(:,3);
    v3.mtscca(:,k) = v_mtscca(:,3);
    u4.mtscca(:,k) = u_mtscca(:,4);
    v4.mtscca(:,k) = v_mtscca(:,4);
end
disp('===========================');

% #########################################################################
% find the best K results
id_mscca = 1;
id_tgscca = 2;
id_tmt = 3;
% ==============mSCCA=================
% mean top best CC
cc_train.mscca = [corr_train.mscca1',corr_train.mscca2',corr_train.mscca3',corr_train.mscca4'];
cc_test.mscca = [corr_test.mscca1',corr_test.mscca2',corr_test.mscca3',corr_test.mscca4'];
istats.meancctr(id_mscca,:) = mean(abs(cc_train.mscca));
istats.meanccte(id_mscca,:) = mean(abs(cc_test.mscca));
% mean U and V
MU = mean(u1.mscca,2);
MV = mean(v1.mscca,2);
% ==============TGSCCA=================
% mean top best CC
cc_train.tgscca = [corr_train.tgscca1',corr_train.tgscca2',corr_train.tgscca3',corr_train.tgscca4'];
cc_test.tgscca = [corr_test.tgscca1',corr_test.tgscca2',corr_test.tgscca3',corr_test.tgscca4'];
istats.meancctr(id_tgscca,:) = mean(abs(cc_train.tgscca));
istats.meanccte(id_tgscca,:) = mean(abs(cc_test.tgscca));
% mean U and V
TGU = mean(u1.tgscca,2);
TGV = mean(v1.tgscca,2);
% ==============TMTSCCA=================
% mean top best CC
cc_train.tmtscca = [corr_train.mtscca1',corr_train.mtscca2',corr_train.mtscca3',corr_train.mtscca4'];
cc_test.tmtscca = [corr_test.mtscca1',corr_test.mtscca2',corr_test.mtscca3',corr_test.mtscca4'];
istats.meancctr(id_tmt,:) = mean(abs(cc_train.tmtscca));
istats.meanccte(id_tmt,:) = mean(abs(cc_test.tmtscca));
% mean U and V
LMTU = [mean(u1.mtscca,2),mean(u2.mtscca,2),mean(u3.mtscca,2),mean(u4.mtscca,2)];
LMTV = [mean(v1.mtscca,2),mean(v2.mtscca,2),mean(v3.mtscca,2),mean(v4.mtscca,2)];

% figure
% -----u--------
figure(1)
% ground truth
colorValue = 0.01;
subplot(411)
imagesc(u');
caxis([-1*colorValue 1*colorValue]);
colorbar;
% mscca
subplot(412)
imagesc(MU');
caxis([-1*colorValue 1*colorValue]);
colorbar;
% tgscca
subplot(413)
imagesc(TGU');
caxis([-0.5*colorValue 0.5*colorValue]);
colorbar;
% lmtscca
subplot(414)
imagesc(LMTU');
caxis([-1*colorValue 1*colorValue]);
colorbar;
colormap jet;
% -----v--------
figure(2)
% ground truth
colorValue = 0.01;
subplot(411)
imagesc(V');
caxis([-1*colorValue 1*colorValue]);
colorbar;
% mscca
subplot(412)
imagesc(MV');
caxis([-1*colorValue 1*colorValue]);
colorbar;
% tgscca
subplot(413)
% colorValue = 0.05;
imagesc(TGV');
caxis([-5*colorValue 5*colorValue]);
colorbar;
% lmtscca
subplot(414)
colorValue = 0.05;
imagesc(LMTV');
caxis([-1*colorValue 1*colorValue]);
colorbar;
colormap jet;

% CCC
figure(3)
fontsize = 15;
subplot(121)
% traing
mscca=istats.meancctr(1,:);
tgscca=istats.meancctr(2,:);
tmt=istats.meancctr(3,:);
bar([1 2 3 4],[mscca' tgscca' tmt']);
axis square
colormap jet;
% legend('multi-view SCCA','TGSCCA','T-MTSCCA');
set(gca, 'FontSize',fontsize,'XTick',[1 2 3 4],'XTickLabel',{'T1','T2','T3','T4'});
ylabel('Training Correlation Coefficients','FontSize',fontsize)

subplot(122)
% traing
mscca=istats.meanccte(1,:);
tgscca=istats.meanccte(2,:);
tmt=istats.meanccte(3,:);
hb = bar([1 2 3 4],[mscca' tgscca' tmt']);
axis square
colormap jet;
legend('multi-view SCCA','TGSCCA','T-MTSCCA');
set(gca, 'FontSize',fontsize,'XTick',[1 2 3 4],'XTickLabel',{'T1','T2','T3','T4'});
ylabel('Testing Correlation Coefficients','FontSize',fontsize)