function [acc, F1]=PINLFCS(X,Y,tstX,tstY,noiserate,lambda,beta,tau)
tic;
%% import dataset  PIN-LFCS
data=[X,Y];
[num_data,dim_data]=size(data);
%% parameter&&ready
k=size(find(Y==1),1);%the number of positive example
P_prob=k/((1-noiserate)*num_data);%the probability of true P
constantofMu=1/(1-2*P_prob*noiserate);%1-2pn
param.lambda=lambda;%the parameter of regularization%constraint
%Positive Set , Unlabeled Set as noised Negative Set;
totalData_X=X;
totalData_Y=Y;
P_lable=totalData_Y(1:k);
P_data=totalData_X(1:k,:);
N=data(k+1:num_data,:);
sizeofN=size(N,1);%the number of unlabeled samples
g=6;%the partition number of group:g, the minimum number of group
%% Call Median-of-means estimator to estimate the centroid of corrupted negative set
[mean_noisedS]=Median_of_means(N,g); 
mean_noisedS=mean_noisedS';
%% negative instance centriod smooth 
N_feature=N(:,1:dim_data-1);
N_feature=mapminmax(N_feature,0,1);
N_label=N(:,dim_data);
%compute the empirical covariance matrix by eq10 in our paper
%the first term in the Eq10 of the empirical covariance matrix
segma1=zeros(dim_data-1,dim_data-1);
for i=1:sizeofN
    singleX=N_feature(i,:);
    segma1=segma1+(singleX'*singleX)/(sizeofN^2);
end
%the second term in the Eq of the empirical covariance matrix
segma2=((sum((N_feature'.*repmat(N_label',dim_data-1,1))/sizeofN,2))*(sum(N_feature.*repmat(N_label,1,dim_data-1)/sizeofN,1)))/sizeofN;
segma=segma1-segma2;
segma=segma+0.01*eye(dim_data-1);

%% CVX 
w=rand(dim_data-1,1);
mu=((segma)^-1)*w*sqrt(beta/(w'*((segma)^-1)*w))+mean_noisedS;
constant_C=constantofMu;  
C1=1;
C2=0.5;
time=1;
obj1=inf;
while(time<4)
obj=obj1;
    cvx_begin quiet
variables w(dim_data-1) xi(num_data) si(sizeofN) b;    
minimize (param.lambda*sum(w.^2)+C1*sum(xi)+C2*sum(si)-constant_C*(w'*mu));
subject to   
totalData_Y.* (totalData_X* w+b)>=1-xi; 
totalData_Y.* (totalData_X* w+b)<=1+xi.*inv_pos(tau); 
N_label.* (N_feature * w+b)<=si-1;  
N_label.* (N_feature * w+b)>=-si.*inv_pos(tau)-1;  
xi>=0;
si>=0;
cvx_end
obj1=cvx_optval;
eps1=obj-obj1;

time=time+1;
mu=((segma)^-1)*w*sqrt(beta/(w'*((segma)^-1)*w))+mean_noisedS;
end

time = toc;

%% testing
num_test=size(tstX,1);
y=sign(tstX*w+b);
predict_label=y;

tp = sum((predict_label == 1) & (tstY == 1)); % True Positive
fp = sum((predict_label == 1) & (tstY == -1)); % False Positive
fn = sum((predict_label == -1) & (tstY == 1)); % False Negative
tn = sum((predict_label == -1) & (tstY == -1)); % True Negative
acc = (tp + tn) / (tp + fp + fn + tn);
pre = tp / (tp + fp);
rec = tp / (tp + fn);
F1 = 2 * (pre * rec) / (pre + rec);
end