function [mean_noisedS ]=Median_of_means(sampledata,g)
[num, dim]=size(sampledata);
%% randomly partition N into g groups && median-of-means estimator of label mean of U
%the partition number of group:g, the minimum number of group
while(mod(num,g)~=0) 
    g=g+1;
end
%the number of samples in every group: g_number
g_number=num/g;
% partition and compute mean_noisedS(U(Sn)=¦²(yixi/n))
index=randperm(num); %Randomly sample(num numbers)
group_start=1;
group_end=g_number;
mean_noisedS=[];

for i=1:g
    S=sampledata(index(group_start:group_end),:);
    %each group
    group_start=group_end+1;
    group_end=group_start+g_number-1;
    %update
    X=S(:,1:dim-1);
    Y=S(:,dim);
    Y=repmat(Y,1,dim-1);%matrix y duplicated by 1*dim-1
    C=Y.*X;%element product
    mean_noisedS=[mean_noisedS;sum(C,1)/g_number];
end

median_of_mean=[];
%ri=medianj{||¦Ì(Sn[i])-¦Ì(Sn[j])||}for each i ¡Êg,and then set i*=argmin ri ,i¡Êg
%return ¦Ì(Sn)=¦Ì(Sn[i*])
for i=1:g
    S=repmat(mean_noisedS(i,:),g-1,1);
    rest_S=mean_noisedS(setdiff((1:g),i),:);%Remove the remaining rows except row i
    norm_s=sum(abs(S-rest_S).^2,2).^(1/2); %Least squares
    median_of_mean=[median_of_mean;median(norm_s)];
end 
i_min=find(median_of_mean==min(median_of_mean));
mean_noisedS=mean(mean_noisedS(i_min,:),1);%Find the corresponding mean
end