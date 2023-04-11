function [acc, F1]=PINKLFCS(X,Y,tstX,tstY,noiserate,lambda,beta,theta,tau) 
tic;
[num_data,dim_data]=size(X);
k=size(find(Y==1),1);
P_prob=k/((1-noiserate)*num_data);
constantofMu=1/(1-2*P_prob*noiserate);%1-2pn
param.lambda=lambda;
param.t0=150;
tolerance=0.01;
param.t_max=100;
                            
totalData_X=X;
totalData_Y=Y;   %The last column is the label y

N_feature=X(k+1:num_data,:);    %k+1~n is the unlabeled set U~noisy negative sample N
N_label=Y(k+1:num_data,:); 
N=[N_feature N_label];
sizeofN=num_data-k;%Returns the number of rows of the matrix N (2 represents the number of columns), that is, the number of unlabeled samples (n-k)

g=5;%the partition number of group:g, the minimum number of group
%% Call Median-of-means estimator to estimate the centroid of corrupted negative set
[mean_noisedS ]=Median_of_means(N,g); 

%% negative instance centriod smooth 
%compute the empirical covariance matrix by eq14 in our paper
%the first term in the Eq of the empirical covariance matrix
segma1=zeros(dim_data,dim_data);%Calculate the covariance matrix
for i=1:sizeofN
    singleX=N_feature(i,:);   %Features of the i-th sample
    segma1=segma1+(singleX'*singleX)/(sizeofN^2);%Conjugate transpose and multiply (xt*x)
end
%the second term in the Eq of the empirical covariance matrix
segma2=((sum((N_feature'.*repmat(N_label',dim_data,1))/sizeofN,2))*(sum(N_feature.*repmat(N_label,1,dim_data)/sizeofN,1)))/sizeofN;%元素乘积,1/2表示沿哪个轴求和
segma=segma1-segma2;
segma=segma+0.01*eye(dim_data);

%% LDCE with Kernel
% initialize alpha, gamma
alphas=rand(num_data,1);
alphas0=alphas;
gammas=rand(sizeofN,1);
gammas0=gammas;
b=0;
constant_C=-(num_data-k)*(1-tau)*constantofMu/(2*num_data);

tol1=0.1;
tol2=0.1;
C1=0.5;
C2=0.5;
%initial mu 
G_x=exp(-sum(totalData_X.*totalData_X,2)/(2*theta^2));%exp(-||x||^2/(2*theta^2)) /////sum(a,dim);
G_u=exp(-sum(mean_noisedS.*mean_noisedS,2)/(2*theta^2));%exp(-||^u||^2/(2*theta^2))
part1=repmat((alphas.*totalData_Y).*G_x,1,dim_data).*totalData_X;
part2=repmat((gammas.*totalData_Y(k+1:num_data)).*G_x(k+1:num_data,:),1,dim_data).*totalData_X(k+1:num_data,:);
delta=-sum(part1,1)/(2*param.lambda*theta^2)+sum(part2,1)/(2*param.lambda*theta^2);%+constant_C*G_u*mean_noisedS/(2*param.lambda*theta^2);
mu=-sqrt(beta/(delta*(segma^-1)*delta'))*delta*(segma^-1)+mean_noisedS;

part1=repmat(((alphas).*totalData_Y).*G_x,1,dim_data).*totalData_X;
part2=repmat(((gammas).*totalData_Y(k+1:num_data)).*G_x(k+1:num_data,:),1,dim_data).*totalData_X(k+1:num_data,:);


G=calckernel('rbf',theta,totalData_X,totalData_X);
G_mu=calckernel('rbf',theta,mu,totalData_X);
%term1 :ai aj yi yj G(xi,xj)
term1=((alphas.*totalData_Y)*(alphas.*totalData_Y)').*G;
%term2 :ai rj yi yj G(xi,xj)
term2=repmat((alphas.*totalData_Y),1,sizeofN).*repmat((gammas.*totalData_Y(k+1:num_data,:))',num_data,1).*G(:,k+1:num_data);
%term3 :ri rj yi yj G(xi,xj)
term3=((gammas.*totalData_Y(k+1:num_data,:))*(gammas.*totalData_Y(k+1:num_data,:))').*G(k+1:num_data,k+1:num_data);

object=sum(alphas,1)+sum(gammas,1)-sum(sum(term1))/(4*param.lambda)+sum(sum(term2))/(2*param.lambda)-sum(sum(term3))/(4*param.lambda)+sum(alphas.*totalData_Y.*G_mu)*constant_C/(2*param.lambda)-sum(gammas.*totalData_Y(k+1:num_data).*G_mu(k+1:num_data))*constant_C/(2*param.lambda)-constant_C^2/(4*param.lambda);
obj=[object];
%updating alpha,gamma,mu by ACS method
iter=0;
time=1;
while(time<6)
while(iter<param.t_max)
    changed=0;
    for i=1:num_data
        ay=alphas.*totalData_Y;
        ry=gammas.*totalData_Y(k+1:num_data,:);
        G_x1=G(:,i);
        G_mu1=G_mu(i);%calckernel('rbf',theta,totalData_X(i,:),mu);
        f1=1/(2*param.lambda)*sum(ay.*G_x1,1)-1/(2*param.lambda)*sum(ry.*G_x1(k+1:num_data,:),1)-1/(2*param.lambda)*constant_C*G_mu1+b;
        E1=f1-totalData_Y(i);
        bOld=b;
        b1new=0;
        b2new=0;
        b_k1new=0;
        b_k2new=0;
        %update alpha
        %choose i: alphas violated KKT conditions
        if ((totalData_Y(i).*E1<-tol1)&&(alphas(i)<C1))||((totalData_Y(i).*E1>tol1)&&(alphas(i)>-tau*C1))
        %if ((totalData_Y(i).*E1<-tol1)&&(alphas(i)<C1))||((totalData_Y(i).*E1>tol1)&&(alphas(i)>0))
            %choose j: different from i   
            ff=sum(repmat(alphas.*totalData_Y,1,num_data).*G,1)/(2*param.lambda)-sum(repmat(gammas.*totalData_Y(k+1:num_data),1,num_data).*G(k+1:num_data,:),1)/(2*param.lambda)-constant_C*G_mu'/(2*param.lambda)+repmat(b,1,num_data);
            E=ff'-totalData_Y;
            [maxxx j]=max(abs(repmat(E1,num_data,1)-E));
            alpha1=i;
            alpha2=j;
            
            %update alpha1 and alpha2
            alpha1old=alphas(alpha1,1);
            alpha2old=alphas(alpha2,1);
            y1=totalData_Y(alpha1);
            y2=totalData_Y(alpha2);
            E2=E(alpha2);
            if y1~=y2
%                 L=max(0,alpha2old-alpha1old);
%                 H=min(C1,C1+alpha2old-alpha1old);
                L=max(-tau*C1,-tau*C1+alpha2old-alpha1old);
                H=min(C1,C1+alpha2old-alpha1old);
                
            else 
%                 L=max(0,alpha2old+alpha1old-C1);
%                 H=min(C1,alpha2old+alpha1old);
                L=max(-tau*C1,alpha2old+alpha1old-C1);
                H=min(C1,tau*C1+alpha2old+alpha1old);
            end
            
            if L==H
%                 fprintf('H==L\n');
                continue;
            end
            
            G11=G(alpha1,alpha1);
            G12=G(alpha1,alpha2);
            G22=G(alpha2,alpha2);
            kernelParameter=G11-2*G12+G22;
            
            if kernelParameter<=0
%                 fprintf('alphas: G11-2*G12+G22<=0\n');
                continue;
            end
            
            alpha2new=alpha2old+2*param.lambda*(E1-E2)/kernelParameter;
            alpha2new=min(max(alpha2new,L),H);
            
            if abs(alpha2new-alpha2old)<=0.0001
%                 fprintf('change small\n');
                continue;
            end
      
            alpha1new=alpha1old+y1*y2*(alpha2old-alpha2new);
            
            %updating bias b
            
            b2new=-E2-y1*G12*(alpha1new-alpha1old)/(2*param.lambda)-y2*G22*(alpha2new-alpha2old)/(2*param.lambda)+bOld;
            b1new=-E1-y1*G11*(alpha1new-alpha1old)/(2*param.lambda)-y2*G12*(alpha2new-alpha2old)/(2*param.lambda)+bOld;
            
            alphas(alpha1,1)=alpha1new;
            alphas(alpha2,1)=alpha2new;
            changed=1;
        end
        
        %update gamma
        %choose i: gammas violated KKT conditions
      if i>=k+1
        if ((totalData_Y(i).*E1<-tol2)&&(gammas(i-k)<C2))||((totalData_Y(i).*E1>tol2)&&(gammas(i-k)>-tau*C2))
            %choose j: different from i
            ff=sum(repmat(alphas.*totalData_Y,1,num_data).*G,1)/(2*param.lambda)-sum(repmat(gammas.*totalData_Y(k+1:num_data),1,num_data).*G(k+1:num_data,:),1)/(2*param.lambda)-constant_C*G_mu'/(2*param.lambda)+repmat(b,1,num_data);
            E=ff'-totalData_Y;
            [maxxx j]=max(abs(repmat(E1,sizeofN,1)-E(k+1:num_data,:)));            
            gamma1=i;
            gamma2=j;
            
            %update alpha1 and alpha2
            gamma1old=gammas(gamma1-k,1);
            gamma2old=gammas(gamma2,1);
            y1=totalData_Y(gamma1);
            y2=totalData_Y(gamma2+k);
            E2=E(gamma2+k);
            
            if y1~=y2
                L=max(-tau*C2,-tau*C2+gamma2old-gamma1old);
                H=min(C2,C2+gamma2old-gamma1old);
            else 
                L=max(-tau*C2,gamma2old+gamma1old-C2);
                H=min(C2,tau*C2+gamma2old+gamma1old);
            end
            
            if L==H
%                 fprintf('H==L\n');
                continue;
            end
            
            G11=G(gamma1,gamma1);
            G12=G(gamma1,gamma2);
            G22=G(gamma2,gamma2);
            kernelParameter=G11-2*G12+G22;
            
            if kernelParameter<=0
%                 fprintf('gammas: G11-2*G12+G22<=0\n');
                continue;
            end
            
            gamma2new=gamma2old+2*param.lambda*(E1-E2)/kernelParameter;
            gamma2new=min(max(gamma2new,L),H);
            
            if abs(gamma2new-gamma2old)<=0.01
%                 fprintf('change small\n');
                continue;
            end
      
            gamma1new=gamma1old+y1*y2*(gamma2old-gamma2new);
            
            %updating bias b
            b_k2new=-E2-y1*G12*(gamma1new-gamma1old)/(2*param.lambda)-y2*G22*(gamma2new-gamma2old)/(2*param.lambda)+bOld;
            b_k1new=-E1-y1*G11*(gamma1new-gamma1old)/(2*param.lambda)-y2*G12*(gamma2new-gamma2old)/(2*param.lambda)+bOld;
            
            gammas(gamma1-k,1)=gamma1new;
            gammas(gamma2,1)=gamma2new;
            changed=1;
        end
      end
      if(b1new~=0)||(b2new~=0)||(b_k1new~=0)||(b_k2new~=0)
          b=(b1new+b2new+b_k1new+b_k2new)/4;
      else 
          b=bOld;
      end
    end
        iter=iter+1;
end
part1=repmat((alphas.*totalData_Y).*G_x,1,dim_data).*totalData_X;
part2=repmat((gammas.*totalData_Y(k+1:num_data)).*G_x(k+1:num_data,:),1,dim_data).*totalData_X(k+1:num_data,:);
delta=-sum(part1,1)/(2*param.lambda*theta^2)+sum(part2,1)/(2*param.lambda*theta^2);%+constant_C*G_u*mean_noisedS/(2*param.lambda*theta^2)
mu=-sqrt(beta/(delta*(segma^-1)*delta'))*delta*(segma^-1)+mean_noisedS;
time=time+1;

G_mu=calckernel('rbf',theta,mu,totalData_X);
%term1 :ai aj yi yj G(xi,xj)
term1=((alphas.*totalData_Y)*(alphas.*totalData_Y)').*G;
%term2 :ai rj yi yj G(xi,xj)
term2=repmat((alphas.*totalData_Y),1,sizeofN).*repmat((gammas.*totalData_Y(k+1:num_data,:))',num_data,1).*G(:,k+1:num_data);
%term3 :ri rj yi yj G(xi,xj)
term3=((gammas.*totalData_Y(k+1:num_data,:))*(gammas.*totalData_Y(k+1:num_data,:))').*G(k+1:num_data,k+1:num_data);

object=sum(alphas,1)+sum(gammas,1)-sum(sum(term1))/(4*param.lambda)+sum(sum(term2))/(2*param.lambda)-sum(sum(term3))/(4*param.lambda)+sum(alphas.*totalData_Y.*G_mu)*constant_C/(2*param.lambda)-sum(gammas.*totalData_Y(k+1:num_data).*G_mu(k+1:num_data))*constant_C/(2*param.lambda)-constant_C^2/(2*param.lambda);
obj=[obj;object];
end

time = toc;

%% testing
num_test=size(tstX,1);
GT1=calckernel('rbf',theta,tstX(1,:),totalData_X);
gtt=calckernel('rbf',theta,tstX,totalData_X);
gmu=calckernel('rbf',theta,tstX,mu);
f=sum(repmat(alphas.*totalData_Y,1,num_test).*gtt,1)/(2*param.lambda)-sum(repmat(gammas.*totalData_Y(k+1:num_data),1,num_test).*gtt(k+1:num_data,:),1)/(2*param.lambda)-constant_C*gmu/(2*param.lambda)+repmat(b,1,num_test);
predict_label=sign(f');

tp = sum((predict_label == 1) & (tstY == 1)); % True Positive
fp = sum((predict_label == 1) & (tstY == -1)); % False Positive
fn = sum((predict_label == -1) & (tstY == 1)); % False Negative
tn = sum((predict_label == -1) & (tstY == -1)); % True Negative
acc = (tp + tn) / (tp + fp + fn + tn);
pre = tp / (tp + fp);
rec = tp / (tp + fn);
F1 = 2 * (pre * rec) / (pre + rec);
end