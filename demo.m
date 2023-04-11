clear;
load('Fertility.mat') 

noiserate=0.2;  %\eta 
fnoise=0.05;   %\nu 

[num,dim]=size(train_X);
P1=size(find(train_Y==1),1);  %The number of original positive samples
y11=zeros(round(P1*(1-noiserate)),1);y12=2*ones(round(P1*noiserate),1);y13=zeros(num-P1,1);   
Yu1=[y11;y12;y13];Y1=train_Y-Yu1;
%Generate random numbers that obey the normal distribution
R=normrnd(0,fnoise,num,dim); 
train_X=train_X+R;
[numt,dimt]=size(text_X);
Rt=normrnd(0,fnoise,numt,dimt); 
tstX=text_X+Rt;

beta=10^(-5); 
lambda=0.5;   
theta=8;   
tau=0.5;  

%Pin-KLFCS
[acc, F1]=PINKLFCS(train_X,train_Y,text_X,text_Y,noiserate,lambda,beta,theta,tau);

%Pin-LFCS
%[acc, F1, time]=PINLFCS(train_X,train_Y,text_X,text_Y,noiserate,lambda,beta,tau);

output=[acc, F1]


