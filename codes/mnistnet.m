 % Solving a Classification Problem using an Ensemble of Decision Trees
% Script written by Murali Raghu Babu B
%

%importing training data from files.
traind=importdata('train.csv');
traindata=traind.data;
trainlabels=traindata(:,1);
traindata=traindata(:,2:785);
traindlabels2=importdata('trainout');
clearvars -except traindata trainlabels

%importing testing data from files.
test=importdata('test.csv');
testdata=test.data; 
clear test;


% Creating a Pattern Recognition Network
hiddenlayer=[10 10];
mlp=patternnet(hiddenlayer);

%Division of Data for Training, Validation, Testing
%divideFcnn is dividerand (default)
mlp.divideParam.trainRatio = 70/100;            
mlp.divideParam.valRatio = 15/100;
mlp.divideParam.testRatio = 15/100;


% Training the Network
[mlp, tr]=train(mlp,traindata',trainlabels2');

% testing the network
testlabels=mlp(testdata');

[m n]=size(testlabels);
newlabels=zeros(n,1);

for i=1:n
    max=testlabels(1,i);
    maxi=1;
    for j=1:m
        if(max<testlabels(j,i))
            max=testlabels(j,i);
            maxi=j;       
        end
    end
   newlabels(i)=maxi-1;
end

testlabels =newlabels;

A=[1:1:28000];
finallabels=[A;testlabels];

fileID = fopen('netlabels.csv','w');
fprintf(fileID,'%5s\n','Label');
fprintf(fileID,'%6.2f\n',testlabels);
fclose(fileID);

%percentage_accuracy=89.757%

