% Solving a Classification Problem using Decision tree
% Script written by Murali Raghu Babu B
%

%importing training data from files.
traind=importdata('train.csv');
traindata=traind.data;
trainlabels=traindata(:,1);
traindata=traindata(:,2:785);
clearvars -except traindata trainlabels

%importing testing data from files.
test=importdata('test.csv');
testdata=test.data; 
clear test;


%training a KNN classifier with nearest neighbors parameter defined by K
mdl = fitctree(traindata,trainlabels);

%pedicting labels on test data
testlabels = predict(mdl,testdata);

A=[1:1:28000];
finallabels=[A;testlabels];

fileID = fopen('dtreelabels.csv','w');
fprintf(fileID,'%5s','Label');
fprintf(fileID,'%6.2f\n',labels);
fclose(fileID);

%percentage=85.486%
