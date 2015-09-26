% Solving a Classification Problem using KNN
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
mdl = fitcknn(traindata,trainlabels,'NumNeighbors',5);

%pedicting labels on test data
testlabels = predict(mdl,testdata);

A=[1:1:28000];
finallabels=[A;testlabels];

fileID = fopen('knnlabels.csv','w');
fprintf(fileID,'%5s\n','Label');
fprintf(fileID,'%6.2f\n',testlabels);
fclose(fileID);

%percentage_accuracy=96.829%

%
%
