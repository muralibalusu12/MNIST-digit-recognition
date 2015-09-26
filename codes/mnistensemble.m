% Solving a Classification Problem using an Ensemble of Decision Trees
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

% Ensemble of ' Decision Trees '
tree = templateTree('Prune','on','MergeLeaves','on','MinLeaf',1);

%training an ensemble of classifiers with no. of classifiers defined by K=20
ensemble=fitensemble(traindata,trainlabels,'AdaBoostM2',20,tree);
    
%pedicting labels on test data
testlabels=predict(ensemble,testdata);

A=[1:1:28000];
finallabels=[A;testlabels];

fileID = fopen('ensemblelabels.csv','w');
fprintf(fileID,'%5s\n','Label');
fprintf(fileID,'%6.2f\n',labels);
fclose(fileID);

%percentage=96.629%

%%%%%%%%%%%%%% using 30 decision trees %%%%%%%%%%%%%%%%
%training an ensemble of classifiers with no. of classifiers defined by K=20
ensemble=fitensemble(traindata,trainlabels,'AdaBoostM2',30,tree);
    
%pedicting labels on test data
testlabels=predict(ensemble,testdata);

%percentage=97.000%


