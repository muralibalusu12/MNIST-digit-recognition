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
fprintf(fileID,'%5s','Label');
fprintf(fileID,'%6.2f\n',labels);
fclose(fileID);

%percentage=96.829%

% %vector to hold accuracy for various KNN classifiers based on varying K
% percent=zeros(20,1);
% 
% for k=1:20
%     %training a KNN classifier with nearest neighbors parameter defined by
%     mdl = fitcknn(traindata,trainlabels,'NumNeighbors',k);
%     
%     %pedicting labels on test data
%     label = predict(mdl,testdata);
%     
%     %calculating accuracy
%     compare=(testl==label);
%     s=size(compare);
%     compare=sum(compare);    
%     s=s(1);
%     percent(k) = (compare/s)*100;  
% end
% 
%     %plotting accuracy of classification versus performance of the KNN classifiers based on K
%     figure1 = figure;
%     t=[1:1:20];
%     t=t';
% 
%     title('{\bf Accuracy of Classification using KNN- varying K}');
%     xlabel('Value of K- nearest neighbors') ;
%     ylabel('percentage of correct classification');
%     axis([0 20 90 100]); hold on;
%     plot(t,percent,'-rx'); hold on;
%     legend('Using KNN');
% 
%     saveas(figure1,'optdigitKNN.jpg') ; %saving the figure as a jpeg image