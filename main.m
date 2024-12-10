clc
close all
addpath(genpath(pwd));
%%
data=load('Talaromycosis.dat');
X = data(:, 1:end-1);
Y = data(:, end);
cvp = cvpartition(Y, 'KFold', 10);
accuracy = zeros(cvp.NumTestSets, 1);
sensitivity = zeros(cvp.NumTestSets, 1);
specificity = zeros(cvp.NumTestSets, 1);
precision = zeros(cvp.NumTestSets, 1);
MCC = zeros(cvp.NumTestSets, 1);
Fmeasure = zeros(cvp.NumTestSets, 1);
%%
cvFolds = crossvalind('Kfold', Y, 10);
testIdx={};
for k=1:10
    testIdx{k} = (cvFolds == k);
end
for i = 1:10
    disp(['Fold: ', num2str(i), ' ---------']);
    testIx = testIdx{i};
    trainIdx = ~testIx;
    [findonetest,~]=find(testIx==1);
    [findonetrain,~]=find(trainIdx==1);
    %% Optimization process
    N = 20;
    T = 50;
    nVar = size(data,2)-1;
    classifierFhd = Get_FKNN();
    Best_position = bIPCACO(N,T,nVar,data,findonetrain,findonetest,classifierFhd);
    [outClass1]  = classifierFhd(data(findonetrain,:), data(findonetrain,end), data(findonetest,:), data(findonetest,end),Best_position);
    predicted_label = outClass1;
    test_label = data(findonetest,end);
    %% Confusion matrix
    cm = confusionmat(test_label, predicted_label);
    TP = cm(2, 2);
    TN = cm(1, 1);
    FP = cm(1, 2);
    FN = cm(2, 1);
    accuracy(i) = (TP + TN) / (TP + TN + FP + FN);
    sensitivity(i) = TP / (TP + FN);
    specificity(i) = TN / (TN + FP);
    precision(i) = TP / (TP + FP);
    MCC(i) = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN));
    Fmeasure(i) = 2 * (precision(i) * sensitivity(i)) / (precision(i) + sensitivity(i));
end

meanAccuracy = mean(accuracy);
meanSensitivity = mean(sensitivity);
meanSpecificity = mean(specificity);
meanPrecision = mean(precision);
meanMCC = mean(MCC);
meanFmeasure = mean(Fmeasure);

fprintf('Average accuracy: %.2f%%\n', meanAccuracy * 100);
fprintf('Average sensitivity: %.2f%%\n', meanSensitivity * 100);
fprintf('Average specificity: %.2f%%\n', meanSpecificity * 100);
fprintf('Average precision: %.2f%%\n', meanPrecision * 100);
fprintf('Average MCC: %.2f\n', meanMCC);
fprintf('Average F-measure: %.2f\n', meanFmeasure);
