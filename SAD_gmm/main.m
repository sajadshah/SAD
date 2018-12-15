clear;

addpath(genpath([pwd,'/drtoolbox']));
addpath(genpath([pwd,'/mfccplptoolbox']));

load('params.mat');

%createDatasetNew();
createFeaturesInDatasetNew;
%createTestListFile();
disp('Features ready');