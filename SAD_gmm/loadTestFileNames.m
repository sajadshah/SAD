function [ names ] = loadTestFileNames( datasetFolder,son ) %speech of nonspeech
%LOADTESTFILENAMES Summary of this function goes here
%   Detailed explanation goes here


    fileID = fopen(fullfile(datasetFolder,[son,'TestFiles.txt']),'r');
    names = [];
    tline = fgets(fileID);
    i = 1;
    while ischar(tline)
        names(i).name = strcat(tline);
        i=i+1;
        tline = fgets(fileID);
    end

end

