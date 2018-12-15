if exist('data_normal.mat','file')
    load('data_normal.mat');
    return ;    
end

%trainSpMean = mean(speechData);
%trainSpStd = std(speechData);
%trainNspMean = mean(nonspeechData);
%trainNspStd = std(nonspeechData);
trainMean = mean([speechData;nonspeechData]);
trainStd = std([speechData;nonspeechData]);
speechData = normalizeMatrix(speechData,trainMean, trainStd);
speechTestData = normalizeMatrix(speechTestData,trainMean, trainStd);
nonspeechData = normalizeMatrix(nonspeechData,trainMean, trainStd);
nonspeechTestData = normalizeMatrix(nonspeechTestData,trainMean, trainStd);

save('data_normal.mat','speechData','speechTestData','nonspeechData','nonspeechTestData');