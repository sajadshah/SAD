%datasetFolder, noMFCC, noPLP, frameSize, frameStep --> params.mat;
audioFilesFolder = fullfile(datasetFolder, 'audio files');
mkdir(datasetFolder,'feature files');
featureFilesFolder = fullfile(datasetFolder, 'feature files');
wavList = dir(fullfile(audioFilesFolder,'*.wav'));

for i=1:length(wavList)
    disp(sprintf ([num2str(i),'\t/ ',num2str(length(wavList)),':\t',wavList(i).name]));
    wavFile = fullfile(audioFilesFolder,wavList(i).name);
    [~,name,~] = fileparts(wavFile);
    srtFile = fullfile(audioFilesFolder,[name,'.txt']);

    [samples, fs] = audioread(wavFile);
    features = extractFeatures(samples, fs, noMFCC, noPLP, frameSize);
    splitFeaturesTo_S_NS(featureFilesFolder, name, srtFile, features, frameSize, frameStep);    
end