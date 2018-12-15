data = [];
tags = []; % 0 means nonspeech and 1 means speech
dataIndex = 1;
speechfolder = fullfile(datasetFolder,'speech');
nonspeechfolder = fullfile(datasetFolder,'nonspeech');

splist = dir(fullfile(speechfolder,'*.wav'));

for i=1:length(splist)
    disp(splist(i).name);
    disp(['speech ',num2str(i) , '/' , num2str(length(splist))]);
    wavfile = fullfile(speechfolder,splist(i).name);
    try
        [tmp,fs] = audioread(wavfile);
        features = extractFeatures(tmp,fs,noMFCC,noPLP,frameSize);

        [~,name,~] = fileparts(wavfile);
        feafile = fullfile(speechfolder,[name,'.fea']);
        save (feafile, 'features', '-ASCII');
    catch E
        disp(['failded to load speech ', wavfile]);
    end
end

splist = dir(fullfile(nonspeechfolder,'*.wav'));
for i=1:length(splist)
    disp(['nonspeech ',num2str(i) , '/' , num2str(length(splist))])
    wavfile = fullfile(nonspeechfolder,splist(i).name);
    try
        [tmp,fs] = audioread(wavfile);
        features = extractFeatures(tmp,fs,noMFCC,noPLP,frameSize);

        [~,name,~] = fileparts(wavfile);
        feafile = fullfile(nonspeechfolder,[name,'.fea']);
        save(feafile, 'features', '-ASCII');
    catch E
        disp(['failded to load speech ', wavfile]);
    end
end