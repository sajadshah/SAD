% datasetFolder has been already read from params.mat
speechfolder = fullfile(datasetFolder,'speech');
nonspeechfolder = fullfile(datasetFolder,'nonspeech');
percentageOfData = 0.2;
%%loading speech data
splist = dir(fullfile(speechfolder,'*.fea'));
no_speechFiles = ceil(length(splist));
no_allspframes = 0;
for i=1:no_speechFiles
    feafile = fullfile(speechfolder,splist(i).name);
    allFeatures = load(feafile);
    no_allspframes = no_allspframes + size(allFeatures,2);
end
speechTestFileNames=[];
no_speechSelectedTestFrames = 0;
y = randperm(no_speechFiles);
i=1;
while (no_speechSelectedTestFrames < no_allspframes*percentageOfData)
    index = y(i);
    feafile = fullfile(speechfolder,splist(index).name);
    allFeatures = load(feafile);
    no_speechSelectedTestFrames = no_speechSelectedTestFrames + size(allFeatures,2);
    [~,name,~] = fileparts(splist(index).name);
    speechTestFileNames(i).name = name;
    i=i+1;
end
[~,n]=size(speechTestFileNames);
fileID = fopen(fullfile(datasetFolder,'speechTestFiles.txt'),'w');
for i=1:n
    fprintf(fileID,'%s\n',speechTestFileNames(i).name);
end
fclose(fileID);
%%loading non speech data
nsplist = dir(fullfile(nonspeechfolder,'*.fea'));
no_nonspeechFiles = ceil(length(nsplist));
no_allnspframes = 0;
for i=1:no_nonspeechFiles
    feafile = fullfile(nonspeechfolder,nsplist(i).name);
    allFeatures = load(feafile);
    no_allnspframes = no_allnspframes + size(allFeatures,2);
end
nonspeechTestFileNames=[];
no_nonspeechSelectedTestFrames = 0;
y = randperm(no_nonspeechFiles);
i=1;
while (no_nonspeechSelectedTestFrames < no_allnspframes*percentageOfData)
    index = y(i);
    feafile = fullfile(nonspeechfolder,nsplist(index).name);
    allFeatures = load(feafile);
    no_nonspeechSelectedTestFrames = no_nonspeechSelectedTestFrames + size(allFeatures,2);
    [~,name,~] = fileparts(nsplist(index).name);
    nonspeechTestFileNames(i).name = name;
    i=i+1;
end
[~,n]=size(nonspeechTestFileNames);
fileID = fopen(fullfile(datasetFolder,'nonspeechTestFiles.txt'),'w');
for i=1:n
    fprintf(fileID,'%s\n',nonspeechTestFileNames(i).name);
end
fclose(fileID);