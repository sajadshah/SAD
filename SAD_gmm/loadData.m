% datasetFolder has been already read from params.mat
if exist('data.mat','file')
    load('data.mat');
    return ;    
end
speechfolder = fullfile(datasetFolder,'speech');
nonspeechfolder = fullfile(datasetFolder,'nonspeech');
%%loading speech data
speechTestFiles = loadTestFileNames(datasetFolder,'speech');
splist = dir(fullfile(speechfolder,'*.fea'));
no_speechFiles = ceil(length(splist));
no_allspframes = 0;
no_allspframesForTest = 0;
t = 0;
for i=1:no_speechFiles
    disp(['speech for preallo ',num2str(i),'/',num2str(length(splist))]);
    feafile = fullfile(speechfolder,splist(i).name);
    allFeatures = load(feafile);
    if(isNotForTest(splist(i).name,speechTestFiles))
        no_allspframes = no_allspframes + size(allFeatures,2);
    else
        no_allspframesForTest = no_allspframesForTest + size(allFeatures,2);
    end
end
no_features = size(allFeatures,1);
speechData = zeros(no_allspframes,no_features); %preallocation of array
speechTestData = zeros(no_allspframesForTest,no_features); %preallocation of array
disp(['speech array preallocated ',num2str(no_allspframes),' frames']);
index = 1;
indexTest = 1;
for i=1:no_speechFiles    
    disp(['speech loading',num2str(i),'/',num2str(length(splist))]);
    feafile = fullfile(speechfolder,splist(i).name);
    allFeatures = load(feafile);
    if(isNotForTest(splist(i).name,speechTestFiles))
        for j=1:size(allFeatures,2)
            %disp([num2str(i),': ',num2str(j),'/',num2str(size(allFeatures,2))]);
            frameFeatures = allFeatures(:,j);
            speechData(index,:) =  frameFeatures';
            index = index +1 ;
        end
    else
        for j=1:size(allFeatures,2)
            %disp([num2str(i),': ',num2str(j),'/',num2str(size(allFeatures,2))]);
            frameFeatures = allFeatures(:,j);
            speechTestData(indexTest,:) =  frameFeatures';
            indexTest = indexTest +1 ;
        end
    end
end

%%loading non speech data
nonspeechTestFiles = loadTestFileNames(datasetFolder,'nonspeech');
nsplist = dir(fullfile(nonspeechfolder,'*.fea'));
no_nonspeechFiles = ceil(length(nsplist));
no_allnspframes = 0;
no_allnspframesForTest = 0;
for i=1:no_nonspeechFiles
    disp(['non speech for preallo ',num2str(i),'/',num2str(length(nsplist))]);
    feafile = fullfile(nonspeechfolder,nsplist(i).name);
    allFeatures = load(feafile);
    if(isNotForTest(nsplist(i).name,nonspeechTestFiles))
        no_allnspframes = no_allnspframes + size(allFeatures,2);
    else
        no_allnspframesForTest = no_allnspframesForTest + size(allFeatures,2);
    end
end
no_features = size(allFeatures,1);
nonspeechData = zeros(no_allnspframes,no_features); %preallocation of array
nonspeechTestData = zeros(no_allnspframesForTest,no_features); %preallocation of array
disp(['nonspeech array preallocated ',num2str(no_allnspframes),' frames']);
index = 1;
indexTest = 1;
for i=1:no_nonspeechFiles
    disp(['non speech loading',num2str(i),'/',num2str(length(nsplist))]);
    feafile = fullfile(nonspeechfolder,nsplist(i).name);
    allFeatures = load(feafile);
    if(isNotForTest(nsplist(i).name,nonspeechTestFiles))
        for j=1:size(allFeatures,2)
            %disp([num2str(i),': ',num2str(j),'/',num2str(size(allFeatures,2))]);
            frameFeatures = allFeatures(:,j);
            nonspeechData(index,:) = frameFeatures';
            index = index +1 ;
        end
    else
        for j=1:size(allFeatures,2)
            %disp([num2str(i),': ',num2str(j),'/',num2str(size(allFeatures,2))]);
            frameFeatures = allFeatures(:,j);
            nonspeechTestData(indexTest,:) =  frameFeatures';
            indexTest = indexTest +1 ;
        end
    end
end

removeNans();
createContextFeaturesInDataMatrix();

save('data.mat','speechData','speechTestData','nonspeechData','nonspeechTestData');