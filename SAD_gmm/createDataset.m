mdbAddress = 'D:\Sajad\Dataset_MOVIES';
mlist = dir(fullfile(mdbAddress , '!*'));

% datasetFolder has been already read from params.mat
if( exist(datasetFolder,'dir') )
    rmdir(datasetFolder,'s')
end
mkdir(datasetFolder,'speech');
mkdir(datasetFolder,'nonspeech');

speechGaurd = 0.4;
nonspeechGaurd = 0.4;
joinLimit = 0.6;
totalSpeech = 0;
totalNonspeech = 0;

for i=1:length(mlist)
    clear X;
    mfolder = fullfile(mdbAddress,mlist(i).name);    
    disp(['movie ', mfolder]);
    [X, srtData,tsp, tnsp] = readSpeechAndNonSpeechFromMovieBuffered(mfolder,...
                                          speechGaurd,nonspeechGaurd,... %gaurd size in ms
                                          joinLimit);       %join limit
    totalSpeech = totalSpeech + tsp;
    totalNonspeech = totalNonspeech + tnsp;
    disp('saving movie audios');
    for j=1:length(X)
        disp([num2str(j) , '/' , num2str(length(X))]);
        if strcmp(X(j).Tag,'speech')
            ofolder = fullfile('dataset','speech');
        elseif strcmp(X(j).Tag,'nonspeech')
            ofolder = fullfile('dataset','nonspeech');
        end    
        ofile = fullfile(ofolder,[num2str(i),'-',num2str(j),'.wav']);
        audiowrite(ofile,X(j).Samples,X(j).Fs);    
    end
    
%     X = cell2struct(...
%         cat(3,struct2cell(X),struct2cell(tmp)),...
%         fieldnames(X),...
%         1);
end

fileID = fopen(fullfile(datasetFolder,'info.txt'),'w');
fprintf(fileID,'speech guard: %f\n', speechGaurd);
fprintf(fileID,'non speech guard: %f\n', nonspeechGaurd);
fprintf(fileID,'join Limit: %f\n', joinLimit);
fprintf(fileID,'total nonspeech: %s\n', secs2hms(totalNonspeech));
fprintf(fileID,'total speech: %s\n', secs2hms(totalSpeech));
fclose(fileID);

