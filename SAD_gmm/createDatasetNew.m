% joinLimit, outputAudiosLength, mdbAddress, datasetFolder  --> params.mat;
mlist = dir(fullfile(mdbAddress , '!*'));
if( exist(datasetFolder,'dir') )
    rmdir(datasetFolder,'s')
end

totalSpeech = 0; %seconds
totalNonspeech = 0; %seconds

mkdir(datasetFolder,'audio files');
outputFolder = fullfile(datasetFolder,'audio files');

for i=1:length(mlist)
    mfolder = fullfile(mdbAddress,mlist(i).name);
    disp(['movie ', mfolder]);
    
    msrtlist = dir(fullfile(mfolder,'*.srt'));
    msrtFile = fullfile(mfolder,msrtlist(1).name);
    maudiolist = dir(fullfile(mfolder, '*.mp3'));
    maudiofile = fullfile(mfolder,maudiolist(1).name);
    [ts, tns] = splitMovieAudio(msrtFile, maudiofile, outputAudiosLength, joinLimit, outputFolder);
    totalSpeech = totalSpeech + ts;
    totalNonspeech = totalNonspeech + tns;
    disp([num2str(totalSpeech), ' ' , num2str(totalNonspeech)]);
end

fileID = fopen(fullfile(datasetFolder,'info.txt'),'w');
fprintf(fileID,'join Limit: %f\n', joinLimit);
fprintf(fileID,'total nonspeech: %s\n', secs2hms(totalNonspeech));
fprintf(fileID,'total speech: %s\n', secs2hms(totalSpeech));
fclose(fileID);




