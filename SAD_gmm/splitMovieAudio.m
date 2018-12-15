function [ totalSpeech, totalNonspeech ] = splitMovieAudio( srtFile, audioFile, outputAudiosLength, joinLimit, outputFolder)
%SPLITMOVIEAUDIO Summary of this function goes here
%   Detailed explanation goes here
    
    [srtData] = loadSpeechSegments(srtFile);
    [srtData] = joinCloseSubtitleSegments(srtData,joinLimit);
    [srtData] = addNonspeechSegments(srtData);
    [audioParts, totalSpeech, totalNonspeech] = joinSegmentsToParts(srtData, outputAudiosLength);
    saveAudioParts(audioFile, audioParts, outputFolder);
end

function [data]=loadSpeechSegments(srtFile)
    fid=fopen(srtFile,'r');
    Numb = 1;    
    while ~feof(fid)
        Numb_tmp=str2num(fgetl(fid));
        if isempty(Numb_tmp)
            continue
        end
        row.Times = fgetl(fid);
        [row.TimeBegin,row.TimeEnd]=strToBeginEndTimes(row.Times);       
        x = fgetl(fid);
        row.Line = x;
        row.Tag='speech';
        while ~isempty(x) && ~feof(fid)
            x=fgetl(fid);
            row.Line = [row.Line,' ',x];            
        end
        data(Numb) = row;
        Numb = Numb + 1;        
    end
end

function [data]=addNonspeechSegments(srtData)
    index = 1;
    timeBegin = 0;
    for i=1:length(srtData)
        row.TimeBegin = timeBegin;
        row.TimeEnd = srtData(i).TimeBegin;
        timeBegin = srtData(i).TimeEnd;
        row.Times = beginEndTimesToStr(row.TimeBegin, row.TimeEnd);
        row.Tag = 'nonspeech';
        row.Line = 'nonspeech';
        data(index) = row;
        index = index+1;
        
        s = srtData(i);
        row.TimeBegin = s.TimeBegin;
        row.TimeEnd = s.TimeEnd;
        row.Times = beginEndTimesToStr(row.TimeBegin, row.TimeEnd);
        row.Tag = s.Tag;
        row.Line = s.Line;
        data(index)=row;
        index = index+1;
    end
end

function [begin,eind]=strToBeginEndTimes(TIMES)
    begin=str2num(TIMES(1:2))*3600+str2num(TIMES(4:5))*60+str2num(TIMES(7:8))+str2num(TIMES(10:12))/1000;
    eind=str2num(TIMES(18:19))*3600+str2num(TIMES(21:22))*60+str2num(TIMES(24:25))+str2num(TIMES(27:29))/1000;
end
function [TIMES]=beginEndTimesToStr(beginTime, endTime)
    TIMES = [secsToStr(beginTime), ' --> ', secsToStr(endTime)];
end
function [str]=secsToStr(time)
    nhours = floor(time/3600);
    str = num2str(nhours, '%02d');
    nmins = floor((time - 3600*nhours)/60);
    str = [str, ':', num2str(nmins, '%02d')];
    nsecs = floor(time - 3600*nhours - 60*nmins);
    str = [str, ':', num2str(nsecs, '%02d')];
    nmsecs = floor((time - 3600*nhours - 60*nmins - nsecs) * 1000);
    str = [str, ',', num2str(nmsecs, '%03d')];
end

function [ jointData ] = joinCloseSubtitleSegments( srtData , joinLimit )
%JOINCLOSESUBTITLESEGMENTS 
%   

    jDataIndex = 1;
    jointData(1) = srtData(1);
    index = 2;
    while index <= length(srtData)
        if(srtData(index).TimeBegin < jointData(jDataIndex).TimeEnd + joinLimit)
            jointData(jDataIndex).TimeEnd = srtData(index).TimeEnd;
            jointData(jDataIndex).Line = [jointData(jDataIndex).Line, srtData(index).Line]; 
        else
            jDataIndex = jDataIndex + 1;
            jointData(jDataIndex) = srtData(index);
        end
        index = index + 1;
    end
end

function [audioParts, totalSpeech, totalNonspeech] = joinSegmentsToParts(srtData, outputAudiosLength)
    totalSpeech = 0;
    totalNonspeech = 0;
    pIndex = 1;
    pBegin = 0;
    segmentIndex = 1;
    partLength = 0;    
    for i = 1:length(srtData)
        segment = srtData(i);
        duration = (segment.TimeEnd - segment.TimeBegin);
        if(duration <= 0)
            disp('negative duration');
        end
        partLength = partLength + duration;
        pEnd = segment.TimeEnd;
        segmentList(segmentIndex) = segment;
        segmentIndex = segmentIndex + 1;
        if strcmp(segment.Tag,'speech')
            totalSpeech = totalSpeech + duration;
        elseif strcmp(segment.Tag,'nonspeech')
            totalNonspeech = totalNonspeech + duration;
        end

        if(partLength > outputAudiosLength || i==length(srtData))
            part.TimeBegin = pBegin;
            part.TimeEnd = pEnd;
            pBegin = pEnd;
            part.list = segmentList;
            audioParts(pIndex) = part;
            pIndex = pIndex + 1;
            clear segmentList;
            segmentIndex = 1;
            partLength = 0;            
        end        
    end
    for i = 1:length(audioParts)
        part = audioParts(i);
        for j = 1:length(audioParts(i).list)
            segment = audioParts(i).list(j);
            audioParts(i).list(j).TimeBegin = segment.TimeBegin - part.TimeBegin;
            audioParts(i).list(j).TimeEnd = segment.TimeEnd - part.TimeBegin;
            audioParts(i).list(j).Times = beginEndTimesToStr(audioParts(i).list(j).TimeBegin, audioParts(i).list(j).TimeEnd);
        end
    end
end
function []=saveAudioParts(audioFile, audioParts, outputFolder)
    [~,prefix,~] = fileparts(audioFile);
    
    info = audioinfo(audioFile);
    fs = info.SampleRate;
    for i = 1:length(audioParts)
        part = audioParts(i);
        fileName = [prefix,'-',num2str(i)];
        outputFile = fullfile(outputFolder,[fileName,'.txt']);
        fid = fopen(outputFile,'w');
        for j=1:length(audioParts(i).list)
            segment = audioParts(i).list(j);
            fprintf(fid, '%s\n', num2str(j));
            fprintf(fid, '%s\n',segment.Times);
            fprintf(fid, '%s\n\n',segment.Tag);
        end
        fclose(fid);
        samples = [ceil(part.TimeBegin*fs)+1, ceil(part.TimeEnd*fs)];
        [y,~] = audioread(audioFile,samples);
        monoY = (y(:,1) + y(:,2)) / 2;
        outputFile = fullfile(outputFolder,[fileName,'.wav']);
        audiowrite(outputFile,monoY,fs);    
    end
end
