function [] = splitFeaturesTo_S_NS(featureFolder, name, srtFile, features, frameSize, frameStep)
%SPLITFEATURESTO_S_NS Summary of this function goes here
%   Detailed explanation goes here
    [srtData] = loadSrtSegments(srtFile);
    time = frameStep;
    fIndex = 1;
    featureFile = fullfile(featureFolder, [name,'.fea']);
    fid = fopen(featureFile,'w');
    try
        for segment=srtData
            tag = getNumericalTag(segment.Tag);
            while(time < segment.TimeEnd && fIndex < size(features,1))        
                f = features(fIndex, :);
                fIndex = fIndex + 1;            
                fprintf(fid, '%s ', num2str(tag));
                for i=1:length(f)
                    fprintf(fid, '%s ', num2str(f(i)));
                end
                fprintf(fid, '\n');
                time = time + frameStep;
            end
        end
    catch E
        disp(E);
    end
    fclose(fid);

end
function [x] = getNumericalTag(str)
    if strcmp(str,'speech')
        x= 1;
    elseif strcmp(str,'nonspeech')
        x= 0;
    else
        x=-1; %never comes here!
    end
    
end
function [data]=loadSrtSegments(srtFile)
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
        row.Tag = x;
        while ~isempty(x) && ~feof(fid)
            x=fgetl(fid);
            row.Line = [row.Line,' ',x];            
        end
        data(Numb) = row;
        Numb = Numb + 1;        
    end
end

function [begin,eind]=strToBeginEndTimes(TIMES)
    begin=str2num(TIMES(1:2))*3600+str2num(TIMES(4:5))*60+str2num(TIMES(7:8))+str2num(TIMES(10:12))/1000;
    eind=str2num(TIMES(18:19))*3600+str2num(TIMES(21:22))*60+str2num(TIMES(24:25))+str2num(TIMES(27:29))/1000;
end

