function [ tpRates, fpRates ] = rocCurve ( spLL, nspLL)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    noSpeechTestData = size(spLL.speech,1);
    noNonspeechTestData = size(nspLL.speech,1);
    maxLL = max(max([spLL.speech;spLL.nonspeech]) , max([nspLL.speech;nspLL.nonspeech]));
    %minLL = min(min([spLL.speech;spLL.nonspeech]) , min([nspLL.speech;nspLL.nonspeech]));
    minThr = -4.73834791098923e-12;%-maxLL; %-2.29159623192170e-11;
    maxThr = 3.52866716953154e-14;%maxLL;  %3.65101772543222e-13;
    step = (maxThr-minThr)/5000;
    thresholds = [minThr:step:maxThr];
    tpRates = zeros(size(thresholds,2),1);
    fpRates = zeros(size(thresholds,2),1);
    confs = zeros(2,2,size(thresholds,2));
    index = 1;
    for t=thresholds
        disp(index);
        conf = zeros(2,2);
        for i=1:noSpeechTestData
            if spLL.speech(i) > spLL.nonspeech(i) + t   
                %true classification of speech data
                conf(1,1) = conf(1,1)+1;
            else
                conf(1,2) = conf(1,2)+1;
            end
        end
        %disp('speech classified');
        for i=1:noNonspeechTestData
            if nspLL.nonspeech(i) + t > nspLL.speech(i)  
                %true classification of speech data
                conf(2,2) = conf(2,2)+1;
            else
                conf(2,1) = conf(2,1)+1;
            end
        end
        %disp('non speech classified');
        tp = conf(1,1);
        fp = conf(2,1);
        fn = conf(1,2);
        tn = conf(2,2);        
        tpr = tp / (tp+fn);
        fpr = fp / (fp+tn);
        tpRates(index) = tpr;
        fpRates(index) = fpr;
        confs(:,:,index) = conf;
        index=index+1;
    end
    save('roc_info.mat','thresholds','tpRates','fpRates','confs')
    
end

