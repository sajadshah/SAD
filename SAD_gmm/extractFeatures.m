function [ features ] = extractFeatures(samples, fs, noMFCC, noPLP, frameSize)
%EXTRACTFEATURES Summary of this function goes here
%   options is a list containing the following parameters:
%       FrameSize
%       noMFCC
%       noPLP
    mfcc = melfcc(samples, fs, 'numcep',noMFCC);
    plp = rastaplp(samples, fs,1,noPLP);
    
    features = [mfcc; plp]';

end

