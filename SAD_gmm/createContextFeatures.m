function [ cData ] = createContextFeatures( data, noFramesAround )
%CREATECONTEXTFEATURES Summary of this function goes here
%   Detailed explanation goes here
    noFrames = size(data,1);

    reducedData = compute_mapping(data,'PCA',1);
    
    context = zeros(noFrames, 2*noFramesAround);
    for i=1:noFrames
        if(i < noFramesAround+1 || i > noFrames-noFramesAround-1)
            
        else
            for j=1:noFramesAround
                context(i,j)=reducedData(i-j);
                context(i,j+noFramesAround)=reducedData(i+j);
            end
        end
    end
    cData = [data, context];
end

