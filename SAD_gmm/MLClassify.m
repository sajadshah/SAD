function [ labels, likelihoods ] = MLClassify( spgmm, nspgmm, testData )
    pSpeech = pdf(spgmm, testData);
    pNonspeech = pdf(nspgmm, testData);
    likelihoods.speech = pSpeech;
    likelihoods.nonspeech = pNonspeech;
    labels = zeros(size(testData,1),1);
    for i=1:size(testData,1)
        if(pSpeech(i) > pNonspeech(i))  %recognized as speech
            labels(i) = 1;
        else                            %recognized as nonspeech
            labels(i) = -1;
        end
    end

end

