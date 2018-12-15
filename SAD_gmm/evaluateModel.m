function [ info ] = evaluateModel( spgmm, nspgmm, spTestData, nspTestData )
%EVALUATEMODEL Summary of this function goes here
%   output:
%       info: a structure contains FA, FR, EER, Accuracy,
%       FMeasure, ConfusionMatrix
    
    no_classes = 2;
    tp = zeros(no_classes,1);
    tn = zeros(no_classes,1);
    fp = zeros(no_classes,1);
    fn = zeros(no_classes,1);
    precision = zeros(no_classes,1);
    recall = zeros(no_classes,1);
    fmeasure = zeros(no_classes,1);
    confm = zeros(no_classes,no_classes);
    

    
    [spTestLabels,spTestLL] = MLClassify(spgmm,nspgmm,spTestData);
    spTestLabels = postProcess(spTestLabels);
    for i=1:length(spTestData)
        if( spTestLabels(i) == 1 ) % true classification of speech test data
            confm(1,1) = confm(1,1)+1;
        else
            confm(1,2) = confm(1,2)+1;
        end
    end
    
    [nspTestLabels,nspTestLL] = MLClassify(spgmm,nspgmm,nspTestData);
    nspTestLabels = postProcess(nspTestLabels);
    for i=1:length(nspTestData)
        if (nspTestLabels(i) == -1) %true classification of nonspeech data
            confm(2,2) = confm(2,2)+1;
        else
            confm(2,1) = confm(2,1)+1;
        end
    end
    
    
    save('likelihoods.mat','spTestLL','nspTestLL');
    %for i=1:no_classes
    tp = confm(1,1);
    fp = confm(2,1);
    fn = confm(1,2);
    tn = confm(2,2);
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    fmeasure = (2*precision*recall)/(precision+recall);
    %end 
    
    %for i=1:no_classes
    %    confm(i,:)=confm(i,:)/sum(confm(i,:));
    %end

    info.Accuracy = sum(diag(confm))/sum(sum(confm));
    info.FMeasure = fmeasure;    
    info.ConfusionMatrix = confm;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %for i=1:no_classes
    %    [ far,  frr ] = far_frr( tp, fp, tn, fn );
    %end 
    
   tmp = fp + tn;
  if tmp == 0
    far = NaN;
  else
    far = fp / tmp;
  end
  
  % To remember: FRR evaluated over "desired 1"
  tmp = fn + tp;
  if tmp == 0
    frr = NaN;
  else
    frr = fn / tmp;
  end
    
    info.FA = far;
    info.FR = frr;
    
    %rocCurve(spTestLL,nspTestLL);
    
end

