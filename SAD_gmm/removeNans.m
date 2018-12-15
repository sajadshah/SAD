wasnan = any(isnan(speechData),2);
hadNaNs = any(wasnan);
if hadNaNs
    disp('nan!');
    speechData = speechData(~wasnan,:);
end

wasnan = any(isnan(nonspeechData),2);
hadNaNs = any(wasnan);
if hadNaNs
    disp('nan!');
    nonspeechData = nonspeechData(~wasnan,:);
end