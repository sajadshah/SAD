from = 14;
to = 22;
speechData = seperateFeaturesOnData(speechData,from,to);
speechTestData = seperateFeaturesOnData(speechTestData,from,to);
nonspeechData = seperateFeaturesOnData(nonspeechData,from,to);
nonspeechTestData = seperateFeaturesOnData(nonspeechTestData,from,to);
speechData(1:10,:)

