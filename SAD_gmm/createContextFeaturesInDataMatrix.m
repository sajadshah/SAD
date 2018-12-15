noFramesAround = 10;
speechData = createContextFeatures(speechData, noFramesAround);
speechTestData = createContextFeatures(speechTestData, noFramesAround);

nonspeechData = createContextFeatures(nonspeechData, noFramesAround);
nonspeechTestData = createContextFeatures(nonspeechTestData, noFramesAround);
