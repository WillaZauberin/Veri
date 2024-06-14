function predictedLabel=ClassifyAudio(net, featureSequence)
predictedLabel=[];
predictedLabel = classify(net, featureSequence);
end
