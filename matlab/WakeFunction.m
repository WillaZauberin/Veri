[audioFiles, fs] = audioread('E:\XinYuan\VeriHealthi_QEMU_SDK.202405_preliminary\VeriHealthi_QEMU_SDK.202405_preliminary\VeriHealthi_Speech_Command_Dataset\data\pc\0-1.wav');
[features, labels] = extractFeatures(audioFiles, fs);
function [features, labels] = extractFeatures(audioFiles, fs, labels)
    % 创建特征提取器
    afe = audioFeatureExtractor('SampleRate',fs,'Window',hamming(round(0.03*fs),'periodic'), ...
        'OverlapLength',round(0.015*fs),'mfcc',true,'mfccDelta',true,'mfccDeltaDelta',true);
    setExtractorParams(afe,'mfcc','NumCoeffs',13);
    
    % 初始化特征和标签数组
    numFiles = numel(audioFiles);
    features = [];
    labelsList = [];

    for i = 1:numFiles
        audioIn = audioread(audioFiles{i});
        audioIn = (audioIn - mean(audioIn)) / std(audioIn); % 归一化
        audioFeatures = extract(afe, audioIn);
        
        features = [features; audioFeatures];
        labelsList = [labelsList; repmat(labels(i), size(audioFeatures, 1), 1)];
    end
end