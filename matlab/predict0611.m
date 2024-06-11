clc;close all;clear all;
% 加载训练好的模型
load('trainedModel.mat', 'net');

% 读取音频文件
[audioData, fs] = audioread('pc/0-3.wav');
audioData=audioData(:,1);
audioData=audioData(40320:74880);
% 音频特征提取器设置
windowLength = 512;
overlapLength = 384;
afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(windowLength, 'periodic'), 'OverlapLength', overlapLength, ...
    'mfcc', true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 提取特征
mfccs = extract(afe, audioData);
mfccs(~isfinite(mfccs)) = 0;  % 处理无效数据
featureMean = mean(mfccs, 1);

% 确保特征是作为序列传入，即使它只有一个时间步长
featureSequence = {featureMean};  % 使用元胞数组包装特征

% 使用classify进行预测
predictedLabel = classify(net, featureSequence);

% 显示预测结果
disp(['Predicted Label: ', char(predictedLabel)]);


