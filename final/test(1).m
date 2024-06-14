clc;
clear;

% 加载预训练的模型
load('trainedModel.mat', 'net');
% 识别语音位置

% 遍历每个检测到的语音段，提取特征并分类
labels = [];
times = [];
for i = 1:size(speechIdx, 1)

    segmentStart = speechIdx(i, 1);
    segmentEnd = speechIdx(i, 2);
    segmentData = audioIn(segmentStart:segmentEnd);
    features = [];

    % 音频特征提取器设置
windowLength = 240;
overlapLength = 120;
fs=8000;
afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(windowLength, 'periodic'), 'OverlapLength', overlapLength, ...
    'mfcc', true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 提取特征
mfccs = extract(afe, segmentData);
mfccs(~isfinite(mfccs)) = 0;  % 处理无效数据
featureMean = mean(mfccs, 1);

% 确保特征是作为序列传入，即使它只有一个时间步长
featureSequence = {featureMean};  % 使用元胞数组包装特征

% 使用classify进行预测
predictedLabel(i) = classify(net, featureSequence);

end
