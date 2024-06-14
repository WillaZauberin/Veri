clc;
clear;

% 加载预训练的模型
load('trainedModel.mat', 'net');

% 读取音频文件
filePath = "E:\XinYuan\USTC_AAA\pc\test\30-7.wav";
[audioIn, fs] = audioread(filePath);
audioIn = audioIn(:, 1); % 假设是单声道音频


% 设置音频特征提取器
windowLength = 5000;
overlapLength = 1000;
%detectSpeech(audioIn,fs,'Window',hann(windowLength,'periodic'),...
    % 'OverlapLength',1000);

% 检测语音部分
 [speechIdx, speechMask] = detectSpeech(audioIn,fs,'Window',hann(windowLength,'periodic'),...
    'OverlapLength',1000);
afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(windowLength, 'periodic'), 'OverlapLength', overlapLength, ...
    'mfcc', true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 遍历每个检测到的语音段，提取特征并分类
labels = [];
times = [];
for i = 1:size(speechIdx, 1)
    segmentStart = speechIdx(i, 1);
    segmentEnd = speechIdx(i, 2);
    segmentData = audioIn(segmentStart:segmentEnd);
    features = [];

    % 对音频段进行窗函数处理和特征提取
    startIndex = 1;
    hopLength = windowLength - overlapLength;
    while (startIndex + windowLength <= length(segmentData))
        endIndex = startIndex + windowLength - 1;
        windowData = segmentData(startIndex:endIndex);
        feature = extract(afe, windowData);
        feature(~isfinite(feature)) = 0;
        features = [features; mean(feature, 1)];
        startIndex = startIndex + hopLength;
    end

    % 使用预训练的模型对特征进行分类
    if ~isempty(features)
        for idx = 1:size(features, 1)
            predLabel = classify(net, features(idx, :));
            labels = [labels; predLabel];
            startTime = (segmentStart + (idx-1) * hopLength) / fs;
            endTime = (segmentStart + idx * hopLength) / fs;
            times = [times; [startTime, endTime]];
        end
    end
end

% 可视化音频和标记的语音区域
figure;
t = (0:length(audioIn)-1) / fs;
plot(t, audioIn);
xlabel('Time (seconds)');
ylabel('Amplitude');
hold on;
grid on;
% 绘制检测到的语音区域和分类标签
for i = 1:size(speechIdx, 1)
    text(speechIdx(i, 1), max(audioIn), char(labels(i)), ...
        'BackgroundColor', 'blue', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
end

title('Detected Speech and Keywords');
hold off;
