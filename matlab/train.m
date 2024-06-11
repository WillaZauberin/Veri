clc;close all;clear all;
%% 步骤 1: 读取和预处理数据
% 路径设置
wavPath = 'E:\XinYuan\USTC_AAA\pc/';
txtPath = 'E:\XinYuan\USTC_AAA\predict/';

% 获取所有wav文件
wavFiles = dir(fullfile(wavPath, '*.wav'));

% 初始化数据存储
data = {};
labels = {};

% 正样本的标识符映射
labelMap = containers.Map('KeyType','double','ValueType','double');
labelMap(6) = 1; % Hi 芯原
labelMap(7) = 2; % 测体温
labelMap(8) = 3; % 测血压
labelMap(9) = 4; % 测血糖

% 处理每个文件
for k = 1:25
    wavFileName = wavFiles(k).name;
    wavFilePath = fullfile(wavPath, wavFileName);
    txtFileName = replace(wavFileName, '.wav', '.txt');
    txtFilePath = fullfile(txtPath, txtFileName);
    
    % 读取音频文件
    [audioData, fs] = audioread(wavFilePath);
    
    % 解析文件名以确定标签
labelPartStr = extractBefore(extractAfter(wavFileName, '-'), '.');
labelPart = str2double(labelPartStr);

    
    % 检查对应的txt文件是否存在
    if exist(txtFilePath, 'file')
        frameIndices = load(txtFilePath);
    else
        disp(['No corresponding txt file for ', wavFileName]);
        continue;
    end
    
    % 提取关键词音频段
    for j = 1:size(frameIndices, 1)
        startIndex = max(1, frameIndices(j, 1));
        endIndex = min(length(audioData), frameIndices(j, 2));
        
        if startIndex >= endIndex
            disp(['Invalid or out-of-bound frame indices for ', wavFileName, ' at index ', num2str(j)]);
            continue;
        end
        
        keywordClip = audioData(startIndex:endIndex);
        
        % 根据标签分配标识符
        if labelMap.isKey(labelPart)
            label = labelMap(labelPart);
        else
            continue; % 如果标签不在列表中，跳过此文件
        end
        
        % 存储数据和标签
        data{end+1} = keywordClip;
        labels{end+1} = label;
    end
end



%% 步骤 2: 特征提取
if isempty(data)
    error('No valid audio clips were extracted.');
end

% 确保采样率是最初读取的第一个文件的采样率
afe = audioFeatureExtractor('SampleRate',fs, 'mfcc',true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 提取特征
features = [];
for i = 1:length(data)
    audioIn = data{i};
    mfccs = extract(afe, audioIn);
    if isempty(mfccs)
        disp(['MFCC extraction failed for clip ', num2str(i)]);
        continue;
    end
    features = [features; mean(mfccs, 2)']; % 取均值简化处理
end

if isempty(features)
    error('No features were extracted. Please check the input data and extraction process.');
end

%% 步骤 3: 网络训练
% 确保特征提取器使用正确的采样率
afe = audioFeatureExtractor('SampleRate',fs, 'mfcc',true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 提取特征
features = [];
for i = 1:length(data)
    audioIn = data{i};
    mfccs = extract(afe, audioIn);
    if isempty(mfccs)
        disp(['MFCC extraction failed for clip ', num2str(i)]);
        continue;
    end
    features = [features; mean(mfccs, 2)']; % 取均值简化处理
end

if isempty(features)
    error('No features were extracted. Please check the input data and extraction process.');
end

% 网络定义
layers = [
    sequenceInputLayer(size(features, 2))
    bilstmLayer(100, 'OutputMode', 'last')
    fullyConnectedLayer(5) % 四个正样本加一个可能的其他类别
    softmaxLayer
    classificationLayer
];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 27, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

% 转换标签为分类变量
Y = categorical(labels);

% 训练网络
net = trainNetwork(features, Y, layers, options);
