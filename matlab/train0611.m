clc; close all; clear all;

%% 步骤 1: 读取和预处理数据
% 路径设置
wavPath = 'E:\XinYuan\USTC_AAA\pc';
txtPath = 'E:\XinYuan\USTC_AAA\predict';

% 获取所有wav文件
wavFiles = dir(fullfile(wavPath, '*.wav'));

% 初始化数据存储
data = {};
labels = {};

% 正样本和负样本的标识符映射
labelMap = containers.Map('KeyType', 'double', 'ValueType', 'char');
labelMap(1) = 'VeriSilicon'; % VeriSilicon
labelMap(3) = '大 V 大 V'; % 大 V 大 V
labelMap(6) = 'Hi 芯原'; % Hi 芯原
labelMap(7) = '测体温'; % 测体温
labelMap(8) = '测血压'; % 测血压
labelMap(9) = '测血糖'; % 测血糖

% 处理25个文件
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

windowLength = 512;
overlapLength = 384;
hopLength = 256; % 可以根据需要调整
afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(windowLength, 'periodic'), 'OverlapLength', overlapLength, ...
    'mfcc', true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 初始化特征存储，使用cell数组来处理不同长度的特征
features = cell(length(data), 1);

for i = 1:length(data)
    audioIn = data{i};
    try
        mfccs = extract(afe, audioIn');
        if isempty(mfccs)
            disp(['MFCC extraction failed for clip ', num2str(i)]);
            continue;
        end
        % 处理无穷大或NaN值
        mfccs(~isfinite(mfccs)) = 0;

        % 计算所有MFCC向量的均值，确保特征长度一致
        featureMean = mean(mfccs, 1); % 计算每列的均值
        features{i} = featureMean;
    catch ME
        disp(['Error extracting MFCC for clip ', num2str(i), ': ', ME.message]);
    end
end

% 移除空cell元素
features = features(~cellfun('isempty', features));

% 将cell数组转换为矩阵，用于后续处理
featuresMatrix = vertcat(features{:});



%% 步骤 3: 网络训练
% 网络定义
layers = [
    sequenceInputLayer(size(features, 2))
    bilstmLayer(100, 'OutputMode', 'last')
    fullyConnectedLayer(6) % 包括所有正样本和负样本的类别
    softmaxLayer
    classificationLayer
];

% 训练选项
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 27, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% 转换标签为分类变量
Y = categorical(labels);

% 训练网络
net = trainNetwork(features, Y, layers, options);
save('trainedModel.mat', 'net');
disp(net);
