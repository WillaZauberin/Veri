clc
clear
%% 步骤 1: 读取和预处理数据
% 路径设置
wavPath = 'E:/XinYuan/USTC_AAA/pc/';
txtPath = 'E:/XinYuan/USTC_AAA/predict/';

% 获取所有wav文件
wavFiles = dir(fullfile(wavPath, '*.wav'));

% 初始化特征和标签存储
allFeatures = [];
labels = [];

% 处理每个文件
for k = 1:length(wavFiles)
    wavFileName = wavFiles(k).name;
    wavFilePath = fullfile(wavPath, wavFileName);
    txtFileName = replace(wavFileName, '.wav', '.txt');
    txtFilePath = fullfile(txtPath, txtFileName);
    
    % 读取音频文件
    [audio, fs] = audioread(wavFilePath);
    audio = audio(:,1); % 确保是单声道

    % 检查对应的txt文件是否存在
    if exist(txtFilePath, 'file')
        fileID = fopen(txtFilePath, 'r');
        eventlabels = textscan(fileID, '%d,%d');
        fclose(fileID);
    else
        disp(['No corresponding txt file for ', wavFileName]);
        continue;
    end

    % 初始化音频片段标签
    audioLabels = zeros(length(audio), 1);

    % 提取关键词音频段并提取特征
    for j = 1:size(eventlabels{1}, 1)
        startSample = eventlabels{1}(j);
        endSample = eventlabels{2}(j);
        if startSample >= endSample
            disp(['Invalid frame indices for ', wavFileName, ' at index ', num2str(j)]);
            continue;
        end
        audioSegment = audio(startSample:endSample);
        audioLabels(startSample:endSample) = 1;

        % 提取 MFCC 特征
        mfccs = mfcc(audioSegment, fs);
        allFeatures = [allFeatures; mfccs];  % 收集特征
    end
    
    % 合并标签数据
    labels = [labels; audioLabels];
end

%% 步骤 3: 网络训练
% 确保特征提取器使用正确的采样率
afe = audioFeatureExtractor('SampleRate',fs, 'mfcc',true, 'mfccDelta', true, 'mfccDeltaDelta', true);
% 循环遍历每个音频样本
features = [];
for idx = 1:size(allFeatures, 1)
    audioIn = allFeatures(idx, :);  % 从已提取的特征中获取单个特征行
    mfccs = extract(afe, audioIn);
    
    % 如果有必要，调整 mfccs 的维度
    mfccs = mfccs';  % 转置以匹配 [特征数量, 时间步长]

    % 累积特征
    features = [features; mean(mfccs, 2)'];  % 取每个时间步的均值以简化处理
end

% 网络定义
layers = [
    sequenceInputLayer(size(features, 2))
    bilstmLayer(100, 'OutputMode', 'last')
    fullyConnectedLayer(2)  % 正样本和负样本
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
Y = categorical(labels(1:size(features,1)));

% 训练网络
net = trainNetwork(features, Y, layers, options);