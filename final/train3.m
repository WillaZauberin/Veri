clc; close all; clear all;
%% 步骤 1: 读取和预处理数据hh
% 路径设置
wavPath = 'pc/';
txtPath = 'lable/';
% 定义存放bin文件的文件夹路径
folderPath = 'segment\';

% 获取文件夹中所有bin文件的信息
files = dir(fullfile(folderPath, '*.bin'));

% 初始化数据和标签存储的结构
data = {};
lables = {};

% 遍历文件数组
for i = 1:length(files)
    % 获取文件名
    filename = files(i).name;
    
    % 读取bin文件内容
    filepath = fullfile(folderPath, filename);
    fileID = fopen(filepath, 'r');
    audioData = fread(fileID, 'int16'); % 修改此处的数据类型为实际数据类型，如'int16', 'double'等
    fclose(fileID);
    
    % 将数据添加到data数组
    data{end+1} = audioData;
    
   % 解析文件名以获取标签
splitName = strsplit(filename, '-');
label = splitName{2};  % 将标签作为字符串存储
lables{end+1} = label;
end

%% 步骤 2: 特征提取
if isempty(data)
    error('No valid audio clips were extracted.');
end
fs=8000;
windowLength = 240;
overlapLength = 120;

afe = audioFeatureExtractor('SampleRate', fs, ...
    'Window', hann(windowLength, 'periodic'), 'OverlapLength', overlapLength, ...
    'mfcc', true, 'mfccDelta', true, 'mfccDeltaDelta', true);

% 初始化特征存储，使用cell数组来处理不同长度的特征
features = cell(length(data), 1);

for i = 1:length(data)
    audioIn = data{i};
    try
        mfccs = extract(afe, audioIn);
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
Y = categorical(lables);

% 训练网络
net = trainNetwork(features, Y, layers, options);
save('trainedModel.mat', 'net');
disp(net);
