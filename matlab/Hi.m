clear
clc
% 定义音频文件所在的目录
audioFolder = 'E:\XinYuan\USTC_AAA\pc';
audioFiles = dir(fullfile(audioFolder, '*.wav'));  % 获取所有的 .wav 文件
%% 给音频打标签
% 初始化用于存储文件名和标签的变量
fileNames = {audioFiles.name}';
labels = cell(length(fileNames), 1);

% 循环处理每个文件，分配标签
for i = 1:length(audioFiles)
    fileName = audioFiles(i).name;
    % 根据文件名前缀分配标签
    if contains(fileName, '-7')
        labels{i} = 'measure_temperature';  % 测体温
    elseif contains(fileName, '-8')
        labels{i} = 'measure_blood_pressure';  % 测血压
    elseif contains(fileName, '-9')
        labels{i} = 'measure_blood_sugar';  % 测血糖
    elseif contains(fileName, '-6')
        labels{i} = 'wake_word';  % 唤醒词
    else
        labels{i} = 'invalid_command';  % 无效命令
    end
end

% 创建一个表格来保存文件名和标签
labelTable = table(fileNames, labels, 'VariableNames', {'FileName', 'Label'});

% 保存这个表格到 CSV 文件
csvFileName = 'audio_labels.csv';
writetable(labelTable, fullfile(audioFolder, csvFileName));
%fprintf('Label data saved to %s\n', fullfile(audioFolder, csvFileName));

%% 训练模型
% 步骤 1: 加载数据
dataFolder = 'E:\XinYuan\USTC_AAA\pc';
tbl = readtable(fullfile(dataFolder, 'audio_labels.csv'));

% 步骤 2: 预处理和特征提取
features = [];
labels = {};
for i = 1:height(tbl)
    audioFilename = fullfile(dataFolder, tbl.FileName{i});
    [audioIn,fs] = audioread(audioFilename);
    audioFeature = mfcc(audioIn, fs, 'LogEnergy', 'Ignore','NumCoeffs', 26);
    features = [features; mean(audioFeature, 1)];
    labels = [labels; {tbl.Label{i}}];
end

% 步骤 3: 分割数据为训练集和测试集
cv = cvpartition(size(features, 1), 'HoldOut', 0.2);
idxTrain = training(cv);
idxTest = test(cv);
XTrain = features(idxTrain,:);
YTrain = categorical(labels(idxTrain));
XTest = features(idxTest,:);
YTest = categorical(labels(idxTest));

% 步骤 4: 创建和训练模型
layers = [
    sequenceInputLayer(26)%这是网络的输入层
    % 表示输入特征的数量，即每个输入样本的特征向量长度。
    lstmLayer(50,'OutputMode','last')%LSTM 层，具有 50 个神经元。
    % OutputMode 设置为 last 指只返回序列的最后一个输出，这对于许多序列处理任务（如分类）是常见的设置。
    fullyConnectedLayer(numel(unique(labels)))
    %全连接层，其节点数等于标签的唯一值数，即分类任务中的类别数。
    softmaxLayer
    %Softmax 层，用于产生一个概率分布，表示每个类别的预测概率
    classificationLayer
    %分类层，用于计算最终的分类损失
];
%定义了使用 Adam 优化算法的训练选项。
% Adam 是一种效果很好的随机梯度下降方法，适合于大多数深度学习任务。
options = trainingOptions('adam', ...
    'MaxEpochs',30, ...
    'MiniBatchSize', 50, ...
    'Plots', 'training-progress');
%最大迭代次数为 30。这意味着整个训练数据集将被反复使用 30 次来更新网络权重。
%每个小批量包含 50 个样本。小批量大小是平衡训练速度和内存消耗的重要参数。
%在训练过程中显示一个实时图表，图表展示了如训练损失和准确率等关键指标的进展。
net = trainNetwork(XTrain, YTrain, layers, options);
%使用指定的数据（XTrain 和 YTrain），层（layers），以及训练选项（options）来训练定义好的神经网络。
% XTrain 是输入特征，YTrain 是对应的标签。
%net是训练后的网络模型，可以用来对新数据进行预测。

% 步骤 5: 评估模型
YPred = classify(net, XTest);
accuracy = sum(YPred == YTest) / numel(YTest);
fprintf('Test set accuracy: %.2f%%\n', accuracy * 100);

% 步骤 6: 保存模型
save('TrainedKeywordSpottingModel.mat', 'net');
