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
%% 训练
% 预设
% 预设
numCoeffs = 13;  % MFCC系数的数量
audioFiles = dir('E:\XinYuan\USTC_AAA\pc');
features = [];
labels = [];

for i = 1:length(audioFiles)
    % 读取音频文件
    [audioIn, fs] = audioread(fullfile(audioFiles(i).folder, audioFiles(i).name));
    
    % 提取MFCC，确保参数名称正确
    coeffs = mfcc(audioIn, fs, 'NumCoeffs', numCoeffs);
    
    % 平均MFCC以得到单一特征向量
    avgCoeffs = mean(coeffs, 1);
    
    % 存储特征和标签
    features = [features; avgCoeffs];
    labels = [labels; extractBefore(audioFiles(i).name, '_')];  % 假设标签在"_"前
end

% 训练SVM分类器
svmModel = fitcecoc(features, labels);
