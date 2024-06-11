clear
clc
% 定义音频文件所在的目录
%audioFolder = 'E:\XinYuan\USTC_AAA\Labeled';
%audioFiles = dir(fullfile(audioFolder, '*.wav'));  % 获取所有的 .wav 文件
[audio, fs] = audioread('E:\XinYuan\USTC_AAA\data_divided\pc\Train\Hi\0-6.wav');  % 读取音频文件
coeffs = mfcc(audio, fs, 'LogEnergy', 'Replace');  % 提取MFCC
% 假设labels.txt中每行格式为：开始时间 结束时间 标签
opts = detectImportOptions('E:\XinYuan\USTC_AAA\data_divided\pc\Train\Hi\0-6.txt', 'Delimiter', ' ');
labelsData = readtable('E:\XinYuan\USTC_AAA\data_divided\pc\Train\Hi\0-6.txt', opts);

% 为每个标签的时间段计算对应的MFCC
for i = 1:height(labelsData)
    startTime = labelsData(i);
    endTime = labelsData.endTime(i);
    startIndex = round(startTime * fs);
    endIndex = round(endTime * fs);
    segment = audio(startIndex:endIndex);
    segmentCoeffs = mfcc(segment, fs, 'LogEnergy', 'Replace');

    %% 初始化包含特征和标签的表
featureCols = {'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5', ...
               'Feature6', 'Feature7', 'Feature8', 'Feature9', 'Feature10', ...
               'Feature11', 'Feature12', 'Feature13'};
featureTable = table([], [], [], [], [], [], [], [], [], [], [], [], [], ...
    'VariableNames', [featureCols, {'Label'}]);
for i = 1:numFrames
    segment = audio(startIndex:endIndex); % 假设这些索引已正确计算
    segmentCoeffs = mfcc(segment, fs); % 计算MFCC特征

    % 假设你有一个函数来确定这一帧的标签
    label = determineLabel(segment); % 这应是一个自定义函数

    % 将特征和标签添加到表中
    newRow = table(segmentCoeffs(1), segmentCoeffs(2), segmentCoeffs(3), segmentCoeffs(4), segmentCoeffs(5), ...
                   segmentCoeffs(6), segmentCoeffs(7), segmentCoeffs(8), segmentCoeffs(9), segmentCoeffs(10), ...
                   segmentCoeffs(11), segmentCoeffs(12), segmentCoeffs(13), ...
                   label, ...
                   'VariableNames', [featureCols, {'Label'}]);

    featureTable = [featureTable; newRow];
end
save('featureData.mat', 'featureTable');
writetable(featureTable, 'featureData.csv');

end
