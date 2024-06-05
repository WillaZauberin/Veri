%% 读取原始声音数据hello
[audioData, fs] = audioread('E:\XinYuan\VeriHealthi_QEMU_SDK.202405_preliminary\VeriHealthi_QEMU_SDK.202405_preliminary\VeriHealthi_Speech_Command_Dataset\data\pc\0-1.wav');

%% 语音唤醒关键词检测
function isWakeWord = detectWakeWord(audioData, fs)
    % 提取特征
    coefficients = mfcc(audioData, fs);
    % 假设你已经有一个训练好的模型 'wakeWordModel.mat'
    load('wakeWordModel.mat', 'model');
    % 进行分类
    isWakeWord = predict(model, coefficients);
end
%% 命令词识别
function command = recognizeCommand(audioData, fs)
    % 提取特征
    coefficients = mfcc(audioData, fs);
    % 假设你已经有一个训练好的命令词模型 'commandModel.mat'
    load('commandModel.mat', 'model');
    % 进行分类
    [predictedLabel, ~] = classify(model, coefficients);
    command = convertLabelToCommand(predictedLabel);
end
%% 
% 假设 audioData 是完整的输入数据
%fs = 48000; % 示例采样频率
isWakeWordDetected = detectWakeWord(audioData, fs);

if isWakeWordDetected
    fprintf('Hi 芯原! 等待命令...\n');
    % 假设 commandData 是唤醒词后的数据
    command = recognizeCommand(commandData, fs);
    fprintf('识别到命令：%s\n', command);
else
    fprintf('未检测到唤醒词。\n');
end