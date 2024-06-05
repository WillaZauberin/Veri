%% ��ȡԭʼ��������hello
[audioData, fs] = audioread('E:\XinYuan\VeriHealthi_QEMU_SDK.202405_preliminary\VeriHealthi_QEMU_SDK.202405_preliminary\VeriHealthi_Speech_Command_Dataset\data\pc\0-1.wav');

%% �������ѹؼ��ʼ��
function isWakeWord = detectWakeWord(audioData, fs)
    % ��ȡ����
    coefficients = mfcc(audioData, fs);
    % �������Ѿ���һ��ѵ���õ�ģ�� 'wakeWordModel.mat'
    load('wakeWordModel.mat', 'model');
    % ���з���
    isWakeWord = predict(model, coefficients);
end
%% �����ʶ��
function command = recognizeCommand(audioData, fs)
    % ��ȡ����
    coefficients = mfcc(audioData, fs);
    % �������Ѿ���һ��ѵ���õ������ģ�� 'commandModel.mat'
    load('commandModel.mat', 'model');
    % ���з���
    [predictedLabel, ~] = classify(model, coefficients);
    command = convertLabelToCommand(predictedLabel);
end
%% 
% ���� audioData ����������������
%fs = 48000; % ʾ������Ƶ��
isWakeWordDetected = detectWakeWord(audioData, fs);

if isWakeWordDetected
    fprintf('Hi оԭ! �ȴ�����...\n');
    % ���� commandData �ǻ��Ѵʺ������
    command = recognizeCommand(commandData, fs);
    fprintf('ʶ�����%s\n', command);
else
    fprintf('δ��⵽���Ѵʡ�\n');
end