function mfcc=AudioFeaExtractor(windowLength,overlapLength,fs)

    
    % 参数定义
    frameLength = 40000;
    frameStep = 1600;
    numCoeffs = 13;
     
    % 窗函数
    % window = hann(frameLength, 'periodic');
    % windowedFrames = frames .* window;
    % 
    % FFT 和功率谱
    NFFT = 512;
    powerSpectrum = abs(fft(windowLength, NFFT)).^2;
    
    % 梅尔滤波器组
    mfb = melFilterBank(20, NFFT, fs);
    melSpectrum = log(mfb * powerSpectrum(1:(NFFT/2+1), :));
    
    % DCT
    mfcc = dct(melSpectrum);
    mfcc = mfcc(2:(numCoeffs+1), :); % 取MFCC系数
end
