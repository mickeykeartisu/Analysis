import wave
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import librosa
import librosa.display
import torch

class ShortTimeFourierTransformer():
    """ default constructor """
    def __init__(self, inputFilePath, outputFilePath, frameSize, frameShift, title, xlabel, ylabel):
        self.__inputFilePath = inputFilePath    # input file path(.wav)
        self.__outputFilePath = outputFilePath  # output file path(.png)
        self.__frameSize = frameSize    # frame size [ms]
        self.__frameShift = frameShift  # frame shift [ms]
        self.__fftSize = 2**10  # FFT size (dimension)
        self.__xlabel = xlabel  # x axis label
        self.__ylabel = ylabel  # y axis label
        self.__title = title    # figure title

        # set parameters
        self.readInforamtion()
        self.transformUnit()
        self.setAppropriatlyFftSize()
        self.__totalFrameSize = (self.totalSampleSize - self.frameSize) // self.frameShift + 1   # total frame size
        self.initializeSpectrogram()
        self.calculateSpectrogram()
        self.calculateAmplitudedSpectrogram()

        # display information
        self.displayInformation()

    # ---------- Getters ---------- #
    """ input file path getter """
    @property
    def inputFilePath(self):
        return self.__inputFilePath

    """ output file path getter """
    @property
    def outputFilePath(self):
        return self.__outputFilePath

    """ sampling frequency getter """
    @property
    def samplingFrequency(self):
        return self.__samplingFrequency

    """ quantization size getter """
    @property
    def quantizationSize(self):
        return self.__quantizationSize
    
    """ channel getter """
    @property
    def channel(self):
        return self.__channel
    
    """ total sample size getter """
    @property
    def totalSampleSize(self):
        return self.__totalSampleSize
    
    """ original signal getter """
    @property
    def originalSignal(self):
        return self.__originalSignal

    """ x axis label getter """
    @property
    def xlabel(self):
        return self.__xlabel

    """ y axis label getter """
    @property
    def ylabel(self):
        return self.__ylabel

    """ transformed signal (int 32) getter """
    @property
    def transformedSignal(self):
        return self.__transformedSignal
    
    """ frame size getter """
    @property
    def frameSize(self):
        return self.__frameSize
    
    """ frame shift getter """
    @property
    def frameShift(self):
        return self.__frameShift
    
    """ fft size getter """
    @property
    def fftSize(self):
        return self.__fftSize
    
    """ total frame size getter """
    @property
    def totalFrameSize(self):
        return self.__totalFrameSize
    
    """ spectrogram getter """
    @property
    def spectrogram(self):
        return self.__spectrogram
    
    """ title getter """
    @property
    def title(self):
        return self.__title

    """ amplituded spectrogram getter """
    @property
    def amplitudedSpectrogram(self):
        return self.__amplitudedSpectrogram
    
    # ---------- Setters ---------- #
    """ input file path setter """
    @inputFilePath.setter
    def inputFilePath(self, inputFilePath):
        if inputFilePath == '':
            raise ValueError("input file path is empty.")
        self.__inputFilePath = inputFilePath

    """ output file path setter """
    @outputFilePath.setter
    def outputFilePath(self, outputFilePath):
        if outputFilePath == "":
            raise ValueError("output file path is empty")
        self.__outputFilePath = outputFilePath

    """ sampling frequency setter """
    @samplingFrequency.setter
    def samplingFrequency(self, samplingFrequency):
        if samplingFrequency <= 0:
            raise ValueError("sampling frequency is less than or equal 0.")
        self.__samplingFrequency = samplingFrequency
    
    """ quantization size setter"""
    @quantizationSize.setter
    def quantizationSize(self, quantizationSize):
        if quantizationSize <= 0:
            raise ValueError("quantization size is less than or equal 0.")
        self.__quantizationSize = quantizationSize
    
    """ channel setter """
    @channel.setter
    def channel(self, channel):
        if channel<= 0:
            raise ValueError("channel is less than or equal 0.")
        self.__channel = channel
    
    """ total sample size setter """
    @totalSampleSize.setter
    def totalSampleSize(self, totalSampleSize):
        if totalSampleSize <= 0:
            raise ValueError("total sample size is less than or equal 0.")
        self.__totalSampleSize = totalSampleSize

    """ original signal setter """
    @originalSignal.setter
    def originalSignal(self, originalSignal):
        self.__originalSignal = originalSignal
    
    """ x axis label setter """
    @xlabel.setter
    def xlabel(self, xlabel):
        if xlabel == "":
            raise ValueError("xlabel is emply.")
        self.__xlabel = xlabel

    """ y axis label setter """
    @ylabel.setter
    def ylabel(self, ylabel):
        if ylabel == "":
            raise ValueError("ylabel is emply.")
        self.__ylabel = ylabel

    """ transformed signal (int 32) setter """
    @transformedSignal.setter
    def transformedSignal(self, transformedSignal):
        self.__transformedSignal = transformedSignal
    
    """ frame size setter """
    @frameSize.setter
    def frameSize(self, frameSize):
        if frameSize <= 0:
            raise ValueError("frame size is less than or equal 0.")
        self.__frameSize = frameSize

    """ frame shift setter """
    @frameShift.setter
    def frameShift(self, frameShift):
        if frameShift <= 0:
            raise ValueError("frame shift is less than or equal 0.")
        self.__frameShift = frameShift
    
    """ fft size setter """
    @fftSize.setter
    def fftSize(self, fftSize):
        if fftSize <= 0:
            raise ValueError("fft size is less than or equal 0.")
        self.__fftSize = fftSize
    
    """ total frame size setter """
    @totalFrameSize.setter
    def totalFrameSize(self, totalFrameSize):
        if totalFrameSize <= 0:
            raise ValueError("total frame size is less than or equal 0.")
        self.__totalFrameSize = totalFrameSize
    
    """ spectrogram setter """
    @spectrogram.setter
    def spectrogram(self, spectrogram):
        self.__spectrogram = spectrogram
    
    """ title setter """
    @title.setter
    def title(self, title):
        if title == "":
            raise ValueError("title is empty.")
        self.__title = title
    
    """ amplituded spectrogram setter """
    @amplitudedSpectrogram.setter
    def amplitudedSpectrogram(self, amplitudedSpectrogram):
        self.__amplitudedSpectrogram = amplitudedSpectrogram

    # ---------- Methods ---------- #
    """ read information about wave file """
    def readInforamtion(self):
        # open file and read information
        with wave.open(self.inputFilePath) as waveFile:
            self.__samplingFrequency = waveFile.getframerate()  # sampling frequency [Hz]
            self.__quantizationSize = waveFile.getsampwidth()   # quantization size [bit]
            self.__channel = waveFile.getnchannels()  # channel (1 : mono or 2 : stereo)
            self.__totalSampleSize = waveFile.getnframes()  # total sample size
            self.__originalSignal = waveFile.readframes(self.totalSampleSize)   # original signal (int 24)
            self.__transformedSignal = self.setCuda(torch.tensor([unpack("<i", bytearray([0]) + self.originalSignal[self.quantizationSize * k:self.quantizationSize * (k+1)])[0] for k in range(self.totalSampleSize)], dtype=torch.float32)) # transformed signal (int 32)
    
    """ display information about wave file """
    def displayInformation(self):
        print(f"--------------------------------------------------")
        print(f"----------- ShortTimeFourierTransformer ----------")
        print(f"input file path : {self.inputFilePath}")
        print(f"output file path : {self.outputFilePath}")
        print(f"sampling frequency : {self.samplingFrequency} [Hz]")
        print(f"quantization size : {self.quantizationSize} [Byte]")
        print(f"channel : {self.channel}")
        print(f"total sample size : {self.totalSampleSize}")
        print(f"transformed signal shape : {self.transformedSignal.shape}")
        print(f"frame size : {self.frameSize} [samples]")
        print(f"frame shift : {self.frameShift} [samples]")
        print(f"total frame size : {self.totalFrameSize} [frames]")
        print(f"FFT Size : {self.fftSize}")
        print(f"spectrogram shape : {self.spectrogram.shape}")
        print(f"amplituded spectrogram shape : {self.amplitudedSpectrogram.shape}")
        print(f"--------------------------------------------------\n")
    
    """ transform unit ([ms] -> [sampleNumber]) """
    def transformUnit(self):
        self.frameSize = int(self.frameSize * self.samplingFrequency * 0.001)
        self.frameShift = int(self.frameShift * self.samplingFrequency * 0.001)
    
    """ set appropriate fft size """
    def setAppropriatlyFftSize(self):
        while self.fftSize < self.frameSize:
            self.fftSize *= 2
    
    """ initialize spectrogram """
    def initializeSpectrogram(self):
        self.__spectrogram = torch.tensor((self.totalFrameSize, self.frameSize // 2 + 1))   # spectrogram

    """ get hamming window """
    def getHammingWindow(self):
        window = self.setCuda(torch.arange(self.frameSize)) # window size array
        hammingWindow = 0.5 - 0.46 * torch.cos(2 * torch.pi * window / self.frameSize) # hamming window
        return hammingWindow

    """ calculate spectrogram """
    def calculateSpectrogram(self):
        hammingWindow = self.getHammingWindow()  # hamming window
        spectrogram = self.setCuda(torch.zeros((self.totalFrameSize, self.fftSize // 2 + 1), dtype=torch.complex64))  # spectrogram

        # Conduct fft
        for frameIndex in range(self.totalFrameSize):
            extractedSignal = hammingWindow * self.transformedSignal[frameIndex * self.frameShift : frameIndex * self.frameShift + self.frameSize]  # signal multipled by hamming window
            # Generate spectral
            spectrogram[frameIndex, :] = self.setCuda(
                torch.fft.rfft(
                    extractedSignal,  # signal
                    n=self.fftSize  # FFT point size
                )
            )
        self.spectrogram = spectrogram
    
    """ calculate amplituded spectrogram """
    def calculateAmplitudedSpectrogram(self):
        amplitudedSpectrogram = self.setCuda(torch.zeros(self.spectrogram.shape))

        for i in range(self.spectrogram.shape[0]):
            amplitudedSpectrogram[i, :] = self.setCuda(
                torch.sqrt(self.spectrogram[i, :].real ** 2 + self.spectrogram[i, :].imag ** 2)
            )
        self.amplitudedSpectrogram = amplitudedSpectrogram

    """ set CUDA """
    def setCuda(self, torchArray):
        if torch.cuda.is_available():
            torchArray = torchArray.to("cuda")
        return torchArray

    """ display spectrogram """
    def displaySpectrogram(self):
        # Generate canvas and axes
        figure, axes = plt.subplots(
            nrows=1,  # row size
            ncols=1,  # column size
            figsize=(16, 12),  # fig size
            sharex=True,  # share X axis between all axes
            sharey=True # share Y axis between all axes
        )

        # Display spectrogram
        spectrogramImage = librosa.display.specshow(
            data=librosa.amplitude_to_db(
                torch.abs(self.spectrogram).to("cpu").numpy(),  # input amplituded spctrogram
                ref=np.finfo(np.float32).max / (3.5 * 10 ** 27),  # search array
            ).T,  # input spectrogram
            hop_length=self.frameShift,  # X axis scale
            sr=self.samplingFrequency, # samplingRate
            x_axis="time",  # value of X axis
            y_axis="hz", # value of Y axis
            ax=axes, # axes
            vmin=-80,
            vmax=0
        )

        # Set figure detail
        figure.colorbar(
            mappable=spectrogramImage,  # input map
            ax=axes  # axes
        )

        # Set axes detail No Mask
        axes.set_title(self.title) # axes title
        axes.set_xlim(1.6, 2.65) # X axis range
        axes.set_ylim(0, 20000) # Y axis range
        axes.set_xlabel(self.xlabel) # X axis label
        axes.set_ylabel(self.ylabel) # Y axis label

        # Set not convolute
        figure.tight_layout()

        # save figure to output file path
        plt.savefig(self.outputFilePath)

        # close and clear memory
        plt.clf()
        plt.close()

# to test
if __name__ == "__main__":
    for i in range(1, 51):
        # Generate wave file plotter object
        shortTimeFourierTransformer = ShortTimeFourierTransformer(
            inputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト withMask/set1_withMask_word {i}.wav",
            outputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト withMask STFT figure/set1_withMask_STFT_figure_word {i}.svg",
            frameSize=32,
            frameShift=8,
            title = f"With Mask Spectrogram word {i}",
            xlabel="Time [s]",
            ylabel="Frequency [Hz]"
        )

        # plot wave file and save
        shortTimeFourierTransformer.displaySpectrogram()

        # Generate wave file plotter object
        shortTimeFourierTransformer = ShortTimeFourierTransformer(
            inputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト withMask/set1_withMask_word {i}.wav",
            outputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト withMask STFT figure/set1_withMask_STFT_figure_word {i}.jpeg",
            frameSize=32,
            frameShift=8,
            title = f"With Mask Spectrogram word {i}",
            xlabel="Time [s]",
            ylabel="Frequency [Hz]"
        )

        # plot wave file and save
        shortTimeFourierTransformer.displaySpectrogram()

        shortTimeFourierTransformer = ShortTimeFourierTransformer(
            inputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト noMask/set1_noMask_word {i}.wav",
            outputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト noMask STFT figure/set1_noMask_STFT_figure_word {i}.svg",
            frameSize=32,
            frameShift=8,
            title = f"No Mask Spectrogram word {i}",
            xlabel="Time [s]",
            ylabel="Frequency [Hz]"
        )

        # plot wave file and save
        shortTimeFourierTransformer.displaySpectrogram()

        shortTimeFourierTransformer = ShortTimeFourierTransformer(
            inputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト noMask/set1_noMask_word {i}.wav",
            outputFilePath=f"D:/名城大学/研究室/ゼミ/4モーラ単語リスト セット 1/4モーラ単語リスト noMask STFT figure/set1_noMask_STFT_figure_word {i}.jpeg",
            frameSize=32,
            frameShift=8,
            title = f"No Mask Spectrogram word {i}",
            xlabel="Time [s]",
            ylabel="Frequency [Hz]"
        )

        # plot wave file and save
        shortTimeFourierTransformer.displaySpectrogram()