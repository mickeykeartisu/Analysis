import wave
import numpy as np
import matplotlib.pyplot as plt
from struct import unpack
import torch

# wave file plotter class
class WaveFilePlotter():
    """ default constructor """
    def __init__(self, inputFilePath, outputFilePath, xlabel, ylabel, title):
        self.__inputFilePath = inputFilePath    # input file path(.wav)
        self.__outputFilePath = outputFilePath  # output file path(.png)
        self.__xlabel = xlabel  # x axis label
        self.__ylabel = ylabel  # y axis label
        self.__title = title    # title
        self.readInforamtion()

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

    """ title getter """
    @property
    def title(self):
        return self.__title

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

    """ transformed signal (float 32) setter """
    @transformedSignal.setter
    def transformedSignal(self, transformedSignal):
        self.__transformedSignal = transformedSignal

    """ title setter """
    @title.setter
    def title(self, title):
        if title == "":
            raise ValueError("title is empty.")
        self.__title = title

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
            self.__transformedSignal = self.setCuda(torch.tensor([unpack("<i", bytearray([0]) + self.originalSignal[self.quantizationSize * k:self.quantizationSize * (k+1)])[0] for k in range(self.totalSampleSize)], dtype=torch.float32))  # transformed signal (float 32)
    
    """ display information about wave file """
    def displayProperties(self):
        print(f"----------------------------------------------------------")
        print(f"input file path : {self.inputFilePath}")
        print(f"output file path : {self.outputFilePath}")
        print(f"sampling frequency : {self.samplingFrequency} [Hz]")
        print(f"quantization size : {self.quantizationSize} [Byte]")
        print(f"channel : {self.channel}")
        print(f"total sample size : {self.totalSampleSize}")
        print(f"transformed signal shape : {self.transformedSignal.shape}")
        print(f"device : {torch.cuda.get_device_name()}\n")
        print(f"----------------------------------------------------------\n")
    
    """ plot wave file """
    def plotAndSaveWaveFile(self):
        timeAxis = np.arange(self.totalSampleSize) / self.samplingFrequency    # time axis range

        # make canvas
        plt.figure(figsize=(16, 13))

        # plot signal
        plt.plot(timeAxis, self.transformedSignal.to("cpu").detach().numpy().copy())

        # set axis label
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.title(self.title)

        # limit x axis range
        plt.xlim([0, self.totalSampleSize / self.samplingFrequency])
        plt.ylim([-2 ** 30, 2 ** 30])

        # save figure to output file path
        plt.savefig(self.outputFilePath)

        # close and clear memory
        plt.clf()
        plt.close()

    """ set up GPU """
    def setCuda(self, tensorArray):
        if torch.cuda.is_available():
            tensorArray = tensorArray.to("cuda")
        return tensorArray

# do test
if __name__ == "__main__":
    for i in range(1, 51):
        # Generate wave file plotter object
        waveFilePlotter = WaveFilePlotter(
            inputFilePath="./4モーラ単語リスト セット 1/4モーラ単語リスト noMask/set1_noMask_word " + str(i) + ".wav",
            outputFilePath="./4モーラ単語リスト セット 1/4モーラ単語リスト noMask signal figure/set1_noMask_signal_figure_word " + str(i) + ".jpeg",
            xlabel="Time [s]",
            ylabel="Value",
            title = f"set1_noMask_word {i}"
        )

        # plot wave file and save
        waveFilePlotter.plotAndSaveWaveFile()

        # display information
        waveFilePlotter.displayProperties()