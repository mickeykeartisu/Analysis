import ShortTimeFourierTransformer
from ShortTimeFourierTransformer import *
import torch

class DPMatcher():
    """ default constructor """
    def __init__(self, inputSpectrogram1, inputSpectrogram2, outputFilePath):
        self.__inputSpectrogram1 = inputSpectrogram1    # input spectrogram 1
        self.__inputSpectrogram2 = inputSpectrogram2    # input spectrogram 2
        self.__outputFilePath = outputFilePath  # output file path
        
        self.setDistanceBuffer()    # set distance buffer
        self.calculateDistance()    # calculate distance
        print(f"distance shape : {self.distance.shape}")

        self.setCostBuffer()    # set cost buffer
        self.setTrackBuffer()   # set track buffer
        print(f"cost shape : {self.cost.shape}")
        print(f"track shape : {self.track.shape}")

        self.calculateCostAndTrack()    # calculate cost and track
        print(f"total cost shape : {self.totalCost.shape}")
        print(f"minimum cost path shape : {self.minimumCostPath.shape}")

        self.outputMinimumCostPath()    # output result of calculation
        self.calculateAlignSpectorgram()    # calculate aligned spectrogram

    # ---------- Getters ---------- #
    """ input spectrogram 1 getter """
    @property
    def inputSpectrogram1(self):
        return self.__inputSpectrogram1

    """ input spectrogram 2 getter """
    @property
    def inputSpectrogram2(self):
        return self.__inputSpectrogram2
    
    """ distance getter """
    @property
    def distance(self):
        return self.__distance

    """ cost getter """
    @property
    def cost(self):
        return self.__cost
    
    """ track getter """
    @property
    def track(self):
        return self.__track
    
    """ total cost getter """
    @property
    def totalCost(self):
        return self.__totalCost
    
    """ minimum cost path getter """
    @property
    def minimumCostPath(self):
        return self.__minimumCostPath

    """ output file path getter """
    @property
    def outputFilePath(self):
        return self.__outputFilePath

    """ aligned spectrogram getter """
    @property
    def alignedSpectrogram(self):
        return self.__alignedSpectrogram

    # ---------- Setters ---------- #
    """ input spectrogram 1 setter """
    @inputSpectrogram1.setter
    def inputSpectrogram1(self, inputSpectrogram1):
        self.__inputSpectrogram1 = inputSpectrogram1

    """ input spectrogram 2 setter """
    @inputSpectrogram2.setter
    def inputSpectrogram2(self, inputSpectrogram2):
        self.__inputSpectrogram2 = inputSpectrogram2
    
    """ distance setter """
    @distance.setter
    def distance(self, distance):
        self.__distance = distance
    
    """ cost setter """
    @cost.setter
    def cost(self, cost):
        self.__cost = cost

    """ track setter """
    @track.setter
    def track(self, track):
        self.__track = track

    """ total cost setter """
    @totalCost.setter
    def totalCost(self, totalCost):
        self.__totalCost = totalCost

    """ minimum cost path setter """
    @minimumCostPath.setter
    def minimumCostPath(self, minimumCostPath):
        self.__minimumCostPath = minimumCostPath

    """ output file path setter """
    @outputFilePath.setter
    def outputFilePath(self, outputFilePath):
        if outputFilePath == "":
            raise ValueError("output file path is emply.")
        self.__outputFilePath = outputFilePath

    """ aligned spectrogram setter """
    @alignedSpectrogram.setter
    def alignedSpectrogram(self, alignedSpectrogram):
        self.__alignedSpectrogram = alignedSpectrogram

    # ---------- Methods ---------- #
    """ set distance buffer method """
    def setDistanceBuffer(self):
        self.__distance = self.setCuda(torch.zeros((self.inputSpectrogram1.shape[0], self.inputSpectrogram2.shape[0])))
    
    """ calculate distance method """
    def calculateDistance(self):
        for row in range(self.inputSpectrogram1.shape[0]):
            for column in range(self.inputSpectrogram2.shape[0]):
                self.distance[row, column] = torch.sum((self.inputSpectrogram1[row] - self.inputSpectrogram2[column]) ** 2)
            if row % 10 == 0:
                print(f"distance : {row} / {self.inputSpectrogram1.shape[0]}")

    """ set cost buffer """
    def setCostBuffer(self):
        self.cost = self.setCuda(torch.zeros((self.distance.shape[0], self.distance.shape[1])))

    """ set track buffer """
    def setTrackBuffer(self):
        self.track = self.setCuda(torch.zeros(self.distance.shape[0], self.distance.shape[1]))
    
    """ calculate cost and track """
    def calculateCostAndTrack(self):
        self.cost[0, 0] = self.distance[0, 0]    # start point

        # transit vertical
        for row in range(1, self.inputSpectrogram1.shape[0] - 1):
            self.cost[row, 0] = self.cost[row - 1, 0] + self.distance[row, 0]
            self.track[row, 0] = 0
        
        # transit horizontal
        for column in range(1, self.inputSpectrogram2.shape[0] - 1):
            self.cost[0, column] = self.cost[0, column - 1] + self.distance[0, column]
            self.track[0, column] = 2
        
        # transit others
        for row in range(1, self.inputSpectrogram1.shape[0] - 1):
            for column in range(1, self.inputSpectrogram2.shape[0] - 1):
                verticalCost = self.cost[row - 1, column] + self.distance[row, column]  # cost of vertical transit
                diagonalCost = self.cost[row - 1, column - 1] + 2 * self.distance[row, column]  # cost of diagonal transit
                horizontalCost = self.cost[row, column - 1] + self.distance[row, column]    # cost of horizontal transit
                costCandidates = [verticalCost, diagonalCost, horizontalCost]   # candidates of cost
                transition = torch.argmin(torch.tensor(costCandidates))  # select minimum transition cost
                self.cost[row, column] = costCandidates[transition] # record cost
                self.track[row, column] = transition    # record cost index

            if row % 10 == 0:
                print(f"total cost : {row} / {self.inputSpectrogram1.shape[0] - 1}")
        
        # set parameters
        totalCost = self.cost[-1, -1] / (self.inputSpectrogram1.shape[0] + self.inputSpectrogram2.shape[0]) # total cost
        minimumCostPath = []    # minimum cost path
        row = self.inputSpectrogram1.shape[0] - 1 # start row point (back track)
        column = self.inputSpectrogram2.shape[0] - 1  # start column point (back track)

        while True:
            # end point
            if (row < 0) or (column < 0):
                break

            minimumCostPath.append([row, column])   # add path
            print(f"minimum cost path : [{row}, {column}]")

            # back track
            if self.track[row, column] == 0:
                row -= 1
            elif self.track[row, column] == 1:
                row -= 1
                column -= 1
            else:
                column -= 1
        
        self.__totalCost = self.setCuda(torch.tensor(totalCost))
        self.__minimumCostPath = self.setCuda(torch.tensor(minimumCostPath[::-1]))

    """ set CUDA """
    def setCuda(self, torchArray):
        if torch.cuda.is_available():
            torchArray = torchArray.to("cuda")
        return torchArray

    """ output minimum cost path """
    def outputMinimumCostPath(self):
        with open(self.outputFilePath, mode = "w") as openedFile:
            for path in self.minimumCostPath:
                openedFile.write(f"{path[0] : {path[1]}}\n")
    
    """ align spectrogram """
    def calculateAlignSpectorgram(self):
        self.__alignedSpectrogram = torch.zeros(self.inputSpectrogram2.shape)
        for alignmentIndex in range(len(self.minimumCostPath)):
            index = self.minimumCostPath[alignmentIndex][1]
            self.alignedSpectrogram[alignmentIndex, :] = self.inputSpectrogram2[index, :]

    """ display property method """
    def displayProperty(self):
        print(f"---------------------------------------------------------")
        print(f"input spectrogram 1 shape : {self.inputSpectrogram1.shape}")
        print(f"input spectrogram 2 shape : {self.inputSpectrogram2.shape}")
        print(f"distance shape : {self.distance.shape}")
        print(f"cost shape : {self.cost.shape}")
        print(f"track shape : {self.track.shape}")
        print(f"total cost shape : {self.totalCost.shape}")
        print(f"minimum cost path shape : {self.minimumCostPath.shape}")
        print(f"aligned spectrogram shape : {self.alignedSpectrogram.shape}")
        print(f"---------------------------------------------------------\n")

if __name__ == "__main__":
    noMaskSpectrogram = ShortTimeFourierTransformer(
        inputFilePath="./4モーラ単語リスト セット 1/4モーラ単語リスト noMask/set1_noMask_word " + str(1) + ".wav",
        outputFilePath="./4モーラ単語リスト セット 1/4モーラ単語リスト noMask STFT figure/set1_noMask_STFT_figure_word " + str(1) + ".svg",
        frameSize=60,
        frameShift=1,
        title = f"No Mask Spectrogram word {1}",
        xlabel="Time [s]",
        ylabel="Frequency [Hz]"
    )

    noMaskSpectrogram.displayInformation()

    withMaskSpectrogram = ShortTimeFourierTransformer(
        inputFilePath="./4モーラ単語リスト セット 1/4モーラ単語リスト withMask/set1_withMask_word " + str(1) + ".wav",
        outputFilePath="./4モーラ単語リスト セット 1/4モーラ単語リスト withMask STFT figure/set1_withMask_STFT_figure_word " + str(1) + ".svg",
        frameSize=60,
        frameShift=1,
        title = f"With Mask Spectrogram word {1}",
        xlabel="Time [s]",
        ylabel="Frequency [Hz]"
    )

    withMaskSpectrogram.displayInformation()

    dpMathcer = DPMatcher(
        inputSpectrogram1=noMaskSpectrogram.amplitudedSpectrogram,  # input spectrogram 1 (no mask)
        inputSpectrogram2=withMaskSpectrogram.amplitudedSpectrogram,    # input spectrogram 2 (with mask)
        outputFilePath="./output/resultOfDPMatching.txt"    # output file path
    )

    dpMathcer.displayProperty()