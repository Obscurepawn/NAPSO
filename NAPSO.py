import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import time


class stock:
    def __init__(self, days: int = 6):
        self.earnRateArray = np.random.normal(
            random.random(), random.random(), days)
        self.averageEarnRate = np.mean(self.earnRateArray)


class particle:
    def __init__(self, stockNum: int = 10):
        self.investScale = self.divideOne(stockNum)

    def divideOne(self, stockNum: int) -> list:
        dividePoint = []
        investScale = []
        for i in range(stockNum-1):
            temp = random.random()
            if temp not in dividePoint:
                dividePoint.append(temp)
            else:
                i -= 1
        dividePoint.sort()
        investScale.append(dividePoint[0])
        for i in range(1, stockNum-1):
            investScale.append(dividePoint[i]-dividePoint[i-1])
        investScale.append(1-dividePoint[-1])
        return investScale

    def calFitness(self, stockSet: list, cov: np.ndarray, stockNum: int = 10) -> float:
        risk = 0
        profit = 0
        for i in range(stockNum):
            profit += stockSet[i].averageEarnRate*self.investScale[i]
            for j in range(stockNum):
                risk += self.investScale[i]*self.investScale[j]*cov[i][j]
        distance = (1/profit)**2 + risk**2
        return distance

    def neighbourSearch(self, iterTimes: int, maxTimes: int, investScale: list):
        acProb = (maxTimes-iterTimes)/maxTimes
        if random.random() > acProb:
            return
        pos1 = int(random.random()*len(investScale))
        pos2 = int(random.random()*len(investScale))
        reduce1 = random.random()
        reduce2 = random.random()
        difference1 = investScale[pos1]-investScale[pos1]*reduce1
        investScale[pos1] = investScale[pos1]*reduce1
        difference2 = investScale[pos2]-investScale[pos2]*reduce2
        investScale[pos2] = investScale[pos2]*reduce2
        addSum = difference1 + difference2
        self.randomAdd(investScale, addSum)

    def amend(self, investScale: list):
        lack = 1-sum(investScale)
        surplus = 0
        for i in range(len(investScale)):
            surplus += -investScale[i] if investScale[i] < 0 else 0
            investScale[i] = 0 if investScale[i] < 0 else investScale[i]
        if lack < 0:
            surplus -= lack
        else:
            self.randomAdd(investScale, lack)
        self.randomMinus(investScale, surplus)

    def randomMinus(self, investScale: list, surplus: float):
        while surplus > 0:
            for i in range(len(investScale)):
                temp = random.random()
                if investScale[i] > 0:
                    minus = (temp if surplus-temp >= 0 else surplus)
                    minus = minus if investScale[i] - \
                        minus >= 0 else investScale[i]
                    investScale[i] -= minus
                    surplus -= minus

    def randomAdd(self, investScale: list, lack: float):
        while lack > 0:
            for i in range(len(investScale)):
                temp = random.random()
                add = temp if lack-temp >= 0 else lack
                investScale[i] += add
                lack -= add


class NAPSO:
    def __init__(self, stockNum: int = 10, days: int = 6, maxTimes: int = 200, particlesNums: int = 50):
        self.stockNum = stockNum
        self.stockSet = [stock(days) for i in range(stockNum)]
        self.particles = [particle(stockNum) for i in range(100)]
        self.vstack = self.stockSet[0].earnRateArray
        for i in self.stockSet[1:]:
            self.vstack = np.vstack((self.vstack, i.earnRateArray))
        self.cov = np.cov(self.vstack)
        self.pbest = copy.deepcopy(self.particles)
        self.pbest_fit = [i.calFitness(
            self.stockSet, self.cov, stockNum) for i in self.pbest]
        self.gbest = self.findBestGbest()
        self.gbest_fit = self.gbest.calFitness(
            self.stockSet, self.cov, stockNum)
        self.gbest_fit_history = [self.gbest_fit]
        self.iterTimes = 0
        self.maxTimes = maxTimes
        self.psoDataCopy()

    def psoDataCopy(self):
        self.particles_PSO = copy.deepcopy(self.particles)
        self.pbest_PSO = copy.deepcopy(self.pbest)
        self.pbest_fit_PSO = copy.deepcopy(self.pbest_fit)
        self.gbest_PSO = copy.deepcopy(self.gbest)
        self.gbest_fit_PSO = copy.deepcopy(self.gbest_fit)
        self.gbest_fit_PSO_history = [self.gbest_fit_PSO]
        self.iterTimes_PSO = 0
        self.vList = self.randomVList(self.particles_PSO)
        self.c1 = 2
        self.c2 = 2
        self.wBegin = 0.9
        self.wEnd = 0.4

    def randomVList(self, particles: list) -> list:
        vList = []
        for i in range(len(particles)):
            vList.append(self.randomV(particles[i]))
        return vList

    def randomV(self, par: particle) -> list:
        v = []
        for i in range(len(par.investScale)):
            if random.random() <= 0.5:
                v.append(-random.random())
            else:
                v.append(random.random())
        return v

    def particleLearningPSO(self):
        for i in range(len(self.particles_PSO)):
            self.iterTimes_PSO += 1
            pbest = self.pbest_PSO[i]
            gbest = self.gbest_PSO
            v1 = np.array(pbest.investScale) - \
                np.array(self.particles_PSO[i].investScale)
            v2 = np.array(gbest.investScale) - \
                np.array(self.particles_PSO[i].investScale)
            self.particles_PSO[i].investScale, self.vList[i] = self.newResultPSO(
                i, np.array(self.vList[i]), v1, v2)
            self.particles_PSO[i].amend(self.particles_PSO[i].investScale)
            fitness = self.particles_PSO[i].calFitness(
                self.stockSet, self.cov, self.stockNum)
            if fitness < self.gbest_fit_PSO:
                self.pbest_PSO[i] = self.particles_PSO[i]
                self.pbest_fit_PSO[i] = fitness
                self.gbest_PSO = self.particles_PSO[i]
                self.gbest_fit_PSO = fitness
            elif fitness < self.pbest_fit[i]:
                self.pbest_PSO[i] = self.particles_PSO[i]
                self.pbest_fit_PSO[i] = fitness

    def solutionPSO(self, maxTimes: int = 1000) -> (particle, float, str):
        start = time.time()
        for i in range(maxTimes):
            #print(i)
            self.iterTimes_PSO += 1
            self.particleLearningPSO()
            self.gbest_fit_PSO_history.append(self.gbest_fit_PSO)
        end = time.time()
        return self.gbest_PSO, self.gbest_fit_PSO, str(round(end-start, 2))

    def newResultPSO(self, index: int, v: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> (list, list):
        w = self.wBegin - (self.wBegin-self.wEnd) * \
            (self.iterTimes_PSO/self.maxTimes)
        v = w*v + self.c1*random.random()*(v1-v) + self.c2*random.random()*(v2-v)
        ret = v + np.array(self.particles_PSO[index].investScale)
        return list(ret), list(v)

    def findBestGbest(self) -> particle:
        gbest = self.particles[0]
        for i in self.particles:
            if i.calFitness(self.stockSet, self.cov, self.stockNum) < gbest.calFitness(self.stockSet, self.cov, self.stockNum):
                gbest = i
        return gbest

    def particleLearning(self):
        for i in range(len(self.particles)):
            loc = (np.array(self.pbest[i].investScale) +
                   np.array(self.gbest.investScale))/2
            scale = np.array(self.pbest[i].investScale) - \
                np.array(self.gbest.investScale)
            for j in range(len(self.particles[i].investScale)):
                self.gauResult(self.particles[i], loc[j], scale[j], j)
            self.particles[i].amend(self.particles[i].investScale)
            self.particles[i].neighbourSearch(
                self.iterTimes, self.maxTimes, self.particles[i].investScale)
            temp = self.particles[i].calFitness(
                self.stockSet, self.cov, self.stockNum)
            if temp < self.gbest_fit:
                self.gbest = self.particles[i]
                self.gbest_fit = temp
                self.pbest_fit[i] = temp
                self.pbest[i] = self.particles[i]
            elif temp < self.pbest_fit[i]:
                self.pbest_fit[i] = temp
                self.pbest[i] = self.particles[i]

    def gauResult(self, par: particle, loc: float, scale: float, index: int):
        temp = np.random.normal(loc, abs(scale), 1)
        par.investScale[index] = temp[0]

    def solution(self, maxTimes: int = 200) -> (particle, float, str):
        start = time.time()
        for i in range(maxTimes):
            self.iterTimes+1
            self.particleLearning()
            self.gbest_fit_history.append(self.gbest_fit)
        end = time.time()
        return self.gbest, self.gbest_fit, str(round(end-start, 2))


def makePicture(iterTimes: int, gbest_fit_history: list, colour="g", lineWidth: int = 3, name: str = "NAPSO"):
    plt.figure(1)
    plt.title(name)
    plt.xlabel("Iterators", size=14)
    plt.ylabel("GbestFitness", size=14)
    t = np.array([i for i in range(iterTimes+1)])
    gbest = np.array(gbest_fit_history)
    plt.plot(t, gbest, color=colour, linewidth=lineWidth)
    plt.show()

print("wait patiently……")
test = NAPSO()
gbest, gbest_fit, costTime = test.solution()
print("stockSet:")
for i in range(len(test.stockSet)):
    print(i, ": ", test.stockSet[i].earnRateArray)
PSOmaxTimes = 13  # 经典PSO的迭代次数
gbest_PSO, gbest_fit_PSO, costTimePSO = test.solutionPSO(maxTimes=PSOmaxTimes)
print("NMPSO:")
print(gbest.investScale)
print(gbest_fit)
print("PSO:")
print(gbest_PSO.investScale)
print(gbest_fit_PSO)
makePicture(PSOmaxTimes, test.gbest_fit_PSO_history, name="PSO" +
            " costTime:"+costTimePSO+" gbest:"+str(round(gbest_fit_PSO, 2)))
makePicture(test.maxTimes, test.gbest_fit_history, name="NMPSO" +
            " costTime:"+costTime+" gbest:"+str(round(gbest_fit, 2)))
print("press any key to exit")
input()
# for i in test.pbest:
#     print(i.investScale)
