import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

Combination = {}
CombinationX = {}
Time = {

}
Data = {
    "BPR": {},
    "LearningChain": {},
    "MUBPO": {}
}
Performance = {
    "BPR": {},
    "LearningChain": {},
    "MUBPO": {},
}
Regulation = {
    "BPR": {},
    "LearningChain": {},
    "MUBPO": {},
}


def MMBPDScoring(t, l):
    # Reading the dataset
    BPR = pd.read_csv("../Data/MMBPD_Dynamic_BPR.csv").values.tolist()
    LearningChain = pd.read_csv("../Data/MMBPD_Dynamic_LearningChain.csv").values.tolist()
    MUBPO = pd.read_csv("../Data/MMBPD_MU-BPO.csv").values.tolist()

    # BPR.sort(key=lambda x: (x[0], x[4]))
    # LearningChain.sort(key=lambda x: (x[0], x[4]))
    # MUBPO.sort(key=lambda x: (x[0], x[1]))

    # Constructing the data structure
    for index in range(len(BPR)):
        time = BPR[index][0]
        tar = BPR[index][1]
        blocksize = BPR[index][4]
        latency, BPRlatency, LearningChainlatency = BPR[index][5], BPR[index][-2], LearningChain[index][-2]
        throughput, BPRthroughput, LearningChainthroughput = BPR[index][-3], BPR[index][-1], LearningChain[index][-1]
        if time not in Time:
            Time.setdefault(time, tar)
        if tar not in Combination:
            Combination.setdefault(tar, {})
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain"]:
                globals()["Data"][methods].setdefault(tar, {})
                globals()["Data"][methods][tar].setdefault(blocksize,
                                                           [locals()[methods + str("latency")], locals()[methods + str("throughput")]])  # prediction_latency, prediction_throughput
            for methods in ["BPR", "LearningChain", "MUBPO"]:
                globals()["Performance"][methods].setdefault(tar, [9999, 0, 9999, 0, 0, None,
                                                                   None])  # tar:[minlatency,maxlatency,minthroughput,maxthroughput,score,actuallatency,actualthroughput]
                globals()["Regulation"][methods].setdefault(tar, [None, None])  # tar:[regulated tar, regulated block size]
        if blocksize not in Combination[tar]:
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain"]:
                globals()["Data"][str(methods)][tar].setdefault(blocksize, [locals()[methods + str("latency")], locals()[methods + str("throughput")]])
        else:  # Saving the better data for the duplicate
            if throughput > Combination[tar][blocksize][1]:
                Combination[tar][blocksize] = [latency, throughput]
            elif latency < Combination[tar][blocksize][0]:
                Combination[tar][blocksize] = [latency, throughput]

    for index in range(len(MUBPO)):
        tar = round(MUBPO[index][0] / 5) * 5
        blocksize = MUBPO[index][1]
        latency, MUBPOlatency = MUBPO[index][2], MUBPO[index][-2]
        throughput, MUBPOthroughput = MUBPO[index][3], MUBPO[index][-1]
        if tar not in CombinationX:
            CombinationX.setdefault(tar, {})
            CombinationX[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["MUBPO"]:
                globals()["Data"][methods].setdefault(tar, {})
                globals()["Data"][methods][tar].setdefault(blocksize,
                                                           [locals()[methods + str("latency")], locals()[methods + str("throughput")]])  # prediction_latency, prediction_throughput
                globals()["Performance"][methods].setdefault(tar, [9999, 0, 9999, 0, 0, None,
                                                                   None])  # tar:[minlatency,maxlatency,minthroughput,maxthroughput,score,actuallatency,actualthroughput]
                globals()["Regulation"][methods].setdefault(tar, [None, None])  # tar:[regulated tar, regulated block size]
        if blocksize not in CombinationX[tar]:
            CombinationX[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["MUBPO"]:
                globals()["Data"][str(methods)][tar].setdefault(blocksize, [locals()[methods + str("latency")], locals()[methods + str("throughput")]])
        else:  # Saving the better data for the duplicate
            if throughput > CombinationX[tar][blocksize][1]:
                CombinationX[tar][blocksize] = [latency, throughput]
            elif latency < CombinationX[tar][blocksize][0]:
                CombinationX[tar][blocksize] = [latency, throughput]

    # Finding the min and max latency and throughput
    for tar in Combination:
        for blocksize in Combination[tar]:
            for methods in ["BPR", "LearningChain"]:
                pred_latency, pred_throughput = globals()["Data"][methods][tar][blocksize][0], globals()["Data"][methods][tar][blocksize][1]
                if pred_latency < globals()["Performance"][methods][tar][0]:  # lower latency
                    globals()["Performance"][methods][tar][0] = pred_latency
                if pred_latency > globals()["Performance"][methods][tar][1]:  # higher latency
                    globals()["Performance"][methods][tar][1] = pred_latency
                if pred_throughput < globals()["Performance"][methods][tar][2]:  # lower throughput
                    globals()["Performance"][methods][tar][2] = pred_throughput
                if pred_throughput > globals()["Performance"][methods][tar][3]:  # higher throughput
                    globals()["Performance"][methods][tar][3] = pred_throughput
        for i in range(10, int(tar) + 5, 5):
            for blocksize in CombinationX[i]:
                pred_latency, pred_throughput = Data["MUBPO"][i][blocksize][0], Data["MUBPO"][i][blocksize][1]
                if pred_latency < Performance["MUBPO"][tar][0]:  # lower latency
                    Performance["MUBPO"][tar][0] = pred_latency
                if pred_latency > Performance["MUBPO"][tar][1]:  # higher latency
                    Performance["MUBPO"][tar][1] = pred_latency
                if pred_throughput < Performance["MUBPO"][tar][2]:  # lower throughput
                    Performance["MUBPO"][tar][2] = pred_throughput
                if pred_throughput > Performance["MUBPO"][tar][3]:  # higher throughput
                    Performance["MUBPO"][tar][3] = pred_throughput

    # Scoring and getting the optimal BCP and traffic corresponding to actual performance, thus evaluate the effectiveness of BPO methods
    for tar in Combination:
        for methods in ["BPR", "LearningChain"]:
            for blocksize in Combination[tar]:  # Only can regulate the BCP in BPR and LearningChain
                pred_latency, pred_throughput = globals()["Data"][methods][tar][blocksize][0], globals()["Data"][methods][tar][blocksize][1]
                min_latency, max_latency, min_throughput, max_throughput = globals()["Performance"][methods][tar][0], globals()["Performance"][methods][tar][1], \
                                                                           globals()["Performance"][methods][tar][2], globals()["Performance"][methods][tar][3]
                if max_throughput != min_throughput and max_latency != min_latency:
                    score = t * (pred_throughput - min_throughput) / (
                            max_throughput - min_throughput) + l * (
                                    pred_latency - max_latency) / (
                                    min_latency - max_latency)
                if score > globals()["Performance"][methods][tar][4]:
                    globals()["Performance"][methods][tar][4], globals()["Performance"][methods][tar][5], globals()["Performance"][methods][tar][6] = score, \
                                                                                                                                                      Combination[tar][blocksize][
                                                                                                                                                          0], \
                                                                                                                                                      Combination[tar][blocksize][1]
                    Regulation[methods][tar] = [tar, blocksize]
        for i in range(10, int(tar) + 5, 5):  # Not only can regulate the BCP, but also can regulate the transaction traffic in MU-BPO
            for blocksize in CombinationX[i]:
                pred_latency, pred_throughput = Data["MUBPO"][i][blocksize][0], Data["MUBPO"][i][blocksize][1]
                min_latency, max_latency, min_throughput, max_throughput = Performance["MUBPO"][tar][0], Performance["MUBPO"][tar][1], Performance["MUBPO"][tar][2], \
                                                                           Performance["MUBPO"][tar][3]
                score = t * (pred_throughput - min_throughput) / (max_throughput - min_throughput) + l * (
                        pred_latency - max_latency) / (
                                min_latency - max_latency)
                # score = (Data["MUBPO"][i][blocksize][1] - Performance["MUBPO"][tar][2]) / (Performance["MUBPO"][tar][3] - Performance["MUBPO"][tar][2]) * (
                #         Data["MUBPO"][i][blocksize][0] - Performance["MUBPO"][tar][1]) / (
                #                 Performance["MUBPO"][tar][0] - Performance["MUBPO"][tar][1])
                if score > Performance["MUBPO"][tar][4]:
                    Performance["MUBPO"][tar][4], Performance["MUBPO"][tar][5], Performance["MUBPO"][tar][6] = score, \
                                                                                                               CombinationX[i][blocksize][0], \
                                                                                                               CombinationX[i][blocksize][1]
                    Regulation["MUBPO"][tar] = [i, blocksize]



if __name__ == '__main__':
    MMBPDScoring(0.8,0.2)
    for methods in ["BPR", "LearningChain", "MUBPO"]:
        print("TotalThroughput_" + methods + ":", sum([globals()["Performance"][methods][Time[i]][-1] for i in Time]))
        print("AvgThroughput_" + methods + ":", sum([globals()["Performance"][methods][Time[i]][-1] for i in Time]) / len(Time))
        print("TotalLatency_" + methods + ":", sum([globals()["Performance"][methods][Time[i]][-2] for i in Time]))
        print("AvgLatency_" + methods + ":", sum([globals()["Performance"][methods][Time[i]][-2] for i in Time]) / len(Time))
    print("Throughput_Improv. VS BPR", (sum([globals()["Performance"]["MUBPO"][Time[i]][-1] for i in Time]) - sum([globals()["Performance"]["BPR"][Time[i]][-1] for i in Time])) / sum(
        [globals()["Performance"]["BPR"][Time[i]][-1] for i in Time]) * 100, "%")
    print("Throughput_Improv. VS LearningChain",
          (sum([globals()["Performance"]["MUBPO"][Time[i]][-1] for i in Time]) - sum([globals()["Performance"]["LearningChain"][Time[i]][-1] for i in Time])) / sum(
              [globals()["Performance"]["LearningChain"][Time[i]][-1] for i in Time]) * 100, "%")
    print("Latency_Improv. VS BPR", (sum([globals()["Performance"]["BPR"][Time[i]][-2] for i in Time]) - sum([globals()["Performance"]["MUBPO"][Time[i]][-2] for i in Time])) / sum(
        [globals()["Performance"]["BPR"][Time[i]][-2] for i in Time]) * 100, "%")
    print("Latency_Improv. VS LearningChain",
          (sum([globals()["Performance"]["LearningChain"][Time[i]][-2] for i in Time]) - sum([globals()["Performance"]["MUBPO"][Time[i]][-2] for i in Time])) / sum(
              [globals()["Performance"]["LearningChain"][Time[i]][-2] for i in Time]) * 100, "%")
