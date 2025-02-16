import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

Combination = {}
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


def BPDScoring(t, l):
    # Reading the dataset
    BPR = pd.read_csv("../Data/BPD_BPR.csv").values.tolist()
    LearningChain = pd.read_csv("../Data/BPD_LearningChain.csv").values.tolist()
    MUBPO = pd.read_csv("../Data/BPD_MU-BPO.csv").values.tolist()

    BPR.sort(key=lambda x: (x[0], x[1]))
    LearningChain.sort(key=lambda x: (x[0], x[1]))
    MUBPO.sort(key=lambda x: (x[0], x[1]))

    # Constructing the data structure
    for index in range(len(BPR)):
        tar = round(BPR[index][0] / 5) * 5
        blocksize = BPR[index][1]
        latency, BPRlatency, LearningChainlatency, MUBPOlatency = BPR[index][2], BPR[index][-2], LearningChain[index][-2], MUBPO[index][-2]
        throughput, BPRthroughput, LearningChainthroughput, MUBPOthroughput = BPR[index][3], BPR[index][-1], LearningChain[index][-1], MUBPO[index][-1]
        if tar not in Combination:
            Combination.setdefault(tar, {})
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain", "MUBPO"]:
                globals()["Data"][methods].setdefault(tar, {})
                globals()["Data"][methods][tar].setdefault(blocksize,
                                                           [locals()[methods + str("latency")], locals()[methods + str("throughput")]])  # prediction_latency, prediction_throughput
                globals()["Performance"][methods].setdefault(tar, [9999, 0, 9999, 0, 0, None,
                                                                   None])  # tar:[minlatency,maxlatency,minthroughput,maxthroughput,score,actuallatency,actualthroughput]
        if blocksize not in Combination[tar]:
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain", "MUBPO"]:
                globals()["Data"][str(methods)][tar].setdefault(blocksize, [locals()[methods + str("latency")], locals()[methods + str("throughput")]])
        else:  # Saving the better data for the duplicate
            if throughput > Combination[tar][blocksize][1]:
                Combination[tar][blocksize] = [latency, throughput]
            elif latency < Combination[tar][blocksize][0]:
                Combination[tar][blocksize] = [latency, throughput]

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
        for i in range(10, tar + 1, 5):
            for blocksize in Combination[i]:
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
                score = t * (pred_throughput - min_throughput) / (
                        max_throughput - min_throughput) + l * (
                                pred_latency - max_latency) / (
                                min_latency - max_latency)
                if score > globals()["Performance"][methods][tar][4]:
                    globals()["Performance"][methods][tar][4], globals()["Performance"][methods][tar][5], globals()["Performance"][methods][tar][6] = score, \
                                                                                                                                                      Combination[tar][blocksize][
                                                                                                                                                          0], \
                                                                                                                                                      Combination[tar][blocksize][1]
        for i in range(10, tar + 1, 5):  # Not only can regulate the BCP, but also can regulate the transaction traffic in MU-BPO
            for blocksize in Combination[i]:
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
                                                                                                               Combination[i][blocksize][0], \
                                                                                                               Combination[i][blocksize][1]


def HFBTPScoring(t, l):
    # Reading the dataset
    BPR = pd.read_csv("../Data/HFBTP_BPR.csv").values.tolist()
    LearningChain = pd.read_csv("../Data/HFBTP_LearningChain.csv").values.tolist()
    MUBPO = pd.read_csv("../Data/HFBTP_MU-BPO.csv").values.tolist()

    BPR.sort(key=lambda x: (x[0], x[1]))
    LearningChain.sort(key=lambda x: (x[0], x[1]))
    MUBPO.sort(key=lambda x: (x[0], x[1]))

    # Constructing the data structure
    for index in range(len(BPR)):
        tar = round(BPR[index][0] / 5) * 5
        blocksize = BPR[index][1]
        orderer = BPR[index][2]
        if orderer == 3:
            latency, BPRlatency, LearningChainlatency, MUBPOlatency = BPR[index][3], BPR[index][-2], LearningChain[index][-2], MUBPO[index][-2]
            throughput, BPRthroughput, LearningChainthroughput, MUBPOthroughput = BPR[index][4], BPR[index][-1], LearningChain[index][-1], MUBPO[index][-1]
            if tar not in Combination:
                Combination.setdefault(tar, {})
                Combination[tar].setdefault(blocksize, [latency, throughput])
                for methods in ["BPR", "LearningChain", "MUBPO"]:
                    globals()["Data"][methods].setdefault(tar, {})
                    globals()["Data"][methods][tar].setdefault(blocksize,
                                                               [locals()[methods + str("latency")],
                                                                locals()[methods + str("throughput")]])  # prediction_latency, prediction_throughput
                    globals()["Performance"][methods].setdefault(tar, [9999, 0, 9999, 0, 0, None,
                                                                       None])  # tar:[minlatency,maxlatency,minthroughput,maxthroughput,score,actuallatency,actualthroughput]
            if blocksize not in Combination[tar]:
                Combination[tar].setdefault(blocksize, [latency, throughput])
                for methods in ["BPR", "LearningChain", "MUBPO"]:
                    globals()["Data"][str(methods)][tar].setdefault(blocksize, [locals()[methods + str("latency")], locals()[methods + str("throughput")]])
            else:  # Saving the better data for the duplicate
                if throughput > Combination[tar][blocksize][1]:
                    Combination[tar][blocksize] = [latency, throughput]
                elif latency < Combination[tar][blocksize][0]:
                    Combination[tar][blocksize] = [latency, throughput]

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
        for i in range(10, tar + 1, 5):
            for blocksize in Combination[i]:
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
                score = t * (pred_throughput - min_throughput) / (
                        max_throughput - min_throughput) + l * (
                                pred_latency - max_latency) / (
                                min_latency - max_latency)
                if score > globals()["Performance"][methods][tar][4]:
                    globals()["Performance"][methods][tar][4], globals()["Performance"][methods][tar][5], globals()["Performance"][methods][tar][6] = score, \
                                                                                                                                                      Combination[tar][blocksize][
                                                                                                                                                          0], \
                                                                                                                                                      Combination[tar][blocksize][1]
        for i in range(10, tar + 1, 5):  # Not only can regulate the BCP, but also can regulate the transaction traffic in MU-BPO
            for blocksize in Combination[i]:
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
                                                                                                               Combination[i][blocksize][0], \
                                                                                                               Combination[i][blocksize][1]


def MMBPDScoring(t, l):
    # Reading the dataset
    BPR = pd.read_csv("../Data/MMBPD_BPR.csv").values.tolist()
    LearningChain = pd.read_csv("../Data/MMBPD_LearningChain.csv").values.tolist()
    MUBPO = pd.read_csv("../Data/MMBPD_MU-BPO.csv").values.tolist()

    BPR.sort(key=lambda x: (x[0], x[1]))
    LearningChain.sort(key=lambda x: (x[0], x[1]))
    MUBPO.sort(key=lambda x: (x[0], x[1]))

    # Constructing the data structure
    for index in range(len(BPR)):
        tar = round(BPR[index][0] / 5) * 5
        blocksize = BPR[index][1]
        latency, BPRlatency, LearningChainlatency, MUBPOlatency = BPR[index][2], BPR[index][-2], LearningChain[index][-2], MUBPO[index][-2]
        throughput, BPRthroughput, LearningChainthroughput, MUBPOthroughput = BPR[index][3], BPR[index][-1], LearningChain[index][-1], MUBPO[index][-1]
        if tar not in Combination and tar <= 175:
            Combination.setdefault(tar, {})
            Combination[tar].setdefault(blocksize, [latency, throughput])
            for methods in ["BPR", "LearningChain", "MUBPO"]:
                globals()["Data"][methods].setdefault(tar, {})
                globals()["Data"][methods][tar].setdefault(blocksize,
                                                           [locals()[methods + str("latency")], locals()[methods + str("throughput")]])  # prediction_latency, prediction_throughput
                globals()["Performance"][methods].setdefault(tar, [9999, 0, 9999, 0, 0, None,
                                                                   None])  # tar:[minlatency,maxlatency,minthroughput,maxthroughput,score,actuallatency,actualthroughput]
                globals()["Regulation"][methods].setdefault(tar, [None, None])  # tar:[regulated tar, regulated block size]
        if tar <= 175:
            if blocksize not in Combination[tar]:
                Combination[tar].setdefault(blocksize, [latency, throughput])
                for methods in ["BPR", "LearningChain", "MUBPO"]:
                    globals()["Data"][str(methods)][tar].setdefault(blocksize, [locals()[methods + str("latency")], locals()[methods + str("throughput")]])
            else:  # Saving the better data for the duplicate
                if throughput > Combination[tar][blocksize][1]:
                    Combination[tar][blocksize] = [latency, throughput]
                elif latency < Combination[tar][blocksize][0]:
                    Combination[tar][blocksize] = [latency, throughput]

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
        for i in range(10, tar + 1, 5):
            for blocksize in Combination[i]:
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
        for i in range(10, tar + 1, 5):  # Not only can regulate the BCP, but also can regulate the transaction traffic in MU-BPO
            for blocksize in Combination[i]:
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
                                                                                                               Combination[i][blocksize][0], \
                                                                                                               Combination[i][blocksize][1]
                    Regulation["MUBPO"][tar] = [i, blocksize]
    print(Performance["MUBPO"])


def Drawing(dataset, t, l, step):
    if dataset == "BPD":
        BPDScoring(t, l)
    elif dataset == "HFBTP":
        HFBTPScoring(t, l)
    elif dataset == "MMBPD":
        MMBPDScoring(t, l)

    Color = {
        2: [(7 / 255, 7 / 255, 7 / 255), (255 / 255, 59 / 255, 59 / 255)],
        31: [(89 / 255, 169 / 255, 90 / 255), (247 / 255, 144 / 255, 61 / 255), (77 / 255, 133 / 255, 189 / 255)],
        32: [(56 / 255, 89 / 255, 137 / 255), (210 / 255, 32 / 255, 39 / 255), (127 / 255, 165 / 255, 183 / 255)],
        4: [(43 / 255, 85 / 255, 125 / 255), (69 / 255, 189 / 255, 155 / 255), (240 / 255, 81 / 255, 121 / 255),
            (253 / 255, 207 / 255, 110 / 255)],
        5: [(79 / 255, 89 / 255, 109 / 255), (95 / 255, 198 / 255, 201 / 255), (1 / 255, 86 / 255, 153 / 255),
            (250 / 255, 192 / 255, 15 / 255), (243 / 255, 118 / 255, 74 / 255)]
    }

    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40  # 字体大小
    plt.rcParams["figure.figsize"] = (15, 10.5)  # 图大小
    plt.tick_params(top=True, right=True)  # 刻度显示
    plt.tick_params(axis='x', direction="in", pad=10, length=10)
    plt.tick_params(axis='y', direction="in", pad=10, length=10)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xlabel("Transaction Arrival Rate (TPS)")
    ax.set_ylabel("Latency (Seconds)")
    ax2.set_ylabel("Throughput (TPS)")

    plt.xlim(0, 210)
    plt.xticks([0, 40, 80, 120, 160, 200])
    ax.set_ylim(0, 3)
    ax2.set_ylim(0, 200)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_yticks([0, 1, 2, 3])
    ax2.set_yticks([0, 40, 80, 120, 160, 200])

    ax.grid("+")

    tar = list(globals()["Combination"].keys())

    for methods in ["BPR", "LearningChain", "MUBPO"]:
        globals()[methods + "latency"] = [globals()["Performance"][methods][tar][-2] for tar in list(globals()["Performance"][methods].keys())]
        globals()[methods + "throughput"] = [globals()["Performance"][methods][tar][-1] for tar in list(globals()["Performance"][methods].keys())]

    ax.bar([tar[i] - 2 for i in range(0, len(tar), step)], [globals()["BPRlatency"][i] for i in range(0, len(tar), step)],
           label="Latency (BPR)", width=2, zorder=3, edgecolor="black", clip_on=False, color=(70 / 255, 158 / 255, 180 / 255),  alpha=0.6)
    ax.bar([tar[i] for i in range(0, len(tar), step)], [globals()["LearningChainlatency"][i] for i in range(0, len(tar), step)], label="Latency (LearningChain)",
           width=2, zorder=2, edgecolor="black",
           clip_on=False, color=(135 / 255, 207 / 255, 164 / 255),  alpha=0.6)
    ax.bar([tar[i] + 2 for i in range(0, len(tar), step)], [globals()["MUBPOlatency"][i] for i in range(0, len(tar), step)], label="Latency (MU-BPO)",
           width=2, zorder=2, edgecolor="black",
           clip_on=False, color=(245 / 255, 117 / 255, 71 / 255),  alpha=0.6)

    ax2.plot([tar[i] for i in range(0, len(tar), step)], [globals()["BPRthroughput"][i] for i in range(0, len(tar), step)],
             label="Throughput (BPR)", linewidth=3, zorder=3,
             color=(70 / 255, 158 / 255, 180 / 255), marker="o",
             markersize=15, markeredgewidth=3,
             markerfacecolor="none")
    ax2.plot([tar[i] for i in range(0, len(tar), step)], [globals()["LearningChainthroughput"][i] for i in range(0, len(tar), step)],
             label="Throughput (LearningChain)", linewidth=3, zorder=4,
             color=(135 / 255, 207 / 255, 164 / 255), marker="h",
             markersize=15, markeredgewidth=3,
             markerfacecolor="none")
    ax2.plot([tar[i] for i in range(0, len(tar), step)], [globals()["MUBPOthroughput"][i] for i in range(0, len(tar), step)],
             label="Throughput (MU-BPO)", linewidth=3, zorder=4,
             color=(245 / 255, 117 / 255, 71 / 255), marker="s",
             markersize=15, markeredgewidth=3,
             markerfacecolor="none")

    fig.legend(loc=2, fontsize=25, bbox_to_anchor=(0, 1), bbox_transform=ax2.transAxes)
    plt.show()


def DrawingSingle(dataset, t, l, step):
    if dataset == "BPD":
        BPDScoring(t, l)
    elif dataset == "HFBTP":
        HFBTPScoring(t, l)
    elif dataset == "MMBPD":
        MMBPDScoring(t, l)
    Color = {
        2: [(7 / 255, 7 / 255, 7 / 255), (255 / 255, 59 / 255, 59 / 255)],
        31: [(89 / 255, 169 / 255, 90 / 255), (247 / 255, 144 / 255, 61 / 255), (77 / 255, 133 / 255, 189 / 255)],
        32: [(56 / 255, 89 / 255, 137 / 255), (210 / 255, 32 / 255, 39 / 255), (127 / 255, 165 / 255, 183 / 255)],
        4: [(43 / 255, 85 / 255, 125 / 255), (69 / 255, 189 / 255, 155 / 255), (240 / 255, 81 / 255, 121 / 255),
            (253 / 255, 207 / 255, 110 / 255)],
        5: [(79 / 255, 89 / 255, 109 / 255), (95 / 255, 198 / 255, 201 / 255), (1 / 255, 86 / 255, 153 / 255),
            (250 / 255, 192 / 255, 15 / 255), (243 / 255, 118 / 255, 74 / 255)]
    }

    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40  # 字体大小
    plt.rcParams["figure.figsize"] = (13, 10.5)  # 图大小
    plt.tick_params(top=True, right=True)  # 刻度显示
    plt.tick_params(axis='x', direction="in", pad=10, length=10)
    plt.tick_params(axis='y', direction="in", pad=10, length=10)

    plt.grid("+")

    plt.xlabel("Transaction Arrival Rate (TPS)")
    plt.xlim(10, 200)
    plt.xticks([10, 40, 80, 120, 160, 200])

    tar = list(globals()["Combination"].keys())

    for methods in ["BPR", "LearningChain", "MUBPO"]:
        globals()[methods + "latency"] = [globals()["Performance"][methods][tar][-2] for tar in list(globals()["Performance"][methods].keys())]
        globals()[methods + "throughput"] = [globals()["Performance"][methods][tar][-1] for tar in list(globals()["Performance"][methods].keys())]

    if t == 1:
        plt.ylabel("Throughput (TPS)")
        plt.ylim(0, 200)
        plt.yticks([0, 40, 80, 120, 160, 200])
        plt.plot([tar[i] for i in range(0, len(tar), step)], [globals()["BPRthroughput"][i] for i in range(0, len(tar), step)],
                 label="BPR", linewidth=3, zorder=3,
                 color=(70 / 255, 158 / 255, 180 / 255), marker="o",
                 markersize=12, markeredgewidth=3,
                 markerfacecolor="none", clip_on=False)
        plt.plot([tar[i] for i in range(0, len(tar), step)], [globals()["LearningChainthroughput"][i] for i in range(0, len(tar), step)],
                 label="LearningChain", linewidth=3, zorder=4,
                 color=(135 / 255, 207 / 255, 164 / 255), marker="h",
                 markersize=12, markeredgewidth=3,
                 markerfacecolor="none", clip_on=False)
        plt.plot([tar[i] for i in range(0, len(tar), step)], [globals()["MUBPOthroughput"][i] for i in range(0, len(tar), step)],
                 label="MU-BPO", linewidth=3, zorder=4,
                 color=(245 / 255, 117 / 255, 71 / 255), marker="s",
                 markersize=12, markeredgewidth=3,
                 markerfacecolor="none", clip_on=False)
    else:
        plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.ylabel("Latency (Seconds)")
        plt.ylim(0, 3)
        plt.yticks([0, 1, 2, 3])
        plt.plot([tar[i] for i in range(0, len(tar), step)], [globals()["BPRlatency"][i] for i in range(0, len(tar), step)],
                 label="BPR", linewidth=3, zorder=3,
                 color=(70 / 255, 158 / 255, 180 / 255), marker="o",
                 markersize=12, markeredgewidth=3,
                 markerfacecolor="none", clip_on=False)
        plt.plot([tar[i] for i in range(0, len(tar), step)], [globals()["LearningChainlatency"][i] for i in range(0, len(tar), step)],
                 label="LearningChain", linewidth=3, zorder=4,
                 color=(135 / 255, 207 / 255, 164 / 255), marker="h",
                 markersize=12, markeredgewidth=3,
                 markerfacecolor="none", clip_on=False)
        plt.plot([tar[i] for i in range(0, len(tar), step)], [globals()["MUBPOlatency"][i] for i in range(0, len(tar), step)],
                 label="MU-BPO", linewidth=3, zorder=4,
                 color=(245 / 255, 117 / 255, 71 / 255), marker="s",
                 markersize=12, markeredgewidth=3,
                 markerfacecolor="none", clip_on=False)

    plt.legend(loc="upper left", fontsize=30)
    plt.show()


def DrawingRegulation(dataset, t, l, step):
    if dataset == "BPD":
        BPDScoring(t, l)
    elif dataset == "HFBTP":
        HFBTPScoring(t, l)
    elif dataset == "MMBPD":
        MMBPDScoring(t, l)

    Color = {
        2: [(7 / 255, 7 / 255, 7 / 255), (255 / 255, 59 / 255, 59 / 255)],
        31: [(89 / 255, 169 / 255, 90 / 255), (247 / 255, 144 / 255, 61 / 255), (77 / 255, 133 / 255, 189 / 255)],
        32: [(56 / 255, 89 / 255, 137 / 255), (210 / 255, 32 / 255, 39 / 255), (127 / 255, 165 / 255, 183 / 255)],
        4: [(43 / 255, 85 / 255, 125 / 255), (69 / 255, 189 / 255, 155 / 255), (240 / 255, 81 / 255, 121 / 255),
            (253 / 255, 207 / 255, 110 / 255)],
        5: [(79 / 255, 89 / 255, 109 / 255), (95 / 255, 198 / 255, 201 / 255), (1 / 255, 86 / 255, 153 / 255),
            (250 / 255, 192 / 255, 15 / 255), (243 / 255, 118 / 255, 74 / 255)]
    }

    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 40  # 字体大小
    plt.rcParams["figure.figsize"] = (15, 10.5)  # 图大小
    plt.tick_params(top=True, right=True)  # 刻度显示
    plt.tick_params(axis='x', direction="in", pad=10, length=10)
    plt.tick_params(axis='y', direction="in", pad=10, length=10)

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xlabel("Transaction Arrival Rate (TPS)")
    ax2.set_ylabel("Regulated TT (TPS)")
    ax.set_ylabel("Regulated BCP (Block Size)")

    plt.xlim(0, 200)
    plt.xticks([10, 40, 80, 120, 160, 200])
    ax.set_ylim(0, 400)
    ax2.set_ylim(0, 200)
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    ax.set_yticks([0, 100, 200, 300, 400])
    ax2.set_yticks([10, 40, 80, 120, 160, 200])

    ax.grid("+")

    tar = list(globals()["Combination"].keys())

    for methods in ["BPR", "LearningChain", "MUBPO"]:
        globals()[methods + "tt"] = [globals()["Regulation"][methods][tar][0] for tar in list(globals()["Performance"][methods].keys())]
        globals()[methods + "blocksize"] = [globals()["Regulation"][methods][tar][1] for tar in list(globals()["Performance"][methods].keys())]

    # ax2.bar([tar[i] - 2 for i in range(0, len(tar), step)], [globals()["BPRlatency"][i] for i in range(0, len(tar), step)],
    #        label="Latency (BPR)", width=2, zorder=3, edgecolor="black", clip_on=False, color=(70 / 255, 158 / 255, 180 / 255), hatch="x", alpha=0.8)
    # ax2.bar([tar[i] for i in range(0, len(tar), step)], [globals()["LearningChainlatency"][i] for i in range(0, len(tar), step)], label="Latency (LearningChain)",
    #        width=2, zorder=2, edgecolor="black",
    #        clip_on=False, color=(135 / 255, 207 / 255, 164 / 255), hatch=".", alpha=0.8)
    ax.bar([tar[i] for i in range(0, len(tar), step)], [globals()["MUBPOblocksize"][i] for i in range(0, len(tar), step)], label="Regulated BCP",
           width=2, zorder=2, edgecolor="black",
           clip_on=False, color="orange", alpha=0.8)

    # ax.plot([tar[i] for i in range(0, len(tar), step)], [globals()["BPRthroughput"][i] for i in range(0, len(tar), step)],
    #          label="Throughput (BPR)", linewidth=3, zorder=3,
    #          color=(70 / 255, 158 / 255, 180 / 255), marker="o",
    #          markersize=15, markeredgewidth=3,
    #          markerfacecolor="none")
    # ax.plot([tar[i] for i in range(0, len(tar), step)], [globals()["LearningChainthroughput"][i] for i in range(0, len(tar), step)],
    #          label="Throughput (LearningChain)", linewidth=3, zorder=4,
    #          color=(135 / 255, 207 / 255, 164 / 255), marker="h",
    #          markersize=15, markeredgewidth=3,
    #          markerfacecolor="none")
    ax2.plot([tar[i] for i in range(0, len(tar), step)], [globals()["MUBPOtt"][i] for i in range(0, len(tar), step)],
             label="Regulated TT", linewidth=3, zorder=4,
             color="red", marker="h",
             markersize=18, markeredgewidth=2,
             markerfacecolor="none")

    fig.legend(loc=2, fontsize=25, bbox_to_anchor=(0, 1), bbox_transform=ax2.transAxes)
    plt.show()


if __name__ == '__main__':
    # MMBPDScoring(0.8, 0.2)
    # tar = [i for i in range(10, 180, 5)]
    # for methods in ["BPR", "LearningChain", "MUBPO"]:
    #     print("TotalThroughput_" + methods + ":", sum([globals()["Performance"][methods][i][-1] for i in tar]))
    #     print("AvgThroughput_" + methods + ":", sum([globals()["Performance"][methods][i][-1] for i in tar]) / len(tar))
    #     print("TotalLatency_" + methods + ":", sum([globals()["Performance"][methods][i][-2] for i in tar]))
    #     print("AvgLatency_" + methods + ":", sum([globals()["Performance"][methods][i][-2] for i in tar]) / len(tar))
    # # for blocksize in [400]:
    # #     print("TotalThroughput_" + str(blocksize) + ":", sum([Combination[i][blocksize][1] for i in tar]))
    # #     print("AvgThroughput_" + str(blocksize) + ":", sum([Combination[i][blocksize][1] for i in tar]) / len(tar))
    # #     print("TotalLatency_" + str(blocksize) + ":", sum([Combination[i][blocksize][0] for i in tar]))
    # #     print("AvgLatency_" + str(blocksize) + ":", sum([Combination[i][blocksize][0] for i in tar]) / len(tar))
    # print("Throughput_Improv. VS BPR", (sum([globals()["Performance"]["MUBPO"][i][-1] for i in tar]) - sum([globals()["Performance"]["BPR"][i][-1] for i in tar])) / sum(
    #     [globals()["Performance"]["BPR"][i][-1] for i in tar]) * 100, "%")
    # print("Throughput_Improv. VS LearningChain",
    #       (sum([globals()["Performance"]["MUBPO"][i][-1] for i in tar]) - sum([globals()["Performance"]["LearningChain"][i][-1] for i in tar])) / sum(
    #           [globals()["Performance"]["LearningChain"][i][-1] for i in tar]) * 100, "%")
    # print("Latency_Improv. VS BPR", (sum([globals()["Performance"]["BPR"][i][-2] for i in tar]) - sum([globals()["Performance"]["MUBPO"][i][-2] for i in tar])) / sum(
    #     [globals()["Performance"]["BPR"][i][-2] for i in tar]) * 100, "%")
    # print("Latency_Improv. VS LearningChain",
    #       (sum([globals()["Performance"]["LearningChain"][i][-2] for i in tar]) - sum([globals()["Performance"]["MUBPO"][i][-2] for i in tar])) / sum(
    #           [globals()["Performance"]["LearningChain"][i][-2] for i in tar]) * 100, "%")
    # Drawing("MMBPD", 0.8, 0.2, 2)  # dataset, weight_throughput, weight_latency, plot_step, the performance corresponding to the default block size
    # DrawingSingle("BPD", 1, 0, 1)
    DrawingRegulation("MMBPD", 0, 1, 1)
