import matplotlib.pyplot as plt

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
# plt.rcParams["font.family"] = "Calibri"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 35  # 字体大小
plt.rcParams["figure.figsize"] = (14, 10)  # 图大小
plt.tick_params(top=True, right=True)  # 刻度显示
plt.tick_params(axis='x', direction="in", pad=10, length=10)
plt.tick_params(axis='y', direction="in", pad=10, length=10)

data = {
    "throughput": {10: [22, 48, 75, 100, 122, 137, 138], 50: [22, 48, 75, 100, 122, 140, 139], 100: [22, 48, 75, 100, 122, 142, 141]},
    "latency": {10: [0.32, 0.26, 0.23, 0.22, 0.21, 15, 32], 50: [0.92, 0.95, 0.65, 0.6, 0.58, 11.2, 31.5], 100: [0.94, 1.2, 1.4, 1.35, 1.3, 10, 31.2]}
}

transactiontraffic = [i for i in range(25, 200, 25)]

fig, ax = plt.subplots()
ax.grid("+")
ax2 = ax.twinx()

ax.set_xlabel("Transaction Arrival Rate (TPS)")
ax2.set_ylabel("Throughput (TPS)")
ax.set_ylabel("Latency (Seconds)")

ax.set_xlim(10, 190)
ax2.set_ylim(10, 175)
ax.set_xticks([i for i in range(25, 200, 25)])
ax2.set_yticks([i for i in range(25, 200, 25)])
ax.set_ylim(0, 45)
ax.set_yticks([i for i in range(3, 46, 7)])

ax.bar([i - 3 for i in transactiontraffic], data["latency"][10], width=3, zorder=3, color=Color[32][1], hatch="/", clip_on=False,
       label="Latency (Block Size=10)")
ax.bar(transactiontraffic, data["latency"][50], width=3, zorder=3, color=Color[5][4], hatch="/", clip_on=False,
       label="Latency (Block Size=50)")
ax.bar([i + 3 for i in transactiontraffic], data["latency"][100], width=3, zorder=3, color=Color[5][1], hatch="/", clip_on=False,
       label="Latency (Block Size=100)")

ax2.plot(transactiontraffic, data["throughput"][10], linewidth=3, zorder=3, color="limegreen", marker="x", markersize=15, markeredgewidth=3, markerfacecolor="none", clip_on=False,
         label="Throughput (Block Size=10)")
ax2.plot(transactiontraffic, data["throughput"][50], linewidth=3, zorder=4, color="royalblue", marker="x", markersize=15, markeredgewidth=3, markerfacecolor="none", clip_on=False,
         label="Throughput (Block Size=50)")
ax2.plot(transactiontraffic, data["throughput"][100], linewidth=3, zorder=5, color="red", marker="x", markersize=15, markeredgewidth=3, markerfacecolor="none", clip_on=False,
         label="Throughput (Block Size=100)")

rect = plt.Rectangle((143, 45), 38, 102, alpha=0.8, color="white", ec="black", linewidth=2, linestyle="--", zorder=6)
ax2.add_patch(rect)
ax.text(70, 2.8,
        s="1) Marginal utility for throughput.\n2) Throughput is no increase clearly\nbut rapidly increases latency.",
        bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="black"), fontsize=20)
ax.plot([68.8, 143.3], [7.8, 9.6], color="black", zorder=4)
ax.plot([138, 181.3], [7.8, 9.6], color="black", zorder=4)

rect = plt.Rectangle((120, 115), 10, 15, alpha=0.8, color="white", ec="black", linewidth=2, linestyle="--", zorder=6)
ax2.add_patch(rect)
ax.text(95, 35,
        s="Optimal BPO point.",
        bbox=dict(boxstyle="square", color="BlanchedAlmond", ec="black"), fontsize=20)
ax.plot([94, 120], [34.2, 32.7], color="black", zorder=4)
ax.plot([130, 133.3], [32.7, 34.2], color="black", zorder=4)


ax2.text(145.5, 156, s="142", fontsize=25, color="red")
ax2.text(170.5, 156, s="141", fontsize=25, color="red")
ax2.arrow(150, 155, 0, -9, head_width=2, head_length=2, fc='black', ec='black',zorder=7)
ax2.arrow(175, 155, 0, -10, head_width=2, head_length=2, fc='black', ec='black',zorder=7)

fig.legend(loc=2, fontsize=20, bbox_to_anchor=(0, 1), bbox_transform=ax2.transAxes)
plt.show()
