import matplotlib.pyplot as plt
import torch

uio_data = torch.load("uio-record-100k.pt")
llama_data = torch.load("llama-4k.pt")
longchat_data = torch.load("longchat-100k.pt")
longalpaca_data = torch.load("longalpaca-100k.pt")

names = ['uio', 'llama', 'longchat', 'longalpaca']
color = [
    '#2E865F',  # Deep Sage
    '#FFC499',  # Warm Terracotta
    '#786C3B',  # Earthy Sienna
    '#3498DB'  # Deep Navy
]
datas = [uio_data, llama_data, longchat_data, longalpaca_data]

points = [1024, 2048, 3072, 4096, 5120, 6144, 8192, 16384, 32768, 98304]


xs = {
    "uio": [x[0] for x in uio_data[0]],
    "llama": [x[0] for x in llama_data[0]],
    "longchat": [x[0] for x in longchat_data[0]],
    "longalpaca": [x[0] for x in longalpaca_data[0]]
}
ys = {
    "uio": [x[1] for x in uio_data[0]],
    "llama": [x[1] for x in llama_data[0]],
    "longchat": [x[1] for x in longchat_data[0]],
    "longalpaca": [x[1] for x in longalpaca_data[0]]
}
ys2 = {
    "uio": [x[1] for x in uio_data[1]],
    "llama": [x[1] for x in llama_data[1]],
    "longchat": [x[1] for x in longchat_data[1]],
    "longalpaca": [x[1] for x in longalpaca_data[1]]
}

if 'name' == 'uio':
    ys2['uio'] = [y / 1024 / 1024 / 24576 * 100 - 20.93 for y in ys2['uio']]

plt.figure(figsize=(20,2))
for point_idx, point in enumerate(points):

    time_values = []
    mems_values = []

    for name in names:
        if point in xs[name]:
            idx = xs[name].index(point)
            time_values.append(ys[name][idx])
            mems_values.append(ys2[name][idx])
        else:
            time_values.append(0)
            mems_values.append(0)

    plt.subplot(1, 10, point_idx + 1)
    plt.title(f'{point}')
    plt.ylim(bottom=0, top=50)
    plt.bar(time_values, color=color, width=0.2)

plt.savefig(f"time.jpg", dpi=640)



# plt.figure(figsize=(8,2))

# xs = [data[0] for data in uio_data[0]]
# ys = [data[1] for data in uio_data[0]]
# plt.plot(xs, ys)

# xs = [data[0] for data in llama_data[0]]
# ys = [data[1] for data in llama_data[0]]
# plt.plot(xs, ys)

# xs = [data[0] for data in longchat_data[0]]
# ys = [data[1] for data in longchat_data[0]]
# plt.plot(xs, ys)

# xs = [data[0] for data in longalpaca_data[0]]
# ys = [data[1] for data in longalpaca_data[0]]
# plt.plot(xs, ys)
# plt.xticks(['4096', '16384', '32768', '65536', '99328'], labels=['4K', '16K', '32K', '64K', '100K'])

# plt.savefig("throughput.png", dpi=640)