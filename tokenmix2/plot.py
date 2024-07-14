import matplotlib.pyplot as plt
from numpy import mean, array



def plot(filename, agg, linewidth=0.3, marker=None, keyword=''):
    with open(filename, 'r') as f:
        lines = f.read().split('\n')[:-1]

    lines_ = []
    for line in lines:
        if keyword == '':
            try:
                line = float(line)
                lines_.append(line)
            except:
                pass
        else:
            if line.startswith(keyword) and float(line.replace(keyword, '')) > 0:
                lines_.append(float(line.replace(keyword, '')))
    lines = lines_

    agg_lines = []
    for i in range(agg):
        lines_i = [0] * i + lines + [0] * (agg - i - 1)
        agg_lines.append(lines_i)

    agg_lines = mean(array(agg_lines), axis=0)
    if agg > 1:
        plt.plot(agg_lines[agg:-agg], linewidth=linewidth, marker=marker, markersize=0.5)
    else:
        plt.plot(agg_lines, linewidth=linewidth, marker=marker, markersize=0.5)


from functools import partial
plot_train = partial(plot, agg=10)
plot_eval = partial(plot, agg=1, linewidth=1, marker='p')


# plt.figure(figsize=(21, 7))
# plt.subplot(151)
# plt.title("language modeling loss")
# plot_train("log/arch20-8.log", keyword="my loss: ", agg=100)
# # plot_train("log/arch18-1.log", keyword="my loss: ", agg=5000)
# # plot_train("log/success10.log", keyword="my loss: ", agg=5000)
# plt.subplot(152)
# plt.title("avg. gradient norm")
# plot_train("log/arch20-8.log", keyword="gd_norm_avg: ", agg=10)
# plt.ylim(bottom=0)
# plt.subplot(153)
# plt.title("max gradient norm")
# plot_train("log/arch20-8.log", keyword="gd_norm_max: ", agg=10)
# plt.ylim(bottom=0)
# plt.subplot(154)
# plt.title("avg. gradient norm ratio")
# plot_train("log/arch20-8.log", keyword="gd_norm_ratio_avg: ", agg=10)
# # plot_train("log/arch18-1.log", keyword="gd_norm_ratio_avg: ", agg=10)
# plt.ylim(bottom=0, top=1)
# plt.subplot(155)
# plt.title("max gradient norm ratio")
# plot_train("log/arch20-8.log", keyword="gd_norm_ratio_max: ", agg=10)
# # plot_train("log/arch18-1.log", keyword="gd_norm_ratio_max: ", agg=10)
# plt.ylim(bottom=0, top=1)
# plt.savefig("fig.jpg", dpi=640)

plt.figure(figsize=(7,7))
plot_train("log/arch21-1.log", keyword="my loss: ", agg=100)
plt.savefig("fig.jpg", dpi=640)