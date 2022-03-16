import json
import matplotlib.pyplot as plt


def autolabel(ax, rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%0.4f' % float(height),
        ha='center', va='bottom')

def draw_bar(x, t, y, subplot, color, x_lab, y_lab, width=0.2):
    plt.subplot(subplot)
    plt.xticks(x, t)
    ax1 = plt.gca()
    ax1.set_xlabel(x_lab)
    ax1.set_ylabel(y_lab, color=color)
    rects1 = ax1.bar(x, y, color=color, width=width)
    ax1.tick_params(axis='y', labelcolor=color)
    autolabel(ax1, rects1)

def fix_len(name, length):
    if len(name)<length:
        name+=(" "*(length-len(name)))
    return name
def format_print(name, values):
    [a, b] =values
    a=str(a)
    b=str(b)
    name = fix_len(name, 16)
    a = fix_len(a, 24)
    b = fix_len(b, 24)
        
    print("{}{}{}".format(name, a, b))
    
def load_res(json_file):
    with open(json_file) as f:
        data = json.load(f)
        return data

res_32 = load_res('32.json')
res_8 = load_res('8.json')
   
accuracys = [res_32['accuracy'], res_8['accuracy']]
throughputs = [res_32['throughput'], res_8['throughput']]             
latencys = [res_32['latency'], res_8['latency']]

format_print('Model', ['FP32', 'INT8'])
format_print('throughput(fps)', throughputs)
format_print('latency(ms)', latencys)
format_print('accuracy(%)', accuracys)

accuracys_perc = [accu*100 for accu in accuracys]

t = ['FP32', 'INT8']
x = [0, 1]
plt.figure(figsize=(16,6))
draw_bar(x, t, throughputs, 131, 'tab:green', 'Throughput(fps)', '', width=0.2)
draw_bar(x, t,  latencys, 132, 'tab:blue', 'Latency(ms)', '', width=0.2)
draw_bar(x, t,  accuracys_perc, 133, '#28a99d', 'Accuracys(%)', '', width=0.2)
plt.savefig("fp32_int8_aboslute.png")
print("\nSave to fp32_int8_aboslute.png\n")

throughputs_times = [1, throughputs[1]/throughputs[0]]
latencys_times = [1, latencys[1]/latencys[0]]
accuracys_times = [1, accuracys_perc[1]/accuracys_perc[0]]

format_print('Model', ['FP32', 'INT8'])
format_print('throughput_times', throughputs_times)
format_print('latency_times', latencys_times)
format_print('accuracy_times', accuracys_times)

plt.figure(figsize=(16,6))
draw_bar(x, t, throughputs_times, 131, 'tab:green', 'Throughput Normalized (big is better)', '', width=0.2)
draw_bar(x, t, latencys_times, 132, 'tab:blue', 'Latency Normalized (small is better)', '', width=0.2)
draw_bar(x, t, accuracys_times, 133, '#28a99d', 'Accuracys Normalized (small is better)', '', width=0.2)

plt.savefig("fp32_int8_times.png")
print("\nSave to fp32_int8_times.png")
