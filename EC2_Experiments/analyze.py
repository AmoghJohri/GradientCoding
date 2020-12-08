import matplotlib.pyplot as plt 
import statistics

name = "Data/data4.txt"

def remove_max(d):
    for each in d.keys():
        d[each].remove(max(d[each]))

def plot_all(d):
    for each in d.keys():
        plt.plot([i for i in range(len(d[each]))], d[each])
    plt.xlabel("Iterations")
    plt.ylabel("Time Taken (s)")
    plt.legend([each for each in d.keys()], loc ="upper left") 
    plt.title(name + " - Time Taken per Iteration")
    plt.grid()
    plt.show()

f = open(name)
d = {}
for line in f:
    aux = line.split(" ")
    aux = aux[-1]
    aux = aux.split("_")
    if len(aux) > 2:
        aux = aux[:2]
    if len(aux) < 2 or aux[-1][0] == "_" or aux[-1] == "":
        continue
    if aux[0] not in d.keys():
        d[aux[0]] = [float(aux[1])]
    else:
        d[aux[0]] = d[aux[0]] + [float(aux[1])]


plot_all(d)

average_times = []
for each in d.keys():
    average_times.append(sum(d[each])/len(d[each]))
    plt.plot(d[each])
    plt.xlabel("Iterations")
    plt.ylabel("Time Taken (s)")
    plt.title(name + " - Time Taken per Iteration")
    plt.grid()
    plt.show()

print(max(average_times))

mean_average = statistics.mean(average_times)
median_average = statistics.median(average_times)

print("Number of Nodes: ", len(d.keys()))
print("Mean: ", mean_average)
print("Median: ", median_average)

plt.plot(["Node_" + str(i) for i in range(len(d.keys()))], average_times, 'bo')
plt.hlines(mean_average, xmin=-1, xmax = len(d.keys()) + 1, colors='green' )
plt.hlines(median_average, xmin=-1, xmax = len(d.keys()) + 1, colors='red')
plt.xlabel("Compute Instances")
plt.ylabel("Average Time (s)")
plt.legend(('Average Time (s)', 'Mean', 'Median'), loc ="upper right") 
plt.title(name + "Average Time")
plt.grid()
plt.show()