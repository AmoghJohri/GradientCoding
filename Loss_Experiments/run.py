import shutil
import matplotlib.pyplot as plt
import distributed_gradient_descent
import distributed_gradient_descent_ignore
import distributed_gradient_descent_uncoded
import distributed_gradient_descent_frc
import distributed_gradient_descent_frc_shuffling

dataset   = "wine"
N         = 10
time_stop = 0.02
s_ratio   = 0.5
alpha     = 0.0000015

t1, l1 = distributed_gradient_descent.run(dataset=dataset, N=N, time_stop=time_stop, alpha=alpha)
t2, l2 = distributed_gradient_descent_ignore.run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha)
t3, l3 = distributed_gradient_descent_uncoded.run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha)
t4, l4 = distributed_gradient_descent_frc.run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha)
t5, l5 = distributed_gradient_descent_frc_shuffling.run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha)

plt.grid()
plt.plot(t1, l1)
plt.plot(t2, l2)
plt.plot(t3, l3)
plt.plot(t4, l4)
plt.plot(t5, l5)
plt.xlabel("Time (s)")
plt.ylabel("Absolute Loss")
plt.legend(["Ideal", "Ignore Stragglers", "Uncoded", "FRC", "FRC-Shuffling"], loc="upper right")
plt.title("Comparative Analysis - " + dataset + " Dataset")
plt.show()

shutil.rmtree("__pycache__")