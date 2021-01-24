import shutil
import matplotlib.pyplot as plt
import distributed_gradient_descent
import distributed_gradient_descent_ignore
import distributed_gradient_descent_uncoded
import distributed_gradient_descent_frc
import distributed_gradient_descent_frc_shuffling
import distributed_gradient_descent_crc
import distributed_gradient_descent_crc_shuffling

""""
Datasets: 

> boston        
> iris
> diabetes
> digits        
> linnerud
> wine
> breast_cancer

"""

dataset   = "breast_cancer"
N         = 8
time_stop = 0
s_ratio   = 0.5
alpha     = 0.001
iteration = [i for i in range(1001)]

# t1, l1 = distributed_gradient_descent.              run(dataset=dataset, N=N, time_stop=time_stop,                  alpha=alpha, adversarial=True, iteration=1000)
t2, l2 = distributed_gradient_descent_ignore.       run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha, adversarial=True, iteration=1000)
# t3, l3 = distributed_gradient_descent_uncoded.      run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha, adversarial=True, iteration=1000)
t4, l4 = distributed_gradient_descent_frc.          run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha, adversarial=True, iteration=1000)
t5, l5 = distributed_gradient_descent_frc_shuffling.run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha, adversarial=True, iteration=1000)
t6, l6 = distributed_gradient_descent_crc.          run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha, adversarial=True, iteration=1000)
t7, l7 = distributed_gradient_descent_crc_shuffling.run(dataset=dataset, N=N, time_stop=time_stop, s_ratio=s_ratio, alpha=alpha, adversarial=True, iteration=1000)

plt.grid()
# plt.plot(t1, l1)
plt.plot(iteration, l2)
# plt.plot(t3, l3)
plt.plot(iteration, l4)
plt.plot(iteration, l5)
plt.plot(iteration, l6)
plt.plot(iteration, l7)
plt.xlabel("Number of Iterations")
plt.ylabel("Absolute Loss")
plt.legend(["Ignore Stragglers", "FRC", "FRC-Shuffling", "CRC", "CRC-Shuffling"], loc="upper right")
plt.title("Comparative Analysis - " + dataset + " Dataset")
plt.show()

shutil.rmtree("__pycache__")