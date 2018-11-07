import matplotlib.pyplot as plt

filename = 'logger.log'
with open(filename) as file_object:
    index=0
    x_list = []
    gpu_list = []
    cpu_list = []
    for line in file_object:
        temp = line.rstrip("\n").split(":")
        # print(temp)
        if index == 0:
            x_list.append(float(temp[3]))
        elif index == 1:
            gpu_list.append(float(temp[3]))
        elif index == 2:
            cpu_list.append(float(temp[3]))
        index = (index+1)%3
    # print(x_list)
    # print(gpu_list)
    # print(cpu_list)
    plt.subplot(121)
    plt.plot(x_list,gpu_list,label='gpu')
    plt.plot(x_list,cpu_list,label='cpu')
    plt.title("GPU vs CPU")
    plt.xlabel('parallellism thread number')
    plt.ylabel("time(ms)")

    plt.subplot(122)
    plt.plot(x_list,gpu_list,label='gpu')
    plt.title("GPU")
    plt.xlabel('parallellism thread number')
    plt.ylabel("time(ms)")

    plt.legend()#显示图例
    plt.show()

