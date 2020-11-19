import numpy as numy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
epoch = [x for x in range(0, 1200, 50)] + [1221]
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
              'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']


fix, ax = plt.subplots(1, 1)

file_name = 'aps.log'
prec_list = []
with open(file_name, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.find('* All Loss') > -1:
            prec_list.append(float(line.split()[-3]))

ax.plot(epoch, prec_list, color_list[-2], label='With APS')

file_name = 'no_aps.log'
prec_list = []
with open(file_name, 'r') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if line.find('* All Loss') > -1:
            prec_list.append(float(line.split()[-3]))

ax.plot(epoch, prec_list, color_list[-1], label='Without APS')

ax.set_xlabel('iteration', fontsize=16)
ax.set_ylabel('testing accuracy', fontsize=16)

plt.tick_params(labelsize=12)

handles1, labels1 = ax.get_legend_handles_labels()
plt.legend(handles1, labels1, loc='buttom right', fontsize=12)
plt.show()
# plt.savefig('res18_lars.png')
