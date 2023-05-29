from matplotlib import pyplot as plt
import pickle

fig, ax =plt.subplots()

with open("training_data/dqn_prova", "rb") as f:
    list =pickle.load(f)

with open("training_data/ppo_prova", "rb") as f:
    list1 =pickle.load(f)
ax.plot(list1, color='r', label='dqn')
ax.plot(list, label='ppo')
plt.legend()
#ax.axhline(color='black', linewidth=1)
ax.set_xlabel("Episodes")
ax.set_ylabel("Average reward")
plt.show()

"""
with open("training_data/dqn_sp_exploit", "rb") as f:
    list =pickle.load(f)

with open("training_data/dqn_exploit", "rb") as f:
    list1 =pickle.load(f)
ax.plot(list1, color='r', label='dqn_exploit')
ax.plot( list, label='dqn_sp_exploit')
plt.legend()
ax.axhline(color='black', linewidth=1)
ax.set_xlabel("Episodes")
ax.set_ylabel("Average reward")

plt.show()"""


