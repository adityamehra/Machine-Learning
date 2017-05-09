from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd



data = pd.read_csv("../data/berkeley.csv", header= 0, sep=",")


countMaleAdmit = 0
countMaleReject = 0
countFemaleAdmit = 0
countFemaleReject = 0

for index, row in data.iterrows():
    if row["Gender"] == "Male" and row["Admit"] == "Admitted":
        countMaleAdmit += row["Freq"]
    elif row["Gender"] == "Male" and row["Admit"] == "Rejected":
        countMaleReject += row["Freq"]
    elif row["Gender"] == "Female" and row["Admit"] == "Admitted":
        countFemaleAdmit += row["Freq"]
    elif row["Gender"] == "Female" and row["Admit"] == "Rejected":
        countFemaleReject += row["Freq"]

counts = [ [ countMaleAdmit/(countMaleAdmit+countMaleReject)*100, countMaleReject/(countMaleAdmit+countMaleReject)*100 ] ]
counts.append( [ countFemaleAdmit/(countFemaleAdmit+countFemaleReject)*100, countFemaleReject/(countFemaleAdmit+countFemaleReject)*100 ] )
# counts = np.array(counts)
data1 = pd.DataFrame(counts, columns=['Admit', 'Reject'])
ax = data1.plot(kind = "bar", alpha = 1.0, rot=0)
ax.set_xticklabels(["Male", "Female"])
plt.xlabel('Gender', fontsize=18)
plt.ylabel(" Percentage", fontsize=16)
plt.show()

countMaleAdmits = { "A" : 0, "B" : 0, "C" : 0, "D" : 0, "E": 0, "F" : 0}
countMaleRejects = { "A" : 0, "B" : 0, "C" : 0, "D" : 0, "E": 0, "F" : 0}
countFemaleAdmits = { "A" : 0, "B" : 0, "C" : 0, "D" : 0, "E" :0, "F" : 0}
countFemaleRejects = { "A" : 0, "B" : 0, "C" : 0, "D" : 0, "E" : 0, "F" : 0}

depts = ["A", "B", "C", "D", "E", "F"]

for index, row in data.iterrows():
    for dept in depts:
        if row["Gender"] == "Male" and row["Admit"] == "Admitted" and row["Dept"] == dept:
            countMaleAdmits[dept] += row["Freq"]
        elif row["Gender"] == "Male" and row["Admit"] == "Rejected" and row["Dept"] == dept:
            countMaleRejects[dept] += row["Freq"]
        elif row["Gender"] == "Female" and row["Admit"] == "Admitted" and row["Dept"] == dept:
            countFemaleAdmits[dept] += row["Freq"]
        elif row["Gender"] == "Female" and row["Admit"] == "Rejected" and row["Dept"] == dept:
            countFemaleRejects[dept] += row["Freq"]

deptCounts = []

for dept in depts:
    deptCount = [countMaleAdmits[dept], countMaleRejects[dept], countFemaleAdmits[dept], countFemaleRejects[dept]]
    deptCounts.append(deptCount);

deptCounts = np.array(deptCounts)
print deptCounts.shape

# data1 = pd.DataFrame(deptCounts, columns=['Male Admit', 'Male Reject', 'Female Admit', 'Female Reject'])
# data1.plot.bar()
# plt.show()

deptCounts = []

for dept in depts:
    deptCount = [countMaleAdmits[dept] / (countMaleAdmits[dept] + countMaleRejects[dept]) * 100, countFemaleAdmits[dept]/(countFemaleAdmits[dept]+countFemaleRejects[dept]) * 100]
    deptCounts.append(deptCount);

data1 = pd.DataFrame(deptCounts, columns=['Male Admit', 'Female Admit'])
ax1 = data1.plot(kind='bar',alpha=1.0, rot=0)
ax1.set_xticklabels(depts)
plt.xlabel('Dept', fontsize=18)
plt.ylabel("Percentage", fontsize=16)
plt.show()

# print countMaleAdmits
# print countMaleRejects
# print countFemaleAdmits
# print countFemaleAdmits["A"]
# print countFemaleAdmits["A"] + countFemaleRejects["A"]
# print countFemaleAdmits["A"] / (countFemaleAdmits["A"] + countFemaleRejects["A"])
# print countFemaleRejects
