import numpy as np
import matplotlib.pyplot as plt
import re

input_file = "alltoall_dvv_stretching_UNM_UNM-stacks_10_0.1-0.2Hz_40.0-100.0s.npz.npy"
alpha = 1
beta = 1
stackstep = 10. * 86400.  # how long the original stacks were (to correct for gaps!)
fill_value = 0.0
fill_value_err = 100.0

all_to_all = np.load(input_file, allow_pickle=True)
dat = dict(all_to_all.tolist())
print(dat["dvv_data"].shape)
print(dat["timestamps"].shape)
dvv_par = dat["dvv_data"][0:4950]
dvv_err = dat["dvv_err"][0:4950]
t = dat["timestamps"][0:100]

# step 1: Correct the time axis, i.e. fill in gaps
t_clean = []
current_tmin = t[0]
t_clean.append(t[0])
print(t)
for ixt, t_or in enumerate(t):
    if ixt == 0:
        continue
    # print(t_or)
    # print(t_clean)
    # only let time axis increase (no overlaps)
    if t_or > current_tmin:
        current_tmin = t_or
    else:
        print("overlap", t_or, t[ixt-1])
        continue  # if overlapping, jump to the next
    
    if t_or - t[ixt - 1] >= 2 * stackstep:
        # we have a gap
        ttemp = t[ixt - 1] + stackstep
        while ttemp <= t_or - stackstep:
            t_clean.append(ttemp)
            ttemp += stackstep
        t_clean.append(t_or)
    else:
        # no gap, just normal
        t_clean.append(t_or)
n = len(t_clean)
n_old = len(t)
# print(t_clean)
#            perc = (t[ixt + 1] - ttemp) / (t[ixt + 1] - t_or)

colorst = np.linspace(0, t[-1], n_old)
colorstclean = np.linspace(0, t_clean[-1], n)
plt.scatter(colorst, t, marker="o", c=colorst)
plt.scatter(colorstclean, t_clean, marker="v",
            c=colorstclean, cmap=plt.cm.PuOr)
for tt in t:
    plt.hlines(xmin=0, xmax=colorst[-1], y=tt, alpha=0.4)
plt.show()


dvv_clean = []
err_clean = []
for i in range(n):
    if t_clean[i] not in t:
        # gap
        for j in range(i + 1, n):
            dvv_clean.append(fill_value)
            err_clean.append(fill_value_err)
    else:
        ix1 = np.where(t == t_clean[i])[0][0]
        for j in range(i + 1, n):
            if t_clean[j] not in t:
                # gap
                err_clean.append(fill_value_err)
                dvv_clean.append(fill_value)
            else:
                # find global index of sample
                ix2 = np.where(t == t_clean[j])[0][0]
                global_index = 0
                for k in range(ix1):
                    for l in range(k + 1, n_old):
                        global_index += 1
                global_index += (ix2 - ix1 - 1)

                # get the dvv sample and append
                dvv_clean.append(dvv_par[global_index])
                err_clean.append(dvv_err[global_index])

plt.scatter(t[1:n_old], dvv_par[:n_old-1], alpha=0.5)
plt.scatter(t_clean[1:n], dvv_clean[0: n-1], marker="x", alpha=0.5, color="r")
plt.scatter(t[2:n_old], dvv_par[n_old-1: 2*n_old-3], alpha=0.5)
plt.scatter(t_clean[2:n], dvv_clean[n-1: 2*n-3], marker="x", alpha=0.5, color="g")
plt.scatter(t[3:n_old], dvv_par[2*n_old-3: 3*n_old-6], alpha=0.5, color="lightgreen")
plt.scatter(t_clean[5:n], dvv_clean[4*n-10: 5*n-15], marker="x", alpha=0.5, color="purple")
plt.show()


results = {}
results["dvv_max"] = dat["dvv_max"]
results["input_data"] = dat["input_data"]
results["f0Hz"] = dat["f0Hz"]
results["f1Hz"] = dat["f1Hz"]
results["w0s"] = dat["w0s"]
results["w1s"] = dat["w1s"]
results["dvv_data"] = dvv_clean
results["dvv_err"] = err_clean
results["timestamps"] = t_clean
output_file = re.sub("alltoall", "ungap", input_file)
np.save(output_file, results)
