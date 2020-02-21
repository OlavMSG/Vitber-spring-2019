"""import """
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
from scipy.stats import norm


"""For nicer plotting"""
sym.init_printing()

fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (16, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)

"""Program"""
N = 1000  # number of particles
Ntime = 100  # number of timesteps
h = 1  # steplength

part_pos_list = np.zeros(N, dtype=np.int)  # N particles in x=0.

# random walk in 1D
for t in range(Ntime):
    r_list = np.random.random(N)  # list with N random numbers between 0 and 1
    for i in range(N):
        if r_list[i] >= 0.5:
            part_pos_list[i] += h  # One step to the right
        else:
            part_pos_list[i] -= h  # One step to the left

# find standard deviation mu and variance sigma to the normal distribution best suited to part_pos_list
mu, sigma = norm.fit(part_pos_list)
print("mu =", mu, "sigma =", sigma)

# pre-plotting
xMax = np.max(np.abs(part_pos_list))  # maximum absolute x position value
xRange = (-xMax * 1.1, xMax * 1.1)  # Range for the plot
xAx = np.linspace(*xRange, 1000)  # list og x values for normal distribution
p = norm.pdf(xAx, mu, sigma)  # normal distribution

# plotting
savename = "RandomWalkIn1D"
fig, ax = plt.subplots(1, 1, num=savename)
# new axis for p
ax2 = ax.twinx()
# Set ax's patch invisible
ax.patch.set_visible(False)
# move ax in front
ax.set_zorder(ax2.get_zorder() + 1)
# Histogram, bins is given as the number of possible positions for a particle
# distribution=False because True will mess with scaling
hist, bins = np.histogram(part_pos_list, bins=(2 * Ntime + 1), density=True, range=xRange)
widths = np.diff(bins)
hist *= 0.2
# 0.2 to match the normal distribution, because the particles jump an even number of times  (Ntime=100)
# all particles must end up on an even x-value. Therefore, the density of particles is one half of the amount of particles
# in the nearby point. Setting the area of the histogram to sum up to 0.2  to get a normal curve that "fits".
ax.grid()  # grid
ax.bar(bins[:-1], hist, widths, color='g', label="Particle distribution")
ax.set_xlim(*xRange)
ax.set_ylabel("Probability density")
ax.set_xlabel("Number of steps")
ax.set_title("Random walk in 1D\n$\\mu$ = " + "{:.2f}".format(mu) + " $\\sigma$ = " + "{:.2f}".format(sigma))


# plot p with ax, not ax2
ax2.plot(xAx, p, linewidth=2, color="r", label="Normal distribution")
ax2.set_ylim(ax.get_ylim())
# Turn off tick labels
ax2.set_yticklabels([])

# legend
# asking matplotlib for the plotted objects and their labels
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax.get_legend_handles_labels()
# legend outside plot
ax2.legend(lines + lines2, labels + labels2, loc=9, bbox_to_anchor=(0.5, -0.15), ncol=2)

# adjust and save
plt.subplots_adjust(bottom=0.2)
#plt.savefig(savename + ".pdf", bbox_inches='tight')
plt.show()




