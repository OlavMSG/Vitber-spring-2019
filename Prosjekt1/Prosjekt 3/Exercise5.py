"""import """
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym


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
# NB! Vfunc(x) = V(x) / k

def Vfunc_5_1(x, h):
    return x

def Vfunc_5_2(x, h):
    if -3*h < x < 3*h:
        return 1
    else:
        return 0


def Vfunc_5_3(x, h):
    if x <= -3*h:
        return - 1
    elif x >= 3*h:
        return 1
    else:
        return - 1 + 2 * (x + 3) / 6



def randomWalk(Vfunc, betak, N, Ntime, h):
    pos = np.zeros(N, dtype=np.int)  # N particles in x=0.

    # random walk in 1D
    for t in range(Ntime):
        r_list = np.random.random(N)  # liste with N random numbers between 0 and 1
        for i in range(N):
            # probability of moving right
            P_pluss = 1 / (1 + np.e ** (- betak * (Vfunc(pos[i] - h, h) - Vfunc(pos[i] + h, h))))
            if r_list[i] < P_pluss:
                pos[i] += h  # One step to the right
            else:
                pos[i] -= h  # One step to the left

    return pos

def plotDist(ax, pos, Ntime):
        xMax = np.max(np.abs(pos))  # maximum absolute x position value
        xRange = (-xMax * 1.1, xMax * 1.1)  # Range for the plot
        xAx = np.linspace(*xRange, 1000)  # list og x values for Vfunc
        # Histogram, bins is given as the number of possible positions for a particle
        # density=True, for probability distribution
        ax.hist(pos, color='g', density=True, range=xRange, bins=(2 * Ntime + 1),
                label="Particle distribution")
        ax.set_xlim(*xRange)
        ax.set_xlabel("Number of steps")
        ax.set_ylabel("Particle distribution")
        return ax, xAx

def plotPotential(ax, xAx, Vfunc, h):
    V_list = np.array([Vfunc(x, h) for x in xAx])  # list of V values
    # Normalize V_list such that V_list has values between -1 and 1.
    V_list = V_list / np.max(np.abs(V_list))
    # new axis for V
    ax2 = ax.twinx()
    ax2.plot(xAx, V_list, "--k", label="Potential")
    ax2.set_ylabel("Potential")
    return ax2


def runAndPlotRandomWalk(Vfunc, betak_list, Exercise, nCols=2, N=1000, Ntime=100, h=1, save=False):
    # NB! Vfunc(x) = V(x) / k
    # N number of particles
    # Ntime number of timesteps
    # h  steplength

    numPlots = len(betak_list)
    nRows = numPlots // nCols + numPlots % nCols
    # make betak_list's length to nRows*nCols by
    while len(betak_list) < nRows * nCols:
        betak_list = np.append(betak_list, np.nan)
    # reshape beta_list
    betak_list = betak_list.reshape((nRows, nCols))

    savename = "RandomWalkIn1DInPotential" + Exercise[0] + Exercise[-1]
    fig, axs = plt.subplots(nRows, nCols, num=savename, figsize=(18, 6 * nRows))
    fig.suptitle("Random walk in 1D in potential " + Exercise, fontsize=20)
    if nCols == 1:
        axs = np.array([axs, ])
    if nRows == 1:
        axs = np.array([axs, ])

    for i in range(nRows):
        for j in range(nCols):
            betak = betak_list[i, j]
            if np.isnan(betak):
                # don't show plot
                axs[i, j].set_axis_off()
            else:
                pos = randomWalk(Vfunc, betak, N, Ntime, h)
                ax, xAx = plotDist(axs[i, j], pos, Ntime)
                ax.set_title("$\\beta k$ = " + "{:.2f}".format(betak))
                ax.grid()  # grid
                ax2 = plotPotential(ax, xAx, Vfunc, h)
                # Set ax's patch invisible
                ax.patch.set_visible(False)
                # move ax in front
                ax.set_zorder(ax2.get_zorder() + 1)
                if i == 0 and j == 0:
                    # legend
                    # asking matplotlib for the plotted objects and their labels
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax2.legend(lines + lines2, labels + labels2, loc=9,
                               bbox_to_anchor=(1.2, -0.15 - 1.3 * (nRows - 1)), ncol=2)
    # adjust
    plt.subplots_adjust(hspace=0.3, wspace=0.45)
    # save
    if save:
        plt.savefig(savename + ".pdf", bbox_inches='tight')


"""Exercise 5"""
Vfunc_dict = {'5.1': Vfunc_5_1, '5.2': Vfunc_5_2, '5.3': Vfunc_5_3}
betak_list = np.array([0.10, 0.50, 1.00, 5.00])
for Exercise in Vfunc_dict:
    runAndPlotRandomWalk(Vfunc_dict[Exercise], betak_list, Exercise)
plt.show()







