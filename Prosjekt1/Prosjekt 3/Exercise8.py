"""import """
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import scipy.constants as spc


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
#Function that, given a list of positions and the with of the membrane (2h), calculates the number of particles in the cell
#and returns the concentration, using that 1 particle represents 0.1 mM.
def cons(pos, h):
    # Number of particles inside the cell.
    c = (pos < -h).sum()
    return c *1e-4

#Function that returns the potential, given an x value, the width of the membrane (2h), the time-dependent potential Vt
#and the constant potential corresponding to the specific ion in
def VTot(x, h, Vt, V_ion):
    if x < - h:
        return Vt
    elif x <= h:
        return - Vt * (x - h) / (2 * h) + V_ion
    else:
        return 0


# random walk in 1D
def step(pos, h, Vt, betaV_ion, L):
    N = len(pos)
    r_list = np.random.random(N)  # liste with N random numbers between 0 and 1
    for i in range(N):
        # probability of moving right
        P_pluss = 1 / (1 + np.e ** (-(VTot(pos[i] - h, h, Vt, betaV_ion) - VTot(pos[i] + h, h, Vt, betaV_ion))))
        if (pos[i] + h > L / 2):
            pos[i] -= h  # boundary condition, must move one step the left
        elif (pos[i] - h < - L / 2):
            pos[i] += h  # boundary condition, must move one step the right
        elif r_list[i] < P_pluss:
            pos[i] += h  # One step to the right
        else:
            pos[i] -= h  # One step to the left



def gateSwitch(betaVK, betaVNa, betaVt, betaVopen, betaV0closed, treshLow, treshHigh):
    if betaVt <= treshLow:
        betaVK = betaV0closed
        betaVNa = betaVopen
    if betaVt >= treshHigh:
        betaVK = betaVopen
        betaVNa = betaV0closed
    return betaVK, betaVNa


def SSP(posK, posNa, h):
    # find the args' that sort posK and posNa
    argK = np.argsort(posK)
    argNa = np.argsort(posNa)
    # find the args' of the posK sorted by argK that give values bigger than h, that will say outside the cell
    argK_out = np.argwhere(posK[argK] > h).flatten()
    # find the args' of the posNa sorted by argNa that give values less than -h, that will say inside the cell
    argNa_in = np.argwhere(posNa[argNa] < -h).flatten()

    # Do nothing if there are less than 2 K+ outside the cell or less than 3 Na+ inside the cell
    if (len(argK_out) >= 2) and (len(argNa_in) >= 3):
        # move the 2 K+ closest to the membrane to just inside the cell, -h - 1
        # ion's on the membrane +-h do not move
        posK[argK[argK_out[:2]]] = -h - 1
        # and move the 3 Na+ closest to the membrane to just outside the cell, h + 1
        # ion's on the membrane +-h do not move
        posNa[argNa[argNa_in[-3:]]] = h + 1


def randomWalkCell(betaVopen, betaV0closed, NNa_out, NK_out, NNa_in, NK_in, Ntime, h, L, Cc,
                   ebeta, treshLow, treshHigh, pump):
    # Constant outside consecration
    out_cons = (NNa_out + NK_out) * 1e-4


    # start for particles, in = start -L/4, out = start L/4, if L/4 is a decimal number, the decimals are cut of.
    posK = np.concatenate((np.full(NK_out, L / 4, dtype=np.int), np.full(NK_in, - L / 4, dtype=np.int)))
    posNa = np.concatenate((np.full(NNa_out, L / 4, dtype=np.int), np.full(NNa_in, - L / 4, dtype=np.int)))

    # arrays for V(t) and t
    VtArr = np.zeros(Ntime + 1, dtype=np.float)
    tArr = np.arange(Ntime + 1)  # array with int from 0 to Ntime.
    betaVK, betaVNa = 0, 0
    for t in range(Ntime + 1):  # t goes between 0 and Ntime
        # calculate for the previous step in the simulation
        consK_in = cons(posK, h)  # inside consecration of K+
        consNa_in = cons(posNa, h)  # inside consecration of Na+
        # calculate dimensionless (relative) energy
        betaVt = ((consK_in + consNa_in) - out_cons) * ebeta / Cc
        VtArr[t] = betaVt

        if t == Ntime:
            # end simulation, do not do another time step
            break

        # Do one step in the simulation
        if pump:
            if t % 10 == 0:
                SSP(posK, posNa, h)

        betaVK, betaVNa = gateSwitch(betaVK, betaVNa, betaVt, betaVopen, betaV0closed, treshLow, treshHigh)
        # Do one time step
        step(posK, h, betaVt, betaVK, L)
        step(posNa, h, betaVt, betaVNa, L)

    return tArr, VtArr, posK, posNa

def plotDist(ax, pos, L):
    xMax = np.max(np.abs([np.min(pos), np.max(pos)]))  # maximum absolute x position value
    xRange = (-xMax * 1.1, xMax * 1.1)  # Range for the plot
    xAx = np.linspace(*xRange, 1000)  # list og x values for Vfunc
    # Histogram, bins is given as the number of possible positions for a particle
    # density=True, for probability distribution
    ax.hist(pos, color='g', density=True, range=xRange, bins=(L + 1),
            label="Particle distribution")
    ax.set_xlim(*xRange)
    ax.set_xlabel("Number of steps")
    ax.set_ylabel("Particle distribution")
    return ax


def plotVoltage(ax, t, V, ebeta):
    ax2 = ax.twinx()
    ax2.plot(t, V, label="$V(t)$")
    ax2.set_ylabel("$\\beta e \\mathrm{V}$")
    ax.set_ylim(np.array(ax2.get_ylim()) / ebeta *1e3)
    ax.set_ylabel("$\\mathrm{mV}$")
    ax.set_xlabel("t")
    return ax, ax2

def runAndplotV(Ntime, Exercise, betaVopen=1, betaV0closed=50, NNa_out=1450, NK_out=50,  NNa_in = 50,
                NK_in = 1400, h=1, L=50, Cc=0.07, T0=310, treshLow=-70e-3, treshHigh=30e-3, pump=True, save=False, hist=False):
    # NNa_out and NK_out Number of particle outside the cell
    #  NNa_in and NK_in Number of particle inside the cell
    # Ntime Number of time steps
    # h Step length
    # L System length
    # T0, temperature
    # Conversion factor to go from energy to the dimensionless "energy" used.
    ebeta = spc.e / (spc.k * T0)
    # boundary condition for the gateSwitch, make dimensionless
    treshLow *= ebeta
    treshHigh *= ebeta

    savename = "ActionPotential" + Exercise[0] + Exercise[-1]
    fig, ax = plt.subplots(1, 1, num=savename)
    tArr, VtArr, posK, posNa = randomWalkCell(betaVopen, betaV0closed, NNa_out, NK_out, NNa_in, NK_in, Ntime, h, L, Cc,
                                              ebeta, treshLow, treshHigh, pump)
    ax, ax2 = plotVoltage(ax, tArr, VtArr, ebeta)
    pumptext = {'8.1': '', '8.2': ' with SSP'}
    ax.set_title("Action potential V(t)" + pumptext[Exercise])
    ax.grid()
    # legend
    # asking matplotlib for the plotted objects and their labels
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc=2, bbox_to_anchor=(0.88, 1.13), ncol=1)
    # save, default False
    if save:
        plt.savefig(savename + ".pdf", bbox_inches='tight')
    # make histogram of particle distribution, default False
    if hist:  #
        # histogram K
        fig_2, ax_2 = plt.subplots(1, 1, num=savename + "histK")
        ax_2 = plotDist(ax_2, posK, L)
        lines_2, labels_2 = ax.get_legend_handles_labels()
        ax_2.grid()
        ax_2.set_title("Histogram K" + pumptext[Exercise])
        ax_2.legend(lines_2, labels_2, bbox_to_anchor=(0.3, -0.15), ncol=2)
        # save, default False
        if save:
            plt.savefig(savename + "histK.pdf", bbox_inches='tight')
        # histogram Na
        fig_3, ax_3 = plt.subplots(1, 1, num=savename + "histNa")
        ax_3 = plotDist(ax_3, posNa, L)
        lines_3, labels_3 = ax.get_legend_handles_labels()
        ax_3.grid()
        ax_3.set_title("Histogram Na" + pumptext[Exercise])
        ax_3.legend(lines_3, labels_3, bbox_to_anchor=(0.3, -0.15), ncol=2)
        # save, default False
        if save:
            plt.savefig(savename + "histNa.pdf", bbox_inches='tight')


# Ntime Number of time steps (first parameter)
"""Exercise 8.1"""
runAndplotV(1000, '8.1', pump=False)
"""Exercise 8.2"""
runAndplotV(5000, '8.2')

plt.show()
















