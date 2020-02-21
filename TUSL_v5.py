# -*- coding: utf-8 -*-
"""import start"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym
import scipy.constants as spc
import scipy.linalg as spl

"""import slutt"""

"""for pen plotting start"""
# Initialiserer pen visning av uttrykkene
sym.init_printing()

# Plotteparametre for C% fC% store, tydelige plott som utnytter tilgjengelig skjermareal
fontsize = 20
newparams = {'axes.titlesize': fontsize, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 7,
             'figure.figsize': (16, 7), 'ytick.labelsize': fontsize,
             'xtick.labelsize': fontsize, 'legend.fontsize': fontsize,
            'legend.handlelength': 1.5}
plt.rcParams.update(newparams)
"""for pen plotting slutt"""

"""tester start"""
def testEvalues(N, w, w2, r):
    ok = 0
    for i in range(N):
        if np.round(w[i], r) == np.round(w2[i], r):
            ok += 1
    print("Egenverdier like analytisk og numerisk: ", ok, "av", N, "dette gjelder opp til", r, "desimal")


def testPsiAbsKvadrtLik1(N, psi):
    ok = np.sum(np.sum(psi**2, axis=1))
    print("Antall egenvektorer som har absoultkvadratsum 1: ", np.round(ok, 10), "av", N)


def testOrtogonalitetOgFullstendihet(N, psi):
    test = np.matmul(psi, psi.T)
    sum1 = np.sum(test)  # summen av hele matrisen
    sum2 = np.trace(test)  # summen langs diagonalen
    if np.round(sum1, 10) == np.round(sum2, 10) == N:
        # Hvis dette stemmer er psi.T = psi.inverse og dermed er matrix psi orthogonal
        # og invertibel, hvilket betyr at vektorene(Bølgefunksjonene) i psi er ortogonale
        # og at de spenner R^N, altså vi har et fullstedig sett.
        print("Bølgefunksjonen er ortogonale og danner et fullstedingsett")




"""tester start slutt"""

"""Lage analytisk boks start"""
def makepsi2andw2(N, Dx):
    L = (N + 1) * Dx
    w2 = np.array([spc.hbar ** 2 * (j * spc.pi / L) ** 2 / (2 * spc.m_e) for j in range(1, N + 1)])
    psi2 = np.zeros(shape=(N, N))
    x = np.linspace(1, N, N) * Dx
    for j in range(1, N + 1):
        psi2[:, j - 1] = np.sqrt(2 / L) * np.sin(j * spc.pi / L * x)
    psi2 = psi2 * np.sqrt(Dx) # konstant som fikser ortognalitet
    return psi2, w2


"""Lage analytisk boks slutt"""


"""Finne E 'analytisk' atom"""
def FinnEAnalytiskAtom(N, L, V_0, w, Nbot):
    print("Finner analytiske E-verdier numerisk, vent litt...")
    l = L / 2
    V_0 = V_0
    bundet = 0
    E_list = np.zeros(N)
    for i in range(N):
        if w[i] < V_0:
            bundet += 1
    for n in range(N):
        # print(n)  # viser at løkka går
        # Antar analytisk i nærheten av nummerisk
        potE_list = np.linspace(w[n] - 1e-26, w[n] + 1e-26, Nbot)
        if n%2 == 1:
            test_list = np.abs(np.array([np.tan(np.sqrt(2 * spc.m_e * Ei) * l / spc.hbar)
                                         - np.sqrt((V_0 - Ei) / Ei, dtype=np.complex128) for Ei in potE_list]))
        else:
            test_list = np.abs(np.array([np.tan(np.sqrt(2 * spc.m_e * Ei) * l / spc.hbar)
                                         + np.sqrt(Ei / (V_0 - Ei), dtype=np.complex128) for Ei in potE_list]))
        k = np.argmin(test_list)
        E_list[n] = potE_list[k]
    print("Analytiske E-verdier funnet numerisk")
    testEvalues(N, w, E_list, 5)

"""potensial start"""

def Vfunc(V_0, b, omega, N_omega):
    # Lager potensiale og finner N, L gitt lengdene omega og b,
    # og antall brønner N_omega
    # omega er en lengden på en brønn, nødvendigvis ikke et heltall n ganger Dx (n * Dx), runder derfor av til nærmeste heltall,
    # samme for b, lengden mellombrønner. Lengden representeres da av round(omega / Dx, 0) * Dx og round(b / Dx, 0) * dx.
    # må også være av typen int
    omega = int(round(omega / Dx, 0))
    b = int(round(b / Dx, 0))

    k = 10 * omega  # minste avstand
    if k == 0:
        k = 500 # vil ha mist N = 1000

    V = np.array([V_0] * k)
    minx = len(V) # start på brønner

    V = np.append(V, np.array(([0] * omega + [V_0] * b) * N_omega))

    maxx = len(V) - b # slutt på brønner
    if k >= b:
        V = np.append(V, np.array([V_0] * (k - b)))
    else:
        V = V[:maxx + k] # må fjerne litt bakerst
    N = len(V)
    L = (maxx - minx + 1) * Dx * 1e9  # lengde på Atom/ Molekyl i nm
    return V, N, L, minx, maxx


"""potensial slutt"""

"""TSUL"""
def TUSL(Dx, N, V):
    # Gitt Dx, N, og V, lager matrisen H og
    # Finner w = egenverdiene (J) og psi = egenvektorene til matrisen H
    # d = liste med diagonalelementer i Hamiltonmatrisen H
    d = V + spc.hbar ** 2 / (spc.m_e * Dx ** 2)
    # e = verdi til ikke-diagonale elementer i H, dvs H(i,i+-1)
    e = np.array([- spc.hbar ** 2 / (2 * spc.m_e * Dx ** 2)]*(N-1))
    # Initialisering av matrisen H: Legger inn verdi 0 i samtlige elementer
    # Finner w = egenverdiene (J) og psi = egenvektorene til matrisen H
    w, psi = spl.eigh_tridiagonal(d, e)
    return w, psi


"""TSUL slutt"""


"""Printing av egenverdier og plotting start"""
def printEvalues(w, V_0, navn, r, antall):
    # evalues = liste med energiegenverdier i enheten eV
    evalues = np.round(w / spc.electron_volt, r)
    V_0 = V_0 / spc.electron_volt
    # Skriver ut de 6 laveste energiegenverdiene i enheten eV
    # print(w[0],w[1],w[2],w[3],w[4],w[5])
    print("V_0 =", round(V_0, 0), "eV")
    print("De", antall, "første egenverdiene for", navn, "i enheten eV er: ")
    string = ""
    antallbudet = 0
    for i in range(antall):
        if evalues[i] < V_0:
            antallbudet += 1
        string += str(evalues[i]) + " "
        if (i + 1)%10 == 0:
            string += "\n"
    print(string)
    print("Antall bundene tilstander er:", antallbudet, "av de", antall, "første tilstandene.")


def plotWaveFunctionsWithV(N, psi, V, Dx, navn, antall_list, k, r):
    # k, skaleringskonstant
    # V til ev
    V = V * k / spc.electron_volt
    # z = liste med posisjonsverdier (nm)
    z = np.linspace(1, N, N) * Dx * 1e9
    # Plotter bølgefunksjonene som tilsvarer de 4 laveste egenverdiene
    plt.figure('Bølgefunksjoner med V ' + navn)
    plt.axhline(y=0, linewidth=1, color='k')
    plt.plot(z, V - r, label='$V$', color='b')
    c = 0
    colors = np.array(['m', 'g', 'r', 'c'])
    psi = psi
    for i in antall_list:
        labeli = "$\psi_{" + str(i + 1) + "}$"
        plt.plot(z, psi[:, i], label=labeli, color=colors[c])
        c += 1
        if c == 4:
            c = 0
    plt.title('Bølgefunksjoner med V ' + navn, fontsize=20)
    plt.xlabel('$z$ (nm)', fontsize=20)
    ylabel = '$\psi$ / $E$'
    plt.ylabel(ylabel, fontsize=20)
    plt.legend(loc='upper left')


def plotEnergyLevels(N, w, V, Dx, navn, antall):
    # evalues = liste med energiegenverdier i enheten eV
    evalues = w / spc.electron_volt
    # V til ev
    V = V / spc.electron_volt
    # z = liste med posisjonsverdier (nm)
    z = np.linspace(1, N, N) * Dx * 1e9
    # Plotter potensialet og en strek for hver av de 4 laveste egenverdiene
    # Samme fargerekkefølge som for bølgefunksjonene
    # (b=blå, m=magenta g=grønn, r=rød, c=cyan)
    plt.figure('Energinivå ' + navn)
    plt.plot(z, V, label='$V$', color='b')
    c = 0
    colors = np.array(['m', 'g', 'r', 'c'])
    for i in range(antall):
        plt.axhline(y=evalues[i], linewidth=2, color=colors[c])
        c += 1
        if c == 4:
            c = 0
    plt.title('Energinivå ' + navn, fontsize=20)
    plt.xlabel('$z$ (nm)', fontsize=20)
    plt.ylabel('$E$ (eV)', fontsize=20)
    # plt.xlim(np.min(z) - Dx, np.max(z) + Dx)
    plt.legend(loc='lower left')


def plotPobabilityDensityWithV(N, psi, V, Dx, navn, antall_list, k, r):
    # k, skaleringskonstant
    # V til ev
    V = V * k / spc.electron_volt
    # z = liste med posisjonsverdier (nm)
    z = np.linspace(1, N, N) * Dx * 1e9
    # Plotter sannsynlighetstettheten for de 4 laveste tilstandene
    plt.figure('Sannsynlighetstetthet med V ' + navn)
    plt.axhline(y=0, linewidth=1, color='k')
    plt.plot(z, V - r, label='$V$', color='b')
    c = 0
    colors = np.array(['m', 'g', 'r', 'c'])
    for i in antall_list:
        labeli = "$|\psi_{" + str(i + 1) + "}|^2$"
        plt.plot(z, np.abs(psi[:, i]) ** 2, label=labeli, color=colors[c])
        c += 1
        if c == 4:
            c = 0
    plt.title('Sannsynlighetstetthet med V ' + navn, fontsize=20)
    plt.xlabel('$z$ (nm)', fontsize=20)
    ylabel = '$|\psi|^2$ / $E$'
    plt.ylabel(ylabel, fontsize=20)

    plt.legend(loc='upper left')


def plotEnergibandbreddensomFunkAvN_omega(B_list, N_omega_list, bandnummer):
    bandnummer = str(bandnummer)
    B_list = B_list / spc.electron_volt
    plt.figure("Energibåndbredde som funksjon av N_omega, Energibånd " + bandnummer)
    plt.plot(N_omega_list, B_list)
    plt.title("Energibåndbredde $\Delta E_{" + bandnummer + "}(N_{\omega})$", fontsize=20)
    plt.xlabel("$N_{\omega}$")
    plt.ylabel("$\Delta E_{" + bandnummer + "}$(eV)", fontsize=20)
    # for lagring av plott, siden det tar lang tid å kjøre
    # Ninfo = str(len(N_omega_list))+ '_max' + str(np.max(N_omega_list))
    # plt.savefig("DeltaE_"+ båndnummer + "_N_omega_len" + Ninfo + ".pdf")


def printBindingsenergiH_2(w1, w2, r):
    w1 = w1 / spc.electron_volt
    w2 = w2 / spc.electron_volt
    H = round(w1[0], r)
    H2 = round(2 * w1[0], r)
    H_2 = round(2 * w2[0], r)
    BH_2 = 2 * w1[0] - 2 * w2[0]
    BH_2p = round(BH_2, r + 2)
    print("Energi ett H atom", H, "eV\nEnergi to H atom", H2,
          "eV\nEnergi ett H_2 molekyl", H_2, "eV\nBindingsenergi H_2", BH_2p, "eV")
    BH_2sann = 4.5  # funnet på nett
    k = round(BH_2sann / BH_2, 2)
    print("Bindingsenergien for H_2 regnet ut her er ca.", k,
          "ganger mindre enn virkelig verdi.")

def printBindingsenergiHe_2(w1, w2, r):
    w1 = w1 / spc.electron_volt
    w2 = w2 / spc.electron_volt
    He = round(2 * w1[0], r)
    He2 = round(4 * w1[0], r)
    He_2 = round(2 * w2[0] + 2 * w2[1], r)
    BHe_2 = 4 * w1[0] - (2 * w2[0] + 2 * w2[0])
    BHe_2p = round(BHe_2, r + 9)
    print("Energi ett He atom", He, "eV\nEnergi to He atom", He2,
          "eV\nEnergi ett He_2 molekyl", He_2, "eV\nBindingsenergi He_2", BHe_2p, "eV")
    BHe_2sann = 1e-7  # funnet på nett
    k = round(BHe_2sann / BHe_2, 2)
    print("Bindingsenergien for He_2 regnet ut her er ca.", k,
          "ganger mindre enn virkelig verdi.")

def printHalvlederVsIsolator(w, N_omega, r):
    w = w / spc.electron_volt
    E2max = round(w[2* N_omega - 1], r)
    print("Energien til elektorene med høyest energi er:", E2max, "eV")
    E2g = w[2 * N_omega] - w[2 * N_omega - 1]
    print("Båndgapet opp til neste ledige tilstand er:", round(E2g, r), "eV")
    if 0 < E2g < 3:
        print("Materialet (krystallen) er en halvleder.")
    else:
        print("Materialet (krystallen) er en isolator.")

def printBolgelengdeOgK(w, V_0, nummer, r):
    if w[nummer - 1] > V_0:  # ubunden tilstand
        # utenfor brønnområde
        E_ku = w[nummer - 1] - V_0
        lambda_u = 2 * spc.pi / (np.sqrt(2 * spc.m_e * E_ku) / spc.hbar)
        E_ku = round(E_ku / spc.electron_volt, r)
        lambda_u = round(lambda_u * 1e9, r)
        print("Utenfor brønnområde er E_k =", E_ku, "som gir bølgelengden", lambda_u, "nm, for den ubundene tilstanden n =", nummer)
        # innenfor brønnområde
        E_ki = w[nummer - 1]
        lambda_i = 2 * spc.pi / (np.sqrt(2 * spc.m_e * E_ki) / spc.hbar)
        E_ki = round(E_ki / spc.electron_volt, r)
        lambda_i = round(lambda_i * 1e9, r)
        print("Innenfor brønnområde er E_k =", E_ki, "eV, som gir bølgelengden", lambda_i,
              "nm, for den ubundene tilstanden n =", nummer)

"""Printing av egenverdier og plotting slutt"""

"""Tidsutvikling"""
def makePsi0(p0, x0, sigma, N, Dx):
    k0 = p0 / spc.hbar
    x = np.linspace(1, N, N) * Dx
    normfactor = (2 * np.pi * sigma ** 2) ** (-0.25)
    gaussinit = np.exp(-(x - x0) ** 2 / (4 * sigma ** 2))
    planewavefactor = np.exp(1j * k0 * x)
    Psi0 = normfactor * gaussinit * planewavefactor
    return Psi0

def plotPsi0withV(N, Psi0, V, Dx, navn, r):
    z = np.linspace(1, N, N) * Dx * 1e9
    plt.figure(navn)
    plt.plot(z, V - r, label='$V$', color='b')
    plt.plot(z, np.abs(Psi0 ** 2), label='$|\Psi(x,0)|^2$', color='m')
    plt.title(navn, fontsize=20)
    plt.xlabel('$z$ (nm)', fontsize=20)
    plt.ylabel('$|\Psi(x,0)|^2$', fontsize=20)

def makePsi_t_list(Psi0, psi, w, p0, N, Dx, Ntid):
    psi_Complex = psi * (1.0 + 0.0j)
    c = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        c[n] = np.vdot(psi_Complex[:, n], Psi0)
        # Setter tidssteget til 1/100 av tiden det tar for bølgepakken å
        # flytte seg en lengde lik lengden av hele z-området
    v0 = p0 / spc.m_e
    tidssteg = N * Dx / (v0 * 100)
    tid = np.linspace(0, Ntid - 1, Ntid) * tidssteg
    Psi_t_list = np.array([[0] *N] * Ntid, dtype=np.complex128)
    Psi_t = np.zeros(N, dtype=np.complex128)
    i = 0
    for t in tid:
        for n in range(N):
            Psi_t = Psi_t + c[n] * psi_Complex[:, n] * np.exp(-1j *w[n] * t / spc.hbar)
        Psi_t_list[i] = Psi_t
        i += 1
    rhomax = np.max(np.abs(Psi_t_list[0])) #Psi_t_list[0] er den rekonstruerte Psi[0]
    return Psi_t_list, rhomax





"""Kjør program start"""




"""!!!!Selve kjøringen!!!!!"""

"""Liten bruker interface..."""
boks, atom, molekyl, krystall4a, krystall4b, krystall4cd, tidsutvikling = False, False, False, False, False, False, False
print("0) Avslutt\n1) Boks, oppgave 1\n2) Atom, oppgave 2a\n3) Molekyl, oppgave 3"
      "\n4) Krystaller, oppgave 4a\n5) Krystall, oppgave 4b, Tar kanskje 5 min!"
      "\n6) Krystaller, oppgave 4cd, kjører N_omega = 100\n7) Tidsutvikling av 4d")
while True:
    try:
        valg = int(input("Valg (0-7): "))
        break
    except ValueError:
        print("Ikke et tall, prøv igjen.")

if valg == 1:
    # oppgave 1
    boks = True
elif valg == 2:
    # oppgave 2a, kjører kun 2a!!!
    atom = True
elif valg == 3:
    # oppgave 3, kjører også 2a!!!
    molekyl = True
elif valg == 4:
    krystall4a = True
elif valg == 5:
    krystall4b = True
elif valg == 6:
    krystall4cd = True
elif valg == 7:
    tidsutvikling = True
else:
    ingenting = "absolut ingenting" # gjør ingenting


"""Tegn forklaring"""
# Dx, lengde på delintervall
# omega er en lengden på en brønn, nødvendigvis ikke et heltall n ganger Dx (n * Dx), runder derfor av til nærmeste heltall,
# samme for b, lengden mellombrønner. Lengden representeres da av round(omega / Dx, 0) * Dx og round(b / Dx, 0) * dx.
# N_omega er antall brønner, heltall
# V_0 potensial brønnens dybde
# L(m)
# antallplot = liste med hvilke egenverdier og egenvektorer som plottes
# antallprint = antall egenverdier som printes
# antallplotE = antall energinivå som plottes i energinivåplotett



if boks: # oppgave 1
    navn = "Partikkel i boks numerisk"
    Dx = 5e-12
    antallplot = np.arange(4, step=1)
    antallprint = 6
    antallplotE = 6

    print(navn)
    V, N, L, minx, maxx = Vfunc(0, 0, 0, 0)
    print("Antall delinetvall N =", N, "\nSkrittlengde =", Dx)
    w, psi = TUSL(Dx, N, V)
    printEvalues(w, 0, navn, 9, antallprint)
    print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N, psi, V, Dx, navn, antallplot, 0, 5e-2) # har ikke betydning her
    plotPobabilityDensityWithV(N, psi, V, Dx, navn, antallplot, 0, 1e-4)
    plotEnergyLevels(N, w, V, Dx, navn, antallplotE)
    # tester
    testPsiAbsKvadrtLik1(N, psi)
    testOrtogonalitetOgFullstendihet(N, psi)

    # Lager psi2 og w2 analytisk
    navn = "Partikkel i boks analytisk"
    print(navn)
    print("Antall delinetvall N =", N, "\nSkrittlengde =", Dx)
    psi2, w2 = makepsi2andw2(N, Dx)
    printEvalues(w2, 0, navn, 9, antallprint)
    print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N, psi, V, Dx, navn, antallplot, 0, 5e-2)
    plotPobabilityDensityWithV(N, psi, V, Dx, navn, antallplot, 0, 1e-4)
    plotEnergyLevels(N, w2, V, Dx, navn, antallplotE)
    # tester
    testPsiAbsKvadrtLik1(N, psi2)
    testOrtogonalitetOgFullstendihet(N, psi2)
    # sjekker om egenverdier er like, siste tall er avrunding til r desimaler
    testEvalues(N, w, w2, 10)
    plt.show()

if atom:
    # oppgave 2a
    navn = "Atom"
    Dx = 5e-12 # gir potensialbrønn i atom rundt 1 - 5 Å ssom er gjennomsnittlig atom radius
    omega = 50 * Dx # = 2.5Å
    V_0 = 36 * spc.electron_volt
    antallplot = np.arange(2, step=1)
    antallprint = 8
    antallplotE = 4

    print(navn)
    V1, N1, L1, minx1, maxx1 = Vfunc(V_0, 0, omega, 1)
    print("Antall delinetvall N =", N1, "\nSkrittlengde =", Dx, "\nAtombredde =", L1, "nm")
    w1, psi1 = TUSL(Dx, N1, V1)
    printEvalues(w1, V_0, navn, 9, antallprint)
    # nest siste tall er hvilken indeksen til tilstanden = 1, 2, 3, 4, ...
    # siste tall er presisjon i utskrift
    printBolgelengdeOgK(w1, V_0, 4, 2)
    # siste tall er Nbot = antall element som iteres over for å finne en E-Verdi
    FinnEAnalytiskAtom(N1, L1, V_0, w1, 1e4)
    print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N1, psi1, V1, Dx, navn, antallplot, 8e-4, 2.1e-1)
    plotPobabilityDensityWithV(N1, psi1, V1, Dx, navn, antallplot, 8e-5, 3.5e-3)
    plotEnergyLevels(N1, w1, V1, Dx, navn, antallplotE)
    plt.show()

if molekyl:
    # hydrogen
    # oppgave 3, trenger data fra oppgave 2a!!
    # oppgave 2a
    navn = "Hydrogen Atom"
    Dx = 5e-12  # gir potensialbrønn i atom rundt 1 - 5 Å ssom er gjennomsnittlig atom radius
    omega = 50 * Dx  # = 2.5Å
    V_0 = 36 * spc.electron_volt
    antallprint = 4
    antallplotE = 4

    print(navn)
    V1, N1, L1, minx1, maxx1 = Vfunc(V_0, 0, omega, 1)
    print("Antall delinetvall N =", N1, "\nSkrittlengde =", Dx, "\nAtombredde =", L1, "nm")
    w1, psi1 = TUSL(Dx, N1, V1)
    printEvalues(w1, V_0, navn, 9, antallprint)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    plotEnergyLevels(N1, w1, V1, Dx, navn, antallplotE)

    navn = "DiHydrogen Molekyl"
    # Dx, omega og V_0 gitt av atom!!! Oppgave 2a, kjøres automatisk på forhånd
    b = 15 * Dx # = 0.74Å
    antallplot = np.arange(4, step=3)
    antallprint = 7
    antallplotE = 7

    print(navn)
    V2, N2, L2, minx2, maxx2 = Vfunc(V_0, b, omega, 2)
    print("Antall delinetvall N =", N2, "\nSkrittlengde =", Dx, "\nMolekylbredde =", L2, "nm")
    w2, psi2 = TUSL(Dx, N2, V2)
    printEvalues(w2, V_0, navn, 9, antallprint)
    print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N2, psi2, V2, Dx, navn, antallplot, 8e-4, 1.6e-1)
    plotPobabilityDensityWithV(N2, psi2, V2, Dx, navn, antallplot, 8e-5, 3.2e-3)
    plotEnergyLevels(N2, w2, V2, Dx, navn, antallplotE)
    # sitte parameter styrer antall desimaler i svar
    printBindingsenergiH_2(w1, w2, 3)

    # helium
    # oppgave 3, trenger data fra oppgave 2a!!
    # oppgave 2a
    navn = "Helium Atom"
    antallprint = 4
    antallplotE = 4

    print(navn)
    V6, N6, L6, minx6, maxx6 = Vfunc(V_0, 0, omega, 1)
    print("Antall delinetvall N =", N6, "\nSkrittlengde =", Dx, "\nAtombredde =", L6, "nm")
    w6, psi6 = TUSL(Dx, N1, V1)
    printEvalues(w6, V_0, navn, 9, antallprint)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    plotEnergyLevels(N6, w6, V6, Dx, navn, antallplotE)

    navn = "Dihelium Molekyl"
    # Dx, omega og V_0 gitt av atom!!! Oppgave 2a, kjøres automatisk på forhånd
    b = 1040 * Dx  # = 52Å
    antallplot = np.arange(4, step=1)
    antallprint = 7
    antallplotE = 7

    print(navn)
    V7, N7, L7, minx7, maxx7 = Vfunc(V_0, b, omega, 2)
    print("Antall delinetvall N =", N7, "\nSkrittlengde =", Dx, "\nMolekylbredde =", L7, "nm")
    w7, psi7 = TUSL(Dx, N7, V7)
    printEvalues(w7, V_0, navn, 9, antallprint)
    print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N7, psi7, V7, Dx, navn, antallplot, 8e-4, 2.1e-1)
    plotPobabilityDensityWithV(N7, psi7, V7, Dx, navn, antallplot, 8e-5, 3.4e-3)
    plotEnergyLevels(N7, w7, V7, Dx, navn, antallplotE)
    # sitte parameter styrer antall desimaler i svar
    printBindingsenergiHe_2(w6, w7, 3)

    plt.show()

if krystall4a:
    # oppgave 4a, trenger data fra oppgave 2a!!!!
    # oppgave 2a
    navn = "Atom"
    Dx = 5e-12  # gir potensialbrønn i atom rundt 1 - 5 Å ssom er gjennomsnittlig atom radius
    omega = 50 * Dx  # lettes slik
    V_0 = 36 * spc.electron_volt
    antallprint = 6
    antallplotE = 4

    print(navn)
    V1, N1, L1, minx1, maxx1 = Vfunc(V_0, 0, omega, 1)
    print("Antall delinetvall N =", N1, "\nSkrittlengde =", Dx, "\nAtombredde =", L1, "nm")
    w1, psi1 = TUSL(Dx, N1, V1)
    printEvalues(w1, V_0, navn, 9, antallprint)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    plotEnergyLevels(N1, w1, V1, Dx, navn, antallplotE)
    # oppgave 2a
    # Dx, omega og V_0 gitt av atom!!! Oppgave 2a, kjøres automatisk på forhånd
    b = 15 * Dx  # = 0.75Å

    # N_omega = 5
    N_omega = 5
    navn = "Krystall N_omega = " + str(N_omega)
    antallplot = np.arange(4, step=3)
    antallprint = 17
    antallplotE = 17

    print(navn)
    V2, N2, L2, minx2, maxx2 = Vfunc(V_0, b, omega, N_omega)
    print("Antall delinetvall N =", N2, "\nSkrittlengde =", Dx, "\nKrystallbredde =", L2, "nm")
    w2, psi2 = TUSL(Dx, N2, V2)
    printEvalues(w2, V_0, navn, 9, antallprint)
    navn = "Krystall $N_{\omega}$ = " + str(N_omega)
    # print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N2, psi2, V2, Dx, navn, antallplot, 8e-4, 1.3e-1)
    plotPobabilityDensityWithV(N2, psi2, V2, Dx, navn, antallplot, 8e-5, 3.2e-3)
    plotEnergyLevels(N2, w2, V2, Dx, navn, antallplotE)

    # N_omega = 10
    N_omega = 10
    navn = "Krystall N_omega = " + str(N_omega)
    antallplot = np.arange(4, step=3)
    antallprint = 32
    antallplotE = 32

    print(navn)
    V3, N3, L3, minx3, maxx3 = Vfunc(V_0, b, omega, N_omega)
    print("Antall delinetvall N =", N3, "\nSkrittlengde =", Dx, "\nKrystallbredde =", L3, "nm")
    w3, psi3 = TUSL(Dx, N3, V3)
    printEvalues(w3, V_0, navn, 9, antallprint)
    navn = "Krystall $N_{\omega}$ = " + str(N_omega)
    # print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N3, psi3, V3, Dx, navn, antallplot, 8e-4, 1.1e-1)
    plotPobabilityDensityWithV(N3, psi3, V3, Dx, navn, antallplot, 8e-5, 3.2e-3)
    plotEnergyLevels(N3, w3, V3, Dx, navn, antallplotE)
    plt.show()

if krystall4b:
    # oppgave 4b, Tar kanskje 5 min!!
    # oppgave 4a, trenger data fra oppgave 2a!!!!
    # oppgave 2a
    Dx = 5e-12  # gir potensialbrønn i atom rundt 1 - 5 Å ssom er gjennomsnittlig atom radius
    omega = 50 * Dx  # lettes slik
    V_0 = 36 * spc.electron_volt
    b = 50 * Dx
    N_omega_list = np.arange(2, 101, step=1)
    B1_list = np.zeros(len(N_omega_list), dtype=np.float)
    B2_list = B1_list.copy()
    B3_list = B1_list.copy()
    print("Kjører løkke for beregninger, vent litt...")
    i = 0  # telle variabel
    for N_omega in N_omega_list:
        V4, N4, L4, minx4, maxx4 = Vfunc(V_0, b, omega, N_omega)
        w4, psi4 = TUSL(Dx, N4, V4)
        # print("N_omega =", N_omega) # viser at løkka går
        B1_list[i] = w4[N_omega - 1] - w4[0]
        B2_list[i] = w4[2*N_omega -1] - w4[N_omega]
        B3_list[i] = w4[3*N_omega -1] - w4[2*N_omega]
        i +=1
    print("Beregninger ferdig")
    plotEnergibandbreddensomFunkAvN_omega(B1_list, N_omega_list, 1)
    plotEnergibandbreddensomFunkAvN_omega(B2_list, N_omega_list, 2)
    plotEnergibandbreddensomFunkAvN_omega(B3_list, N_omega_list, 3)
    plt.show()

if krystall4cd or tidsutvikling:
    # oppgave 4cd
    # hvis 4b ikke kjørt
    Dx = 5e-12  # gir potensialbrønn i atom rundt 1 - 5 Å ssom er gjennomsnittlig atom radius
    omega = 50 * Dx  # = 2.5Å
    V_0 = 36 * spc.electron_volt
    b = 15 * Dx
    # N_omega = 10 = 0.75Å
    N_omega = 3
    navn = "Krystall N_omega = " + str(N_omega)
    antallplot = np.arange(4, step=3)
    antallprint = 302
    antallplotE = 302

    print(navn)
    V4, N4, L4, minx4, maxx4 = Vfunc(V_0, b, omega, N_omega)
    print("Antall delinetvall N =", N4, "\nSkrittlengde =", Dx, "\nKrystallbredde =", L4, "nm")
    w4, psi4 = TUSL(Dx, N4, V4)
    # Kjør alltid
    printEvalues(w4, V_0, navn, 9, antallprint)
    printHalvlederVsIsolator(w4, N_omega, 2)
    navn = "Krystall $N_{\omega}$ = " + str(N_omega)
    print("V sin periode mellom z =", round(minx4 * Dx * 1e9, 2),
    "nm og z =", round(maxx4 * Dx * 1e9, 2), " nm er ", round((omega + b) * 1e9, 2), "nm")
    print("Ploter nr.", antallplot)
    print("Ploter enrginivå 1 opp til ", antallplotE)
    # nest sitte tall skaler V, siste tall skyver V ned i plott, eks 1 skyver 1 enhet ned
    plotWaveFunctionsWithV(N4, psi4, V4, Dx, navn, antallplot, 8e-5, 2.8e-2)
    plotPobabilityDensityWithV(N4, psi4, V4, Dx, navn, antallplot, 8e-7, 0.5e-4)
    plotEnergyLevels(N4, w4, V4, Dx, navn, antallplotE)

    if tidsutvikling:
        # Lager starttilstanden Psi(x,0)
        p0 = np.sqrt(2 * spc.m_e * 36* spc.electron_volt)
        x0 = N4 * Dx / 2 # midten av intervall
        sigma = L4 * 1e-9 / 10  # leggde av brønn i m
        print(sigma)
        Ntid = 500
        Psi0 = makePsi0(p0, x0, sigma, N4, Dx)
        plotPsi0withV(N4, Psi0, V4, Dx,'Initiell sannsynlighetstetthet', 1e-8)
        print("Tidsutvikling, vent litt...")
        Psi_t_list, rhomax = makePsi_t_list(Psi0, psi4, w4, p0, N4, Dx, Ntid)
        print(rhomax)
        # rohmax trengs til animasjon.
        for i in range(Ntid):
            plotPsi0withV(N4, Psi_t_list[i], V4, Dx, 'Initiell sannsynlighetstetthet', 1e-8)

    plt.show()






