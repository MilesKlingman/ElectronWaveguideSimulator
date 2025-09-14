import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.special import hermite
from matplotlib.widgets import Slider, TextBox

# Constants
hbar = 1.0545718e-34
e = 1.60217662e-19
mu_B = 9.2740100657e-24
k_B = 1.380649e-23
h = 6.62607015e-34
me = 9.10938356e-31

paramsA = {
        'm_x': 1.9 * me, # Device A
        'm_y' : 1.9 * me, # Device A
        'm_z': 6.5 * me, # Device A
        'l_y': 26e-9, # Device A
        'l_z': 8.1e-9, # Device A
        'g' : 0.6, 
        }

paramsB = {
        'm_x': 1.8 * me, # Device B
        'm_y' : 1.8 * me, # Device B
        'm_z': 6.4 * me, # Device B
        'l_y': 27e-9, # Device B
        'l_z': 7.9e-9, # Device B
        'g' : 0.6, 
        }

class System:

    m_x = None
    m_y = None
    m_z = None
    l_y = None
    l_z = None
    g = None
    omega_y = None
    omega_z = None

    #------------------------------------------------------------------------------#
    # Energy spectra calulation

    def __init__(self, params):
        # Physical parameters
        self.m_x = params['m_x'] # Effective masses
        self.m_y = params['m_y']
        self.m_z = params['m_z']
        self.l_y = params['l_y'] # Confinement lengths
        self.l_z = params['l_z']
        self.g = params['g'] # g-factor
        # Calculate derived quantities
        self.omega_y = hbar / (self.m_y * self.l_y**2)
        self.omega_z = hbar / (self.m_z * self.l_z**2)

    def energy(self, n_y, n_z, s, k_x, B):
        """Calculate subband energy"""
        omega_c = e * B / self.m_y
        Omega = np.sqrt(self.omega_y**2 + omega_c**2)
        E_y = hbar * Omega * (n_y + 0.5)
        E_z = hbar * self.omega_z * (2*n_z + 1.5)
        E_spin = -self.g * mu_B * B * s
        E_x = (hbar**2 * k_x**2) / (2*self.m_x) * (1 - omega_c**2/Omega**2)
        return E_y + E_z + E_spin + E_x
#--------------------------------------------------------------------------------#
# Wavefunction generators

class WavefunctionGenerator:

    system = None

    def __init__(self, system):
        self.system = system

    def psi_2d(self, y_grid, z_grid, n_y, n_z, k_x, B):
        """Calculate 2D transverse wavefunction"""
        # Magnetic displacement
        omega_c = e * B / self.system.m_y
        Omega = np.sqrt(self.system.omega_y**2 + omega_c**2)
        y0 = hbar * k_x * omega_c / (self.system.m_y * Omega**2)
        # Length scales
        l_y = np.sqrt(hbar / (self.system.m_y * Omega))
        l_z = np.sqrt(hbar / (self.system.m_z * self.system.omega_z))
        # Scaled coordinates
        xi_y = (y_grid - y0) / l_y
        xi_z = z_grid / l_z
        # Hermite polynomials (use scipy.special)
        H_y = hermite(n_y)(xi_y)
        H_z = hermite(2*n_z + 1)(xi_z)
        # Gaussian envelopes
        gauss_y = np.exp(-xi_y**2 / 2)
        gauss_z = np.exp(-xi_z**2 / 2)
        # Normalization
        norm_y = (self.system.m_y * Omega / (np.pi * hbar))**0.25
        norm_y /= np.sqrt(2**n_y * factorial(n_y))
        norm_z = (self.system.m_z * self.system.omega_z /
        (np.pi * hbar))**0.25
        norm_z *= 2 / np.sqrt(2**(2*n_z+1) * factorial(2*n_z+1))
        return norm_y * norm_z * gauss_y * H_y * gauss_z * H_z

#--------------------------------------------------------------------------------#
# Transport (conductance) calulation

class TransportCalculator:

    system = None
    max_n_y = None
    max_n_z = None

    def __init__(self, system, max_n_y=10, max_n_z=5):
        self.system = system
        self.max_n_y = max_n_y
        self.max_n_z = max_n_z

    def find_subbands(self, B, E_max):
        """Find all subbands below E_max"""
        subbands = []
        for n_y in range(self.max_n_y):
            for n_z in range(self.max_n_z):
                for s in [-0.5, 0.5]:
                    E = self.system.energy(n_y, n_z, s, 0, B)
                    if E < E_max:
                        subbands.append({
                            'n_y': n_y, 'n_z': n_z, 's': s,
                            'E': E, 'label': f"|{n_y},{n_z},{'+' if s>0 else '-'}>"
                        })
        return sorted(subbands, key=lambda x: x['E'])

    def conductance(self, mu, B, T):
        """Calculate conductance at given chemical potential"""
        if T == 0:
            # Zero temperature: sharp steps
            subbands = self.find_subbands(B, mu)
            return len(subbands) * e**2 / h
        else:
            # Finite temperature: thermal broadening
            subbands = self.find_subbands(B, mu + 10*k_B*T)
            G = 0
            for sb in subbands:
                f = 1 / (1 + np.exp((sb['E'] - mu) / (k_B * T)))
                G += f * e**2 / h
        return G
    
#-------------------------------------------------------------------------------------------------#
# Energy spectra calculation

system = None
paramsChoice = ''
params = None

while paramsChoice not in ['A', 'B']:
    paramsChoice = input("Choose device parameters for energy calculation (A or B): ").strip().upper()
    if paramsChoice == 'A':
        params = paramsA
    elif paramsChoice == 'B':
        params = paramsB
    else:
        print("Invalid choice. Please enter 'A' or 'B'.")

system = System(params)

def energy_spectrum_plot(system: System):
    B = 4
    k_vals = np.linspace(-2e8, 2e8, 300)
    state = [0, 0, -1/2]

    fig, ax = plt.subplots(figsize= (8, 5))
    plt.subplots_adjust(bottom = 0.25)

    E_vals = [system.energy(state[0], state[1], state[2], k, B) for k in k_vals]
    line, = ax.plot(k_vals * 1e-8, E_vals)
    ax.set_xlabel("Kx (×10^8 m⁻¹)")
    ax.set_ylabel("Energy")
    ax.set_title(f"Energy vs kx at B = {B} T")
    ax.set_ylim(0, 10e-23)

    ax_b = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_B = Slider(ax_b, 'B (T)', 0.0, 10.0, valinit = B)
    TextBox_ny = plt.axes([0.1, 0.025, 0.1, 0.04])
    text_box_ny = TextBox(TextBox_ny, 'n_y', initial = str(state[0]))
    TextBox_nz = plt.axes([0.3, 0.025, 0.1, 0.04])
    text_box_nz = TextBox(TextBox_nz, 'n_z', initial = str(state[1]))
    TextBox_s = plt.axes([0.5, 0.025, 0.1, 0.04])
    text_box_s = TextBox(TextBox_s, 's (+/- 0.5)', initial = str(state[2]))

    def update(val):
        B = slider_B.val
        state[0] = int(text_box_ny.text)
        state[1] = int(text_box_nz.text)
        state[2] = float(text_box_s.text)
        new_vals = [system.energy(state[0], state[1], state[2], k, B) for k in k_vals]
        line.set_ydata(new_vals)
        fig.canvas.draw_idle()
        # print(system.energy(state[0], state[1], state[2], 0, B))
    
    slider_B.on_changed(update)
    text_box_ny.on_submit(update)
    text_box_nz.on_submit(update)
    text_box_s.on_submit(update)

    plt.show()

energy_spectrum_plot(system)

#---------------------------------------------------------------------------------#
# Wavefunction generator

paramsChoice = ''
while paramsChoice not in ['A', 'B']:
    paramsChoice = input("Choose device parameters for wavefunction calculation (A or B): ").strip().upper()
    if paramsChoice == 'A':
        params = paramsA
    elif paramsChoice == 'B':
        params = paramsB
    else:
        print("Invalid choice. Please enter 'A' or 'B'.")

system2 = System(params)
systemWaveFunc = WavefunctionGenerator(system2)

def wavefunction_plot(system: WavefunctionGenerator):
    B = 4
    kx = 1e8

    y = np.linspace(-100e-9, 100e-9, 500)
    z = np.linspace(-100e-9, 100e-9, 500)
    Y, Z = np.meshgrid(y, z, indexing='ij')
    state = [1, 0]  # n_y, n_z
    psi = system.psi_2d(Y, Z, n_y=state[0], n_z=state[1], k_x = kx, B=B)

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom = 0.25)
    contour = ax.contourf(y * 1e9, z * 1e9, np.abs(psi)**2, levels=100, cmap='RdBu_r')
    ax.set_xlabel('y (nm)')
    ax.set_ylabel('z (nm)')
    ax.set_title(r'$|\phi(y, z)|^2$')
    cb = fig.colorbar(contour, ax=ax, label='Probability Density')

    ax_b = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider_B = Slider(ax_b, 'B (T)', 0.0, 10.0, valinit = B)
    TextBox_ny = plt.axes([0.1, 0.025, 0.1, 0.04])
    text_box_ny = TextBox(TextBox_ny, 'n_y', initial = str(state[0]))
    TextBox_nz = plt.axes([0.3, 0.025, 0.1, 0.04])
    text_box_nz = TextBox(TextBox_nz, 'n_z', initial = str(state[1])) 
    TextBox_kx = plt.axes([0.5, 0.025, 0.1, 0.04])
    text_box_kx = TextBox(TextBox_kx, 'kx value', initial = kx)

    def update(val):
        nonlocal contour, cb
        B = slider_B.val
        state[0] = int(text_box_ny.text)
        state[1] = int(text_box_nz.text)
        # allow updating kx from its textbox (fallback to outer kx)
        try:
            kx_val = float(text_box_kx.text)
        except Exception:
            kx_val = kx
        newpsi = system.psi_2d(Y, Z, n_y=state[0], n_z=state[1], k_x=kx_val, B=B)
        # clear main axes and redraw contour + labels + colorbar
        ax.clear()
        contour = ax.contourf(y * 1e9, z * 1e9, np.abs(newpsi)**2, levels=100, cmap='RdBu_r')
        ax.set_xlabel('y (nm)')
        ax.set_ylabel('z (nm)')
        ax.set_title(r'$|\phi(y, z)|^2$')
        # update existing colorbar to point to the new contour mappable
        try:
            cb.mappable = contour
            cb.update_normal(contour)
        except Exception:
            # fallback: create a new colorbar if update fails
            try:
                cb = fig.colorbar(contour, ax=ax, label='Probability Density')
            except Exception:
                pass
        fig.canvas.draw_idle()
    
    slider_B.on_changed(update)
    text_box_ny.on_submit(update)
    text_box_nz.on_submit(update)
    text_box_kx.on_submit(update)

    plt.show()

wavefunction_plot(systemWaveFunc)
#---------------------------------------------------------------------------------#
# Conductance


def conductance_calculation():
    paramsChoice = ''
    while paramsChoice not in ['A', 'B']:
        paramsChoice = input("Choose device parameters for conductance calculation (A or B): ").strip().upper()
        if paramsChoice == 'A':
            params = paramsA
        elif paramsChoice == 'B':
            params = paramsB
        else:
            print("Invalid choice. Please enter 'A' or 'B'.")

    system3 = System(params)
    transport = TransportCalculator(system3, max_n_y=10, max_n_z=5)

    E_max = float(input("Enter maximum energy E_max (in eV) for subband search: ")) * e

    B_check = 4
    subbands = transport.find_subbands(B=B_check, E_max=E_max)
    if not subbands:
        print(f"No subbands found for B={B_check} T and E_max={E_max}.")
    else:
        print(f"Subbands (B={B_check} T, E_max={E_max}):")
        print(f"{'#':>3} {'Label':>8} {'n_y':>4} {'n_z':>4} {'s':>5} {'E (eV)':>14}")
        for i, sb in enumerate(subbands, start=1):
            E_eV = sb['E'] / e
            print(f"{i:3d} {sb['label']:>8} {sb['n_y']:4d} {sb['n_z']:4d} {sb['s']:5.1f} {E_eV:14.10e}")

    mu = float(input("Enter the chemical potential mu (in eV) for conductance calculation: ")) * e
    T = float(input("Enter the temperature T (in Kelvin) for conductance calculation: "))
    B = float(input("Enter the magnetic field B (in Tesla) for conductance calculation: "))

    G = transport.conductance(mu, B, T)
    print(f"Conductance G at mu={mu/e} eV, B={B} T, T={T} K: {G:.4e} S: {G/(e**2/h):.2f} (in units of e^2/h)")

conductance_calculation()

#---------------------------------------------------------------------------------#
# Fig 2
def fig2ACE(calc, tag):
    T = 50e-3
    mu_grid_meV = np.linspace(0, 1.2, 300)
    mu_grid_J = mu_grid_meV * 1e-3 * e 
    mu_max_meV = mu_grid_meV.max()  

    plt.figure(figsize=(6, 4.5), dpi=140)

    for B in np.arange(0.0, 9.0 + 0.5, 0.5):
        G_list = []
        for mu in mu_grid_J:
            G_list.append(calc.conductance(mu, B, T) / (e**2 / h))
        plt.plot(mu_grid_meV, G_list, linewidth=0.8, alpha=0.95, color='k')

    plt.xlim(0, mu_max_meV)
    plt.ylim(0, None)
    plt.xlabel(r'$\mu$ (meV)')
    plt.ylabel(r'$G$ ($e^2/h$)')
    plt.title(tag)
    plt.tight_layout()
    plt.show()

def fig2BDF(calc, tag):
    T = 50e-3
    mu_grid_meV = np.linspace(0, 3, 300)
    mu_grid_J = mu_grid_meV * 1e-3 * e 
    mu_max_meV = mu_grid_meV.max()

    MU = mu_grid_J[None, :]
    GG = []
    for B in np.linspace(-9.0, 9.0, 181):
        G_list = []
        for mu in mu_grid_J:
            G_list.append(calc.conductance(mu, B, T) / (e**2 / h))
        GG.append(G_list)

    GG = np.array(GG)
    dmu_J = np.gradient(mu_grid_J)
    dGdmu = np.gradient(GG, axis=1) / dmu_J
    dGdmu_meV = dGdmu * (1e-3 * e)

    plt.figure(figsize=(6, 4.5), dpi=140)
    extent = [mu_grid_meV.min(), mu_grid_meV.max(), -9.0, 9.0]
    plt.imshow(dGdmu_meV, aspect='auto', origin='lower', extent=extent)
    plt.xlabel(r'$\mu$ (meV)')
    plt.ylabel(r'$B$ (T)')
    plt.title(tag)
    plt.tight_layout()
    plt.show()

calcA = TransportCalculator(System(paramsA), max_n_y=10, max_n_z=5)
calcB = TransportCalculator(System(paramsB), max_n_y=10, max_n_z=5)

fig2ACE(calcA, 'DeviceA G_vs_mu 0-9T')
fig2BDF(calcA, 'DeviceA dGdmu_map')
fig2ACE(calcB, 'DeviceB G_vs_mu 0-9T')
fig2BDF(calcB, 'DeviceB dGdmu_map')
fig2ACE(calcA, 'Theory G_vs_mu 0-9T')
fig2BDF(calcA, 'Theory dGdmu_map')


#---------------------------------------------------------------------------------#
# Fig 3
def figure_3A():
    calc = TransportCalculator(System(paramsA))

    states = [
        (2, 1),
        (1, 1),
        (0, 1),
        (2, 0),
        (1, 0),
        (0, 0),
    ]

    B = np.linspace(-9, 9, 300)
    
    plt.figure(figsize=(8, 6))

    for n_y, n_z in states:
        E_vals = []
        for B_val in B:
            E_vals.append(calc.system.energy(n_y, n_z, 0.5, k_x = 0, B = B_val) / e * 1e3)
        
        plt.plot(B, np.array(E_vals) / e * 1e3, label=f'$|{n_y},{n_z},\\uparrow\\rangle$')
        plt.plot(B, E_vals)

    plt.xlim(B.min(), B.max())
    plt.ylim(0, 1.2)
    plt.xlabel('Magnetic Field B (T)')
    plt.ylabel('Energy (meV)')
    plt.title('Figure 3A – Subband Energies vs Magnetic Field at kx=0')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Create the system for Device A
def figure_3B():
    systemA = System(paramsA)

    B_fixed = 4.0  

    y_vals = np.linspace(0, 120e-9, 300)

    states = [
        (2, 1),
        (1, 1),
        (0, 1),
        (2, 0),
        (1, 0),
        (0, 0)
    ]

    plt.figure(figsize=(5, 4))

    for (n_y, n_z) in states:
        E_vals = []
        
        omega_c = e * B_fixed / systemA.m_y
        Omega = np.sqrt(systemA.omega_y**2 + omega_c**2)
        y0 = 0.0
        
        E_base = systemA.energy(n_y, n_z, 0.5, k_x=0, B=B_fixed)
        
        for y in y_vals:
            V_y = 0.5 * systemA.m_y * Omega**2 * (y - y0)**2
            E_vals.append((E_base + V_y) / e * 1e3)

        plt.plot(y_vals * 1e9, E_vals)

    plt.xlabel(r'$y$ (nm)')
    plt.ylabel(r'$E$ (meV)')
    plt.xlim(0, 120)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.show()

def figure_3C():
    wavefunc = WavefunctionGenerator(System(paramsA))

    y = np.linspace(-100e-9, 100e-9, 400)
    z = np.linspace(-40e-9, 0, 200)

    states = [
        (2, 1),
        (1, 1),
        (0, 1),
        (2, 0),
        (1, 0),
        (0, 0),
    ]

    labels = [r'$|2,1,\uparrow\rangle$', r'$|1,1,\uparrow\rangle$', r'$|0,1,\uparrow\rangle$',
            r'$|2,0,\uparrow\rangle$', r'$|1,0,\uparrow\rangle$', r'$|0,0,\uparrow\rangle$']

    fig, axes = plt.subplots(len(states), 1, figsize=(15, 10), sharex=True, sharey=True)

    Y, Z = np.meshgrid(y, z, indexing='ij')

    for ax, (n_y, n_z), label in zip(axes, states, labels):
        psi = wavefunc.psi_2d(Y, Z, n_y=n_y, n_z=n_z, k_x=0, B=4)
        im = ax.imshow(psi.T, extent=[y[0]*1e9, y[-1]*1e9, z[0]*1e9, z[-1]*1e9],
                    aspect='auto', cmap='RdBu_r', origin='lower', vmin=-np.max(np.abs(psi)), vmax=np.max(np.abs(psi)))
        ax.set_ylabel('z (nm)', fontsize=10)
        ax.text(-120, 0, label, va='center', ha='right', fontsize=12)

    axes[-1].set_xlabel('y (nm)', fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Re[ϕ(y, z)]', shrink=0.8)
    fig.suptitle(r'Figure 3C – Real Part of $\phi_{n_y,n_z,\uparrow}(y, z)$ at $B = 4$ T', fontsize=14)
    plt.show()

figure_3A()
figure_3B()
figure_3C()