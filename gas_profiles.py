import numpy as np
from swiftsimio import load
from swiftsimio import mask
from velociraptor import load as load_catalogue
import unyt
import matplotlib.pyplot as plt

plt.style.use("mnras.mplstyle")

def make_paths(sim, 
               snap_id=36):
    catalogue_path = "../hydro/L0300N0564_VR18_"+sim+"/stf/snap_"+str(snap_id).zfill(4)+"/"
    catalogue_name = "snap_" + str(snap_id).zfill(4)
    snapshot_path = "../hydro/L0300N0564_VR18_"+sim+"/snapshots/snap_"+str(snap_id).zfill(4)+".hdf5"
    return catalogue_path, catalogue_name, snapshot_path


def get_redshift(sim, snap_id):
    _, _, snapshot_path = make_paths(sim, snap_id=snap_id)
    data = load(snapshot_path)
    redshift = data.metadata.z
    return redshift


def gas_profiles(sim, radii,
                 snap_id=36):

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    
    # General properties for profiles
    T_cut = 1e5
    R200crit = catalogue.radii.r_200crit[0] / catalogue.scale_factor
    M200crit = catalogue.masses.mass_200crit[0]

    N_bins = len(radii) - 1
    radii = radii * R200crit

    # Get cluster centre
    xc = catalogue.positions.xcmbp[0]
    yc = catalogue.positions.ycmbp[0]
    zc = catalogue.positions.zcmbp[0]
    centre = [xc, yc, zc] / catalogue.scale_factor * unyt.Mpc

    # Define region for swiftsimio to read in
    max_region = radii[-1]
    cluster_mask = mask(snapshot_path)
    region = [[centre[0] - max_region, centre[0] + max_region],
              [centre[1] - max_region, centre[1] + max_region],
              [centre[2] - max_region, centre[2] + max_region]]
    cluster_mask.constrain_spatial(region)

    # Load data
    data = load(snapshot_path, mask=cluster_mask)

    # Get cosmology
    Omega_m0 = data.metadata.cosmology.Om0
    Omega_de0 = data.metadata.cosmology.Ode0
    H0 = data.metadata.cosmology.H0
    a = data.metadata.a

    # Sort particle data
    data.gas.coordinates = data.gas.coordinates - centre
    dx = data.gas.coordinates[:,0] 
    dy = data.gas.coordinates[:,1]
    dz = data.gas.coordinates[:,2]
    particle_radii = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Calculate hot gas profiles
    density = np.zeros(N_bins)
    temperature = np.zeros(N_bins)
    for i in range(N_bins):
        hot_gas_mask = np.where((particle_radii > radii[i]) &
                                (particle_radii <= radii[i+1]) &
                                (data.gas.temperatures > T_cut))[0]

        volume = 4/3 * np.pi * (radii[i+1]**3 - radii[i]**3)
        bin_mass = np.sum(data.gas.masses[hot_gas_mask])
        temperature[i] = np.sum(data.gas.masses[hot_gas_mask]*data.gas.temperatures[hot_gas_mask]) / np.sum(data.gas.masses[hot_gas_mask])
        density[i] = bin_mass / volume

    #Calculate constants
    H = np.sqrt(H0**2 * (Omega_m0 * a**-3 + Omega_de0)).value * unyt.km / unyt.s / unyt.Mpc
    rho_crit = 3 * H**2 / (8*np.pi*unyt.G)
    mu_e = 1.14 #mean atomic weight per free electron
    mu = 0.59 #mean molecular weight
    fb = 0.24 #universal baryon fraction #######change
    T200 = unyt.G * M200crit * mu * unyt.mp / (2 * R200crit * a * unyt.kb)
    P200 = 500 * fb * unyt.kb * T200 * rho_crit / (mu * unyt.mp)
    K200 = unyt.kb * T200 / (500 * fb * (rho_crit / (mu_e * unyt.mp))**(2/3))

    temperature = temperature * unyt.K
    density = density * bin_mass.units / volume.units / 1e10
    pressure = density/(mu*unyt.mp) * unyt.kb * temperature
    entropy = unyt.kb * temperature / (density/(mu*unyt.mp))**(2/3)
    return density/rho_crit, temperature/T200, pressure/P200, entropy/K200


def get_avg_profiles(sim, radii, snap_id):
    N_bins = len(radii) - 1
 
    if snap_id > 34: #edge case
        snap_range = np.arange(34, 36+1) #range of snap_ids to average over
    else:
        snap_range = np.arange(snap_id-2, snap_id+2+1)
    N_snap = len(snap_range)
    
    densities = np.zeros(N_bins)
    temperatures = np.zeros(N_bins)
    pressures = np.zeros(N_bins)
    entropies = np.zeros(N_bins)
    z = np.array([])
    for snap in snap_range:
        rho, temp, pres, ent = gas_profiles(sim, radii, snap_id=snap)
        densities = densities + rho
        temperatures = temperatures + temp
        pressures = pressures + pres
        entropies = entropies + ent
        z = np.append(z, get_redshift(sim, snap))
    return densities/N_snap, temperatures/N_snap, pressures/N_snap, entropies/N_snap, z


def plot_compare_gas(comp_sim, *sims, snap_id=36):
    N_sims = len(sims)
    cm = plt.cm.viridis(np.linspace(0,1,N_sims))

    N_bins = 30
    log_radii = np.linspace(-1, np.log10(3), N_bins+1)
    radii = 10 ** log_radii
    rad_mid = 10**((log_radii[1:] + log_radii[:-1]) / 2)

    rho_ref, T_ref, P_ref, K_ref = get_gas_profiles(comp_sim, radii, snap_id=snap_id)

    fig, ax = plt.subplots(ncols=2, nrows=2,
                           gridspec_kw={'hspace' : 0.2, 'wspace' : 0.4})
    for i, sim in enumerate(sims):
        rho, T, P, K = get_gas_profiles(sim, radii, snap_id=snap_id)
        ax[0,0].semilogx(rad_mid, rho/rho_ref, 
                       color=cm[i],
                       label=sim)
        ax[0,1].semilogx(rad_mid, T/T_ref, color=cm[i])
        ax[1,0].semilogx(rad_mid, P/P_ref, color=cm[i])
        ax[1,1].semilogx(rad_mid, K/K_ref, color=cm[i])
    ax[0,0].set_ylabel("$\\rho / \\rho_{{\\rm{{{{{}}}}}}}$".format(comp_sim))
    ax[0,1].set_ylabel("$T/T_{{\\rm{{{{{}}}}}}}$".format(comp_sim))
    ax[1,0].set_ylabel("$P/P_{{\\rm{{{{{}}}}}}}$".format(comp_sim))
    ax[1,1].set_ylabel("$K/K_{{\\rm{{{{{}}}}}}}$".format(comp_sim))
    ax[0,0].legend()
    plt.text(0.45, 0.05, "$r/R_{\\rm{200c}}$", transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.15, left=0.17)
    plt.show()


def plot_all_gas(snap_id=36):
    sims = np.array([["R1", "R2", "R3", "R4"], 
                     ["M4", "K2s", "K3s", "R4"],
                     ["R1", "M2", "M3", "M4"]])
    cmR = plt.cm.winter(np.linspace(0,1,4))
    cmK = plt.cm.autumn(np.linspace(0,1,4))
    cmM = plt.cm.summer(np.linspace(0,1,4))

    N_bins = 30
    log_radii = np.linspace(-1, np.log10(3), N_bins+1)
    radii = 10 ** log_radii
    rad_mid = 10**((log_radii[1:] + log_radii[:-1]) / 2)

    fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(6,7),
                           sharex="col", sharey="row",
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})

    for col in range(3):
        if col == 0:
            cm = cmR
        elif col == 1:
            cm = cmK
        else:
            cm = cmM
        for sim in range(4):
            rho, T, P, K, z_range = get_avg_profiles(sims[col,sim], radii, snap_id=snap_id)
            ax[0,col].loglog(rad_mid, rho*rad_mid**2, color=cm[sim], label=sims[col,sim])
            ax[1,col].semilogx(rad_mid, T, color=cm[sim])
            ax[2,col].loglog(rad_mid, P*rad_mid**3, color=cm[sim])
            ax[3,col].loglog(rad_mid, K, color=cm[sim])
        ax[0,col].legend()
    ax[0,0].set_ylabel("$\\rho / \\rho_{\\rm{crit}} (r/R_{\\rm{200c}})^2$")
    ax[1,0].set_ylabel("$T/T_{200}$")
    ax[2,0].set_ylabel("$P/P_{200} (r/R_{\\rm{200c}})^3$")
    ax[3,0].set_ylabel("$K/K_{200}$")
    
    plt.suptitle(str(np.round(z_range[-1],2)) + "< z <" + str(np.round(z_range[0],2)))
    ax[3,1].set_xlabel("$r/R_{\\rm{200c}}$")
    plt.subplots_adjust(bottom=0.15, left=0.17, right=0.99)
    filename = "gas_profiles_z1.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_gas_properties(*sims, snap_id=36):
    N_sims = len(sims)
    cm = plt.cm.summer(np.linspace(0,1,N_sims))

    N_bins = 30
    log_radii = np.linspace(-1, np.log10(3), N_bins+1)
    radii = 10 ** log_radii
    rad_mid = 10**((log_radii[1:] + log_radii[:-1]) / 2)

    fig, ax = plt.subplots(ncols=2, nrows=2,
                           gridspec_kw={'hspace' : 0.2, 'wspace' : 0.4})
    for i, sim in enumerate(sims):
        rho, T, P, K, z_range = get_avg_profiles(sim, radii, snap_id=snap_id)
        ax[0,0].loglog(rad_mid, rho*rad_mid**2, 
                       color=cm[i],
                       label=sim)
        ax[0,1].semilogx(rad_mid, T, color=cm[i])
        ax[1,0].loglog(rad_mid, P*rad_mid**3, color=cm[i])
        ax[1,1].loglog(rad_mid, K, color=cm[i])
    ax[0,0].set_ylabel("$\\rho / \\rho_{\\rm{crit}} (r/R_{\\rm{200c}})^2$")
    ax[0,1].set_ylabel("$T/T_{200}$")
    ax[1,0].set_ylabel("$P/P_{200}$")
    ax[1,1].set_ylabel("$K/K_{200}$")
    ax[0,0].legend()
    plt.suptitle(str(np.round(z_range[-1],2)) + "< z <" + str(np.round(z_range[0],2)))
    plt.text(0.45, 0.05, "$r/R_{\\rm{200c}}$", transform=fig.transFigure)
    plt.subplots_adjust(bottom=0.15, left=0.17)
    filename = "M_series_gas_profiles.png"
    plt.savefig(filename, dpi=300)
    plt.show()

#plot_gas_properties("R1", "M2", "M3", "M4", snap_id=36)
#plot_compare_gas("R1", "R2", "R3", snap_id=36)
plot_all_gas(snap_id=25)
