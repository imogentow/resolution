import numpy as np
from swiftsimio import load
from velociraptor import load as load_catalogue
import unyt
import matplotlib.pyplot as plt

plt.style.use("mnras.mplstyle")

subhalo_cuts = {"K1" : 2e9, 
                "K2" : 6e8,
                "K3" : 3e8,
                "M3" : 1e10,
                "M4" : 6e9,
                "K2s" : 2e9} #R-series all 0

def make_paths(sim, 
               snap_id=36):
    catalogue_path = "../hydro/L0300N0564_VR18_"+sim+"/stf/snap_"+str(snap_id).zfill(4)+"/"
    catalogue_name = "snap_" + str(snap_id).zfill(4)
    snapshot_path = "../hydro/L0300N0564_VR18_"+sim+"/snapshots/snap_"+str(snap_id).zfill(4)+".hdf5"
    return catalogue_path, catalogue_name, snapshot_path


def get_masses(sim, snap_id=36):

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")

    # Get R200c value to use as boundary for subhalos
    R200crit = catalogue.radii.r_200crit[0]

    # Get centres of halos 
    xc = catalogue.positions.xcmbp
    yc = catalogue.positions.ycmbp
    zc = catalogue.positions.zcmbp

    # Calculate distance from main halo
    dx_squared = (xc[1:] - xc[0])**2
    dy_squared = (yc[1:] - yc[0])**2
    dz_squared = (zc[1:] - zc[0])**2
    dR = np.sqrt(dx_squared + dy_squared + dz_squared)

    radius_mask = np.where(dR < R200crit)[0]
    radius_mask = np.append(0, radius_mask) #add main halo to list

    # Get masses
    masses = catalogue.apertures.mass_star_100_kpc[radius_mask]
    tot_mass = catalogue.masses.mass_200crit_star[0]
    return masses, tot_mass


def get_subhalo_masses(sim, snap_id=36):
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    
    R200crit = catalogue.radii.r_200crit[0]

    # Get centres of halos 
    xc = catalogue.positions.xcmbp
    yc = catalogue.positions.ycmbp
    zc = catalogue.positions.zcmbp

    # Calculate distance from main halo
    dx_squared = (xc[1:] - xc[0])**2
    dy_squared = (yc[1:] - yc[0])**2
    dz_squared = (zc[1:] - zc[0])**2
    dR = np.sqrt(dx_squared + dy_squared + dz_squared)

    radius_mask = np.where(dR < R200crit)[0]
    stellar_masses = catalogue.apertures.mass_star_50_kpc[1:][radius_mask]*1
    subhalo_masses = catalogue.masses.mass_tot[1:][radius_mask]*1

    return subhalo_masses, stellar_masses


def calc_and_plot_cum_sum(sim_series, ax, snap_id=36):
    if sim_series == "R":
        sims = ["R1", "R2", "R3", "R4"]
        cm = plt.cm.winter(np.linspace(0,1,len(sims)))
    elif sim_series == "K":
        sims = ["M4", "K2s", "K3s", "R4"]
        cm = plt.cm.autumn(np.linspace(0,1,len(sims)))
    elif sim_series == "M":
        sims = ["R1", "M2", "M3", "M4"]
        cm = plt.cm.summer(np.linspace(0,1,len(sims)))

    for i, sim in enumerate(sims):
        masses, tot_mass = get_masses(sim, snap_id=snap_id)
        sort_masses = np.flip(np.sort(masses))
        sort_masses = sort_masses[np.where(sort_masses > 0)[0]]
        cumulative_mass = np.cumsum(sort_masses)
        N_halos = len(sort_masses)
        halo_range = np.arange(1, N_halos+1, 1)
        ax.loglog(cumulative_mass, halo_range, color=cm[i], label=sim)
    ax.legend()


def calc_and_plot_sub_mass(sim_series, ax, snap_id=36):
    if sim_series == "R":
        sims = ["R4", "R3", "R2", "R1"]
        cm = plt.cm.winter_r(np.linspace(0,1,len(sims)))
    elif sim_series == "K":
        sims = ["R4", "K3s", "K2s", "M4"]
        cm = plt.cm.autumn_r(np.linspace(0,1,len(sims)))
    elif sim_series == "M":
        sims = ["M4", "M3", "M2", "R1"]
        cm = plt.cm.summer_r(np.linspace(0,1,len(sims)))

    s = 5
    alpha = 1
    lw = 1

    for i, sim in enumerate(sims):
        subs, stellar = get_subhalo_masses(sim, snap_id=snap_id)
        #sort mass cuts
        if sim in subhalo_cuts:
            cut = unyt.unyt_quantity(subhalo_cuts[sim], 'Msun')
        else:
            cut = unyt.unyt_quantity(0, 'Msun')
        safe_halos = np.where((subs > cut) & (stellar > 0))[0]
        frag_halos = np.where((subs < cut) & (stellar > 0))[0]
        ax.scatter(subs[safe_halos], stellar[safe_halos], color=cm[i],
                   marker=".", s=s, alpha=alpha,
                   label=sim)
        ax.scatter(subs[frag_halos], stellar[frag_halos], color=cm[i],
                   marker="x", s=s, alpha=alpha, linewidth=lw)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()


def cumulative_masses(snap_id=36):
    fig, ax = plt.subplots(nrows=3, ncols=1, 
                             figsize=(3,6),
                             sharex=True, 
                             sharey=True,
                             gridspec_kw = {'hspace': 0, 'wspace': 0})
    calc_and_plot_cum_sum("R", ax[0], snap_id=snap_id)
    calc_and_plot_cum_sum("K", ax[1], snap_id=snap_id)
    calc_and_plot_cum_sum("M", ax[2], snap_id=snap_id)
    ax[1].set_ylabel("$N_{\\rm{halos}}$")
    ax[2].set_xlabel("$M_{\\rm{cumulative}} / M_{\\rm{200c},\star}$")
    plt.legend()
    plt.subplots_adjust(left=0.2, right=0.99)
    filename = "plots/cumulative_stellar_mass.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def subhalo_masses(snap_id=36):
    fig, ax = plt.subplots(nrows=3, ncols=1, 
                             figsize=(3,6),
                             sharex=True, 
                             sharey=True,
                             gridspec_kw = {'hspace': 0, 'wspace': 0})
    calc_and_plot_sub_mass("R", ax[0], snap_id=snap_id)
    calc_and_plot_sub_mass("K", ax[1], snap_id=snap_id)
    calc_and_plot_sub_mass("M", ax[2], snap_id=snap_id)
    ax[1].set_ylabel("$M_{\star, \\rm{50 kpc}}$")
    ax[2].set_xlabel("$M_{\\rm{200c}}$")
    plt.subplots_adjust(left=0.2, right=0.99)
    filename = "plots/subhalo_stellar_mass.png"
    plt.savefig(filename, dpi=300)
    plt.show()

cumulative_masses(snap_id=34)
#subhalo_masses()
