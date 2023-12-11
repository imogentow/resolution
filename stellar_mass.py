import numpy as np
from swiftsimio import load
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


def get_masses(sim, snap_id=36):

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    
    data = load(snapshot_path)
    DM_mass = data.dark_matter.masses

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
    masses = catalogue.apertures.mass_star_50_kpc[radius_mask]
    tot_mass = catalogue.masses.mass_200crit_star[0]
    return masses, tot_mass


def calc_and_plot(sim_series, ax, snap_id=36):
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
        ax.loglog(cumulative_mass/tot_mass, halo_range, color=cm[i], label=sim)
    ax.legend()


def cumulative_masses(snap_id=36):
    fig, ax = plt.subplots(nrows=3, ncols=1, 
                             figsize=(3,6),
                             sharex=True, 
                             sharey=True,
                             gridspec_kw = {'hspace': 0, 'wspace': 0})
    calc_and_plot("R", ax[0], snap_id=snap_id)
    calc_and_plot("K", ax[1], snap_id=snap_id)
    calc_and_plot("M", ax[2], snap_id=snap_id)
    ax[1].set_ylabel("$N_{\\rm{halos}}$")
    ax[2].set_xlabel("$M_{\\rm{cumulative}} / M_{\\rm{200c},\star}$")
    plt.legend()
    plt.subplots_adjust(left=0.2, right=0.99)
    filename = "plots/cumulative_stellar_mass.png"
    plt.savefig(filename, dpi=300)
    plt.show()

cumulative_masses()
