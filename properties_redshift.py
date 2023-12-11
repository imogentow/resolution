import numpy as np
from swiftsimio import load
from swiftsimio import mask
from velociraptor import load as load_catalogue
import unyt
import matplotlib.pyplot as plt
import h5py as h5

snap_max = 36

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


def get_main_halo_index(sim, snap_id, prev_halo_id=0):
    if snap_id == 36:
        return 0
    path, catalogue_name, _ = make_paths(sim, snap_id=snap_id+1)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    xc = catalogue.positions.xcmbp[prev_halo_id] / catalogue.scale_factor
    yc = catalogue.positions.ycmbp[prev_halo_id] / catalogue.scale_factor
    zc = catalogue.positions.zcmbp[prev_halo_id] / catalogue.scale_factor
    centre_z0 = [xc, yc, zc]

    path, catalogue_name, _ = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    xc = catalogue.positions.xcmbp / catalogue.scale_factor
    yc = catalogue.positions.ycmbp / catalogue.scale_factor
    zc = catalogue.positions.zcmbp / catalogue.scale_factor
    mass_FOF = catalogue.masses.mass_fof

    dx = xc - centre_z0[0]
    dy = yc - centre_z0[1]
    dz = zc - centre_z0[2]
    dR = np.sqrt(dx**2 + dy**2 + dz**2)
    radial_cut = unyt.unyt_quantity(4000, 'kpc')
    if snap_id <= 15:
        radial_cut = unyt.unyt_quantity(8000, 'kpc')
    halo_restrictions = np.where(dR < radial_cut)[0]
    indexes = np.arange(0,len(mass_FOF))[halo_restrictions]
    max_mass = np.argmax(mass_FOF[halo_restrictions])
    halo_index = indexes[max_mass]
    return halo_index


def get_gas_properties(sim, snap_id):
    T_cut = 1e5 #temperature cut for hot and cold gas

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    
    # General properties for profiles
    #R200crit = catalogue.radii.r_200crit[0] / catalogue.scale_factor
    #M200crit = catalogue.masses.mass_200crit[0]
    #gas_mass = catalogue.masses.mass_gas_500c[0] #giving -ve or 0
    #stellar_mass = catalogue.masses.mass_star_500c[0] #also fives -ve or 0

    h5file = h5.File(path+catalogue_name+".properties", 'r')
    h5dset = h5file['/SO_R_500_rhocrit']
    R500crit = h5dset[...][0]
    h5dset = h5file['/SO_Mass_500_rhocrit']
    M500crit = h5dset[...][0]
    h5dset = h5file['/SO_Mass_gas_500_rhocrit']
    gas_mass = h5dset[...][0]
    h5dset = h5file['/SO_Mass_star_500_rhocrit']
    stellar_mass = h5dset[...][0]
    h5file.close()

    R500crit = unyt.unyt_quantity(R500crit, 'Mpc') / catalogue.scale_factor
    M500crit = unyt.unyt_quantity(M500crit*1e10, 'Msun')
    gas_mass = unyt.unyt_quantity(gas_mass*1e10, 'Msun')
    stellar_mass = unyt.unyt_quantity(stellar_mass*1e10, 'Msun')

    baryon_mass = gas_mass + stellar_mass
    baryon_fraction = baryon_mass / M500crit

    stellar_fraction = stellar_mass / M500crit

    # Get cluster centre
    xc = catalogue.positions.xcmbp[0]
    yc = catalogue.positions.ycmbp[0]
    zc = catalogue.positions.zcmbp[0]
    centre = [xc, yc, zc] / catalogue.scale_factor * unyt.Mpc

    # Define region for swiftsimio to read in
    max_region = R500crit * 1.1
    cluster_mask = mask(snapshot_path)
    region = [[centre[0] - max_region, centre[0] + max_region],
              [centre[1] - max_region, centre[1] + max_region],
              [centre[2] - max_region, centre[2] + max_region]]
    cluster_mask.constrain_spatial(region)

    # Load data
    data = load(snapshot_path, mask=cluster_mask)

    # Sort particle data
    data.gas.coordinates = data.gas.coordinates - centre
    dx = data.gas.coordinates[:,0] 
    dy = data.gas.coordinates[:,1]
    dz = data.gas.coordinates[:,2]
    gas_radii = np.sqrt(dx**2 + dy**2 + dz**2)

    temperature_cut = np.where((data.gas.temperatures > T_cut) & (gas_radii < R500crit))[0]
    hot_gas_mass = np.sum(data.gas.masses[temperature_cut])
    cold_gas_mass = gas_mass - hot_gas_mass
    hot_gas_fraction = hot_gas_mass / M500crit
    cold_gas_fraction = cold_gas_mass / M500crit

    return baryon_fraction, hot_gas_fraction, cold_gas_fraction, stellar_fraction


def calc_and_plot(sim_series, ax):
    if sim_series == "R":
        sims = ["R1", "R2", "R3", "R4"]
        cm = plt.cm.winter(np.linspace(0,1,len(sims)))
    elif sim_series == "K":
        sims = ["M4", "K2s", "K3s", "R4"]
        cm = plt.cm.autumn(np.linspace(0,1,len(sims)))
    elif sim_series == "M":
        sims = ["R1", "M2", "M3", "M4"]
        cm = plt.cm.summer(np.linspace(0,1,len(sims)))

    snap_end = 10
    snap_range = np.arange(snap_max, snap_end, -1)

    halo_id = 0
    for i, sim in enumerate(sims):
        f_baryon = np.zeros(len(snap_range))
        f_hot = np.zeros(len(snap_range))
        f_cold = np.zeros(len(snap_range))
        f_star = np.zeros(len(snap_range))
        redshifts = np.zeros(len(snap_range))
        for j,snap_id in enumerate(snap_range):
            halo_id = get_main_halo_index(sim, snap_id, prev_halo_id=halo_id)
            f_baryon[j], f_hot[j], f_cold[j], f_star[j] = get_gas_properties(sim, snap_id)
            redshifts[j] = get_redshift(sim, snap_id)
        ax[0].plot(redshifts, f_baryon,
                     label=sim,
                     color=cm[i])
        ax[1].plot(redshifts, f_hot,
                  color=cm[i],
                  label=sim)
        ax[2].plot(redshifts, f_cold,
                  color=cm[i],
                  label=sim)
        ax[3].plot(redshifts, f_star,
                  color=cm[i],
                  label=sim)
    ax[0].legend()


def plot_mass_change_all(**kwargs):
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(5.5,6),
                             sharex=True, sharey="row",
                             gridspec_kw = {'hspace': 0, 'wspace': 0})

    calc_and_plot("R", axes[:,0], **kwargs)
    calc_and_plot("K", axes[:,1], **kwargs) 
    calc_and_plot("M", axes[:,2],  **kwargs)

    axes[3,1].set_xlabel("$z$")

    axes[0,0].set_ylabel("$f_{\\rm{baryon, 500c}}$")
    axes[1,0].set_ylabel("$f_{\\rm{hot, 500c}}$")
    axes[2,0].set_ylabel("$f_{\\rm{cold, 500c}}$")
    axes[3,0].set_ylabel("$f_{\star,\\rm{500c}}$")

    fig.subplots_adjust(left=0.15, right=0.99)
    plt.savefig("fractions.png", dpi=300)
    plt.show()


plot_mass_change_all()
#get_gas_properties("R3", 36)
