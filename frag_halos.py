import numpy as np
from swiftsimio import load
from velociraptor import load as load_catalogue
from velociraptor.tools import create_mass_function
import unyt
import matplotlib.pyplot as plt


def make_paths(sim, 
               snap_id=36, 
               sim_type="hydro"):
    if sim_type == "dmo":
        path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/"
        if sim == "K2s" or sim == "K3s":
            sim = "R4" #incorrectly labelled files, just need to change the path name
        catalogue_path = path + "stf/snap_"+sim+"_"+str(snap_id).zfill(4)+"/"
        catalogue_name = "snap_"+sim+"_" + str(snap_id).zfill(4)
        snapshot_path = path + "snapshots/snap_"+sim+"_"+str(snap_id).zfill(4)+".hdf5"
    else:
        catalogue_path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/stf/snap_"+str(snap_id).zfill(4)+"/"
        catalogue_name = "snap_" + str(snap_id).zfill(4)
        snapshot_path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/snapshots/snap_"+str(snap_id).zfill(4)+".hdf5"
    return catalogue_path, catalogue_name, snapshot_path


def subhalo_properties(sim, **kwargs):
    path, catalogue_name, snapshot_path = make_paths(sim, **kwargs)
    catalogue = load_catalogue(path+catalogue_name+".properties")

    N_particles_factor = 100 
    
    data = load(snapshot_path)
    DM_mass = data.dark_matter.masses
    min_mass = N_particles_factor * DM_mass[0]/5
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

    mass_tot = catalogue.masses.mass_tot[1:]
    mass_50 = catalogue.apertures.mass_50_kpc[1:]
    mass_star = catalogue.apertures.mass_star_50_kpc[1:]
    mass_gas = catalogue.apertures.mass_gas_50_kpc[1:]

    subhalo_cut = np.where((mass_tot > min_mass) & (dR < R200crit))[0]
    return mass_tot[subhalo_cut], mass_50[subhalo_cut], mass_star[subhalo_cut], mass_gas[subhalo_cut], dR[subhalo_cut]


def plot_subhalos(*sims, **kwargs):
    
    fig, ax = plt.subplots(nrows=1, ncols=len(sims), figsize=(len(sims)*3.5, 4))
    fig2, ax2 = plt.subplots(nrows=1, ncols=len(sims), figsize=(len(sims)*3.5, 4))
    for i, sim in enumerate(sims):
        mass_tot, mass_50, mass_star, mass_gas, dR = subhalo_properties(sim, **kwargs)
        mass_tot = mass_tot * 1 #unyt converts to correct units this way
        mass_star = mass_star * 1
        mass_gas = mass_gas * 1
        mass_50 = mass_50 * 1
        
        fraction = mass_star / mass_tot
        high_mask = np.where(fraction > 0.4)[0]

        N_sub = len(mass_tot)
        N_star = np.where(mass_star == 0)[0]
        N_gas = np.where(mass_gas == 0)[0]
        N_both = np.where((mass_star == 0) & (mass_gas == 0))[0]
        N_neither = np.where((mass_star > 0) & (mass_gas > 0))[0]

        hist_no_star, edges_no_star = np.histogram(np.log10(mass_tot[N_star]), bins=20)#, density=True)
        hist_no_gas, edges_no_gas = np.histogram(np.log10(mass_tot[N_gas]), bins=20)#, density=True)
        hist_all, edges_all = np.histogram(np.log10(mass_tot), bins=20)#, density=True)
        hist_both, edges_both = np.histogram(np.log10(mass_tot[N_both]), bins=20)#, density=True)
        hist_neither, edges_neither = np.histogram(np.log10(mass_tot[N_neither]), bins=20)#, density=True)

        ax2[i].plot(edges_all[1:], hist_all, color="k", label="All halos")
        ax2[i].plot(edges_no_star[1:], hist_no_star, color="gold", label="No stars")
        ax2[i].plot(edges_no_gas[1:], hist_no_gas, color="c", label="No gas")
        ax2[i].plot(edges_both[1:], hist_both, color="green", label="No gas or stars")
        ax2[i].plot(edges_neither[1:], hist_neither, color="r", label="Both gas and stars")
        ax2[i].set_title(sim)

        ax[i].semilogx(mass_tot, mass_star/mass_tot, "*", color="gold", label="Stars")
        ax[i].semilogx(mass_tot, mass_gas/mass_tot, 'o', color="c", label="Gas")
        ax[i].set_title(sim)

    ax[0].legend()
    ax[0].set_ylabel("$f$")
    fig.text(0.45, 0.02, "$M_{\\rm{tot}}$", transform=fig.transFigure)
    ax2[0].legend()
    plt.show()

sim = "R4"
plot_subhalos("R4", "K3s", "M4")

