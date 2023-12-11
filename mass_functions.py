import numpy as np
from swiftsimio import load
from velociraptor import load as load_catalogue
from velociraptor.tools import create_mass_function
import unyt
import matplotlib.pyplot as plt
import h5py as h5

plt.style.use("mnras.mplstyle")

subhalo_cuts = {"K1" : 2e9, 
                "K2" : 6e8,
                "K3" : 3e8,
                "M3" : 1e10,
                "M4" : 6e9,
                "K2s" : 2e9} #R-series all 0

def make_paths(sim, 
               snap_id=36, 
               sim_type="dmo"):
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


def get_redshift(sim, 
                 snap_id):
    _, _, snapshot_path = make_paths(sim, snap_id=snap_id)
    data = load(snapshot_path)
    redshift = data.metadata.z
    return redshift


def get_mass_function(sim, n_bins=10, 
                      return_bin_edges=False,
                      edges="none",
                      snap_id=36,
                      sim_type="dmo",
                      stellar=False,
                      black_hole=False):
    
    if sim_type == "dmo":
        N_particles_factor = 40
    else:
        N_particles_factor = 40 * 2 #particles are lower mass in hydro due to splitting

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id, 
                                                     sim_type=sim_type)
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

    radius_mask = np.where(dR < 2*R200crit)[0]

    # Get masses
    subhalo_masses = catalogue.masses.mass_tot[1:][radius_mask]
    if stellar:
        masses = catalogue.apertures.mass_star_50_kpc[1:][radius_mask]
        if sim in subhalo_cuts:
            cut = unyt.unyt_quantity(subhalo_cuts[sim], 'Msun')
            subhalo_mask = np.where(subhalo_masses > cut)[0]
            masses = masses[subhalo_mask]
        N_particles_factor /= 16 #stellar mass of a halo will be much less than total halo mass - equivalent of 5 DM particles
    elif black_hole:
        N_particles_factor /= 100
        h5file = h5.File(path+catalogue_name+".properties", 'r')
        h5dset = h5file['/Aperture_SubgridMasses_aperture_total_bh_50_kpc']
        M_BH = h5dset[...]
        h5file.close()
        masses = M_BH[1:][radius_mask] * 1e10
        masses = unyt.unyt_array(masses, "Msun")
        if sim in subhalo_cuts:
            cut = unyt.unyt_quantity(subhalo_cuts[sim], 'Msun')
            subhalo_mask = np.where(subhalo_masses > cut)[0]
            masses = masses[subhalo_mask]
    else:
        masses = subhalo_masses
    masses.convert_to_units('Msun')
    box_volume = catalogue.units.comoving_box_volume

    if edges == "none":
        # Upper and lower bounds of mass function
        lowest_mass = N_particles_factor * DM_mass[0]
        highest_mass = np.sort(masses)[-2]
    else:
        lowest_mass = edges[0]
        highest_mass = edges[-1]
        n_bins = len(edges) - 1

    if lowest_mass > highest_mass:
        a = np.zeros(n_bins)
        a[:] = np.nan
        return a, a, a

    return create_mass_function(
        masses=masses,
        lowest_mass=lowest_mass,
        highest_mass=highest_mass,
        box_volume=box_volume,
        n_bins=n_bins,
        return_bin_edges=return_bin_edges)


def plot_mf_ratio(list_to_compare, 
                  list_of_sims,
                  sim_type="dmo",
                  stellar=False,
                  snap_id=36):
    if stellar:
        sim_type="hydro"
        mass_label = "$M_{\star\\rm{,50kpc}} [\\rm{M}_{\odot}]$"
        mf_label = "$d N(M_{\star\\rm{,50kpc}}) / d \log_{10} M_{\star\\rm{,50kpc}}$"
    else:
        mass_label = "$M_{\\rm{tot}} [\\rm{M}_{\odot}]$"
        mf_label = "$d N(M_{\\rm{tot}}) / d \log_{10} M_{\\rm{tot}}$"

    if sim_type == "dmo":
        ref_sim = "R5"
    else:
        ref_sim = "R4"

    N_sims = len(list_of_sims)
    cm = plt.cm.autumn(np.linspace(0,1,N_sims))

    if len(list_to_compare) != N_sims:
        print("List of comparison cluster must be same length as list of standard clusters.")
        return

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(3.3,5))
    for i in range(N_sims):
        centres, mf, _, bin_edges = get_mass_function(list_of_sims[i], return_bin_edges=True, snap_id=snap_id, stellar=stellar, sim_type=sim_type)
        centres_comparison, mf_comparison, _ = get_mass_function(list_to_compare[i], edges=bin_edges, snap_id=snap_id, stellar=stellar, sim_type=sim_type)

        if len(mf_comparison) > len(mf):
            mf_comparison = mf_comparison[:len(mf)] #match length of standard mass function
        elif len(mf_comparison) < len(mf):
            mf = mf[:len(mf_comparison)] #match length of standard mass function
            centres = centres[:len(centres_comparison)]
        ax[0].loglog(centres, mf, 
                       label=list_of_sims[i], color=cm[i])
        ax[1].semilogx(centres, mf/mf_comparison,
                       label=list_of_sims[i]+ "/" + list_to_compare[i],
                       color=cm[i])

    centres, mf, _ = get_mass_function(ref_sim, snap_id=snap_id, stellar=stellar, sim_type=sim_type)
    ax[0].loglog(centres, mf, label=ref_sim, color="grey")

    if snap_id != 36:
        z = get_redshift(list_of_sims[0], snap_id)
        plt.title("$z=$"+str(np.round(z,2)))
    ax[1].set_xlabel(mass_label)
    ax[0].set_ylabel("$MF = $" + mf_label)
    ax[1].set_ylabel("$MF_{\\rm{K}} / MF_{\\rm{R}}$")
    ax[0].legend()
    ax[1].legend()
    plt.subplots_adjust(left=0.2)
    filename = "compare_MF_K_series.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_mf_redshifts(sim, 
                      *snap_ids):
    mass_label = "$M_{\\rm{tot}} [\\rm{M}_{\odot}]$"
    mf_label = "$d N(M_{\\rm{tot}}) / d \log_{10} M_{\\rm{tot}}$"
    cm = plt.cm.viridis(np.linspace(0,1,len(snap_ids)))

    for i, snap_id in enumerate(snap_ids):
        centres, mf, _ = get_mass_function(sim, snap_id=snap_id)
        z = get_redshift(sim, snap_id)
        plt.loglog(centres, mf,
                   label="z="+str(np.round(z,2)),
                   colour=cm[i])
    plt.xlabel(mass_label)
    plt.ylabel(mf_label)
    plt.legend()
    plt.show()


def compare_dmo_hydro(*sims, 
                      snap_id=36):
    mass_label = "$M_{\\rm{tot}} [\\rm{M}_{\odot}]$"
    mf_label = "$d N(M_{\\rm{tot}}) / d \log_{10} M_{\\rm{tot}}$"
    cm = plt.cm.viridis(np.linspace(0,1,len(sims)))

    for i, sim in enumerate(sims):
        centres_dmo, mf_dmo, _ = get_mass_function(sim, snap_id=snap_id,
                                                   sim_type="dmo")
        centres_hydro, mf_hydro, _ = get_mass_function(sim, snap_id=snap_id,
                                                       sim_type="hydro")
        plt.loglog(centres_dmo, mf_dmo,
                   label=sim, color=cm[i])
        plt.loglog(centres_hydro, mf_hydro,
                   color=cm[i],
                   linestyle="--", label="Hydro")
    plt.xlabel(mass_label)
    plt.ylabel(mf_label)
    plt.legend()
    plt.show()


def calc_and_plot(sim_series, ax, 
                  snap_id=36,
                  sim_type="hydro", 
                  stellar=False,
                  black_hole=False):
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
        centres, mf, _ = get_mass_function(sim, 
                                           snap_id=snap_id, 
                                           stellar=stellar,
                                           black_hole=black_hole,
                                           sim_type=sim_type)
        #mf in kpc^-3 units - maybe use Mpc^-3 instead?
        ax.loglog(centres, mf,
                  label=sim, color=cm[i])
    ax.legend()


def plot_all_series(stellar=False, black_hole=False, sim_type="hydro"):
    if stellar:
        sim_type="hydro"
        mass_label = "$M_{\star\\rm{,50kpc}} / \\rm{M}_{\odot}$"
        mf_label = "$d N(M_{\star\\rm{,50kpc}}) / d \log_{10} M_{\star\\rm{,50kpc}}$"
    elif black_hole:
        sim_type="hydro"
        mass_label = "$M_{\\rm{BH,50kpc}} / \\rm{M}_{\odot}$"
        mf_label = "$d N(M_{\\rm{BH,50kpc}}) / d \log_{10} M_{\\rm{BH,50kpc}}$"
    else:
        mass_label = "$M_{\\rm{tot}} [\\rm{M}_{\odot}]$"
        mf_label = "$d N(M_{\\rm{tot}}) / d \log_{10} M_{\\rm{tot}}$"

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3.3, 7),
                             sharex=True, sharey=True,
                             gridspec_kw={'hspace' : 0, 'wspace' : 0})
    calc_and_plot("R", axes[0], stellar=stellar, black_hole=black_hole, sim_type=sim_type)
    calc_and_plot("K", axes[1], stellar=stellar, black_hole=black_hole, sim_type=sim_type)
    calc_and_plot("M", axes[2], stellar=stellar, black_hole=black_hole, sim_type=sim_type)
    axes[2].set_xlabel(mass_label)
    axes[1].set_ylabel(mf_label)
    plt.subplots_adjust(left=0.15, right=0.99)
    filename = "BH_mass_function.png"
    # plt.savefig(filename, dpi=300)
    plt.show()


def plot_mass_function(*sims, 
                       snap_id=36, 
                       stellar=False, 
                       sim_type="dmo"):
    """
    Plots mass function using VR catalogue properties.
    """
    if stellar:
        sim_type="hydro"
        mass_label = "$M_{\star\\rm{,50kpc}} / \\rm{M}_{\odot}$"
        mf_label = "$d N(M_{\star\\rm{,50kpc}}) / d \log_{10} M_{\star\\rm{,50kpc}}$"
    else:
        mass_label = "$M_{\\rm{tot}} [\\rm{M}_{\odot}]$"
        mf_label = "$d N(M_{\\rm{tot}}) / d \log_{10} M_{\\rm{tot}}$"
    cm = plt.cm.spring(np.linspace(0,1,len(sims)))

    plt.figure(figsize=(5,5))
    for i, sim in enumerate(sims):
        centres, mf, _ = get_mass_function(sim, 
                                           snap_id=snap_id, 
                                           stellar=stellar,
                                           sim_type=sim_type)
        plt.loglog(centres, mf,
                   label=sim, color=cm[i])
    plt.xlabel(mass_label)
    plt.ylabel(mf_label)
    plt.legend()
    plt.subplots_adjust(left=0.1)
    filename = "MF_R_series.png"
    plt.savefig(filename, dpi=300)
    plt.show()

#plot_mf_ratio(["R4","R4", "R4"], ["M4", "K2s", "K3s"], sim_type="hydro")
#plot_mf_ratio(["R4","R3", "R2"], ["M4", "M3", "M2"], sim_type="hydro")
#plot_mf_ratio(["R4", "R4", "R4"], ["M4", "K2s", "K3s"], snap_id=36)#, sim_type="hydro")
#plot_mass_function("R1", "R2", "R3", "R4", "R5")
#plot_mass_function("M4", "K3s", "R4", stellar=True)
#plot_mf_redshifts("R3", 36, 30, 25, 21, 18, 14)
#compare_dmo_hydro("R1", "R2", "R3")
plot_all_series(stellar=True)
