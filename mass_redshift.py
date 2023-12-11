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
        snap = "/snapshots"
        catalogue_path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/stf/snap_"+str(snap_id).zfill(4)+"/"
        catalogue_name = "snap_" + str(snap_id).zfill(4)
        snapshot_path = "../"+sim_type+"/L0300N0564_VR18_"+sim+snap+"/snap_"+str(snap_id).zfill(4)+".hdf5"
    return catalogue_path, catalogue_name, snapshot_path


def get_redshift(sim, snap_id):
    _, _, snapshot_path = make_paths(sim, snap_id=snap_id)
    data = load(snapshot_path)
    redshift = data.metadata.z
    return redshift


def get_main_halo_index(sim, snap_id, sim_type="dmo", prev_halo_id=0):
    if snap_id == 36:
        return 0
    path, catalogue_name, _ = make_paths(sim, snap_id=snap_id+1, sim_type=sim_type)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    xc = catalogue.positions.xcmbp[prev_halo_id] / catalogue.scale_factor
    yc = catalogue.positions.ycmbp[prev_halo_id] / catalogue.scale_factor
    zc = catalogue.positions.zcmbp[prev_halo_id] / catalogue.scale_factor
    centre_z0 = [xc, yc, zc]

    path, catalogue_name, _ = make_paths(sim, snap_id=snap_id, sim_type=sim_type)
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


def get_M200(sim, snap_id, 
             sim_type="dmo", 
             mass_type="total"):
    if mass_type=="stellar" or mass_type=="BH":
        sim_type = "hydro" #double check
    path, catalogue_name, _ = make_paths(sim, snap_id=snap_id, sim_type=sim_type)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    mass_FOF = catalogue.masses.mass_fof
    M200crit = catalogue.masses.mass_200crit
    M200crit_star = catalogue.masses.mass_200crit_star

    if mass_type=="stellar":
        return M200crit_star
    elif mass_type=="BH":
        #h5file = h5.File(path+catalogue_name+".properties", 'r')
        #h5dset = h5file['/Aperture_SubgridMasses_aperture_total_bh_100_kpc']
        #M_BH = h5dset[...]
        #h5file.close()
        M_BH = get_BH_mass(sim, snap_id)
        return M_BH
    return M200crit


def get_BH_mass(sim, snap_id, halo_index=0):
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id, sim_type="hydro")
    catalogue = load_catalogue(path+catalogue_name+".properties")

    #if halo_index == 0:
    #    M200crit = catalogue.masses.mass_200crit
    #    halo_index = np.argmax(M200crit) #get accurate main halo index

    xc = catalogue.positions.xcmbp[halo_index] / catalogue.scale_factor
    yc = catalogue.positions.ycmbp[halo_index] / catalogue.scale_factor
    zc = catalogue.positions.zcmbp[halo_index] / catalogue.scale_factor
    centre = [xc, yc, zc] #add units
    #R200crit = catalogue.radii.r_200crit[halo_index]

    h5file = h5.File(path+catalogue_name+".properties", 'r')
    h5dset = h5file['/SO_R_500_rhocrit']
    R500crit = h5dset[...]
    h5file.close()
    R500crit = unyt.unyt_quantity(R500crit[halo_index], "Mpc") / catalogue.scale_factor

    cluster_mask = mask(snapshot_path)
    region = [[centre[0] - R500crit, centre[0] + R500crit],
              [centre[1] - R500crit, centre[1] + R500crit],
              [centre[2] - R500crit, centre[2] + R500crit]]
    cluster_mask.constrain_spatial(region)

    # Load data
    data = load(snapshot_path, mask=cluster_mask)

    data.black_holes.coordinates = data.black_holes.coordinates - centre
    dx = data.black_holes.coordinates[:,0] 
    dy = data.black_holes.coordinates[:,1]
    dz = data.black_holes.coordinates[:,2]
    radii = np.sqrt(dx**2 + dy**2 + dz**2)
    #aperture_cut = unyt.unyt_quantity(50, "kpc") * catalogue.scale_factor #comoving kpc
    radial_mask = np.where(radii**2 < R500crit**2)[0]
    #total_mass = np.sum(data.black_holes.dynamical_masses[radial_mask])
    total_mass = np.sum(data.black_holes.subgrid_masses[radial_mask])
    return total_mass


def get_centres(sim, snap_id):
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    xc = catalogue.positions.xcmbp[0] / catalogue.scale_factor
    yc = catalogue.positions.ycmbp[0] / catalogue.scale_factor
    zc = catalogue.positions.zcmbp[0] / catalogue.scale_factor
    return xc, yc, zc


def plot_centre_movement(*sims):
    cm = plt.cm.viridis(np.linspace(0,1,len(sims)))
    snap_begin = 10
    snap_range = np.arange(snap_begin, snap_max)
    plt.figure()
    for i, sim in enumerate(sims):
        xc = np.zeros(len(snap_range))
        yc = np.zeros(len(snap_range))
        zc = np.zeros(len(snap_range))
        redshifts = np.zeros(len(snap_range))
        for j,snap_id in enumerate(snap_range):
            xc[j], yc[j], zc[j] = get_centres(sim, snap_id)
            redshifts[j] = get_redshift(sim, snap_id)
        plt.plot(redshifts, xc,
                 label=sim,
                 color=cm[i])
        #plt.plot(redshifts, yc,
        #         color=cm[i], linestyle="--")
        #plt.plot(redshifts, zc,
        #         color=cm[i], linestyle=":")
    plt.legend()
    plt.xlabel("$z$")
    plt.ylabel("$x_{\\rm{centre}}$ (Mpc)") #currently plots in kpc not Mpc
    plt.show()


def calc_and_plot(sim_series, ax, sim_type="hydro", mass_type="stellar"):
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
    if mass_type == "BH":
        snap_end = 12
    snap_range = np.arange(snap_max, snap_end, -1)

    halo_id = 0
    for i, sim in enumerate(sims):
        masses = np.zeros(len(snap_range))
        redshifts = np.zeros(len(snap_range))
        for j,snap_id in enumerate(snap_range):
            halo_id = get_main_halo_index(sim, snap_id, sim_type=sim_type, prev_halo_id=halo_id)
            #if mass_type != "BH":
            halo_mass = get_M200(sim, snap_id, mass_type=mass_type)
            masses[j] = halo_mass#[halo_id]
            #else:
            #    masses[j] = get_BH_mass(sim, snap_id, halo_index=halo_id)
            redshifts[j] = get_redshift(sim, snap_id)
        ax.semilogy(redshifts, masses*1e10,
                     label=sim,
                     color=cm[i])
    ax.legend()


def plot_mass_change_all(mass_type="stellar", **kwargs):
    snap_end = 10
    if mass_type == "M200":
        ylabel = "$M_{\\rm{200c}}(z) / \\rm{M_{\odot}}$"
    elif mass_type == "stellar":
        ylabel = "$M_{\star}(z) / \\rm{M_{\odot}}$"
    elif mass_type == "BH":
        ylabel = "$M_{\\rm{BH}}(z) / \\rm{M_{\odot}}$"

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(3,6),
                             sharex=True, sharey=True,
                             gridspec_kw = {'hspace': 0, 'wspace': 0})

    calc_and_plot("R", axes[0], mass_type=mass_type, **kwargs)
    calc_and_plot("K", axes[1], mass_type=mass_type, **kwargs) 
    calc_and_plot("M", axes[2], mass_type=mass_type, **kwargs) 
    axes[2].set_xlabel("$z$")
    axes[1].set_ylabel(ylabel)
    plt.subplots_adjust(left=0.15)
    filename = mass_type+"_sub_mass_500c.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_mass_change(*sims, sim_type="dmo", mass_type="M200"):
    snap_end = 10
    if mass_type == "M200":
        ylabel = "$M_{\\rm{200c}}(z) / \\rm{M_{\odot}}$"
    elif mass_type == "stellar":
        stellar = True
        ylabel = "$M_{\star}(z) / \\rm{M_{\odot}}$"
    elif mass_type == "BH":
        snap_end = 12
        ylabel = "$M_{\\rm{BH}}(z) / \\rm{M_{\odot}}$"
    snap_range = np.arange(snap_max, snap_end, -1)

    cm = plt.cm.winter(np.linspace(0,1,len(sims)))
    plt.figure()
    halo_id = 0
    for i, sim in enumerate(sims):
        masses = np.zeros(len(snap_range))
        redshifts = np.zeros(len(snap_range))
        for j,snap_id in enumerate(snap_range):
            halo_id = get_main_halo_index(sim, snap_id, sim_type=sim_type, prev_halo_id=halo_id)
            if mass_type != "BH":
                halo_mass = get_M200(sim, snap_id, stellar=stellar)
                masses[j] = halo_mass[halo_id]
            else:
                masses[j] = get_BH_mass(sim, snap_id, halo_index=halo_id)
            redshifts[j] = get_redshift(sim, snap_id)
        plt.semilogy(redshifts, masses*1e10,
                     label=sim,
                     color=cm[i])
    plt.legend()
    plt.xlabel("$z$")
    plt.ylabel(ylabel)
    filename = "R_series_stellar_mass.png"
    plt.savefig(filename, dpi=300)
    plt.show()

#plot_mass_change("R1", "R2", mass_type="stellar")
plot_mass_change_all(mass_type="BH")
#plot_centre_movement("R1")
#get_M200("M3", 35, prev_halo_id=0)


