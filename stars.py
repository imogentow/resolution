import numpy as np
from swiftsimio import load
from swiftsimio import mask
from velociraptor import load as load_catalogue
import matplotlib.pyplot as plt
import unyt
from scipy import spatial
from datetime import datetime
start = datetime.now()

plt.style.use("mnras.mplstyle")


def make_paths(sim, 
               snap_id=36):
    path = "../hydro/L0300N0564_VR18_"+sim+"/stf/snap_"+str(snap_id).zfill(4)+"/"
    catalogue_name = "snap_" + str(snap_id).zfill(4)
    snapshot_path = "../hydro/L0300N0564_VR18_"+sim+"/snapshots/snap_"+str(snap_id).zfill(4)+".hdf5"
    return path, catalogue_name, snapshot_path


def get_all_star_ages(sim, bin_edges, snap_id=36):
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=36)
    catalogue = load_catalogue(path+catalogue_name+".properties")

    xc = catalogue.positions.xcmbp[0] / catalogue.scale_factor
    yc = catalogue.positions.ycmbp[0] / catalogue.scale_factor
    zc = catalogue.positions.zcmbp[0] / catalogue.scale_factor
    centre = [xc, yc, zc]
    R200crit = catalogue.radii.r_200crit[0]

    cluster_mask = mask(snapshot_path)
    region = [[centre[0] - R200crit, centre[0] + R200crit],
              [centre[1] - R200crit, centre[1] + R200crit],
              [centre[2] - R200crit, centre[2] + R200crit]]
    cluster_mask.constrain_spatial(region)

    # Load data
    data = load(snapshot_path, mask=cluster_mask)

    # Don't need to restrict by location?
    #data.stars.coordinates = data.stars.coordinates - centre
    #dx = data.stars.coordinates[:,0] 
    #dy = data.stars.coordinates[:,1]
    #dz = data.stars.coordinates[:,2]
    #radii = np.sqrt(dx**2 + dy**2 + dz**2)
    #radial_mask = np.where(radii**2 < R200crit**2)[0]

    star_ages = data.stars.birth_scale_factors
    hist, _ = np.histogram(star_ages, bins=bin_edges, density=True)
    return hist


def get_subhalo_stellar_ages(sim, snap_id=36):
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=36)
    catalogue = load_catalogue(path+catalogue_name+".properties")

    # Get R200c value to use as boundary for subhalos
    R200crit = catalogue.radii.r_200crit[0]
    masses = catalogue.masses.mass_tot[1:]
    stellar_masses = catalogue.apertures.mass_star_50_kpc[1:]

    xc = catalogue.positions.xcmbp / catalogue.scale_factor
    yc = catalogue.positions.ycmbp / catalogue.scale_factor
    zc = catalogue.positions.zcmbp / catalogue.scale_factor
    halo_centre = [xc[0], yc[0], zc[0]]

    # Calculate distance from main halo
    dx = xc[1:] - xc[0]
    dy = yc[1:] - yc[0]
    dz = zc[1:] - zc[0]
    dR = np.sqrt(dx**2 + dy**2 + dz**2)

    data = load(snapshot_path)
    DM_mass = data.dark_matter.masses[0]
    min_mass = 100 * DM_mass/5 #approx mass of 100 baryonic particles
    aperture_size = unyt.unyt_quantity(50, 'kpc') / catalogue.scale_factor
    data.stars.coordinates = data.stars.coordinates - halo_centre

    radial_mask = np.where((dR < R200crit) & (masses >= min_mass))[0]
    average_ages = np.zeros(len(radial_mask))
    for i,j in enumerate(radial_mask):
        sub_halo_centre = [dx[j], dy[j], dz[j]]
        star_particle_coords = data.stars.coordinates - sub_halo_centre
        star_radii = np.sqrt(star_particle_coords[:,0]**2 + star_particle_coords[:,1]**2 +
                             star_particle_coords[:,2]**2)
        star_mask = np.where(star_radii < aperture_size)[0]
        average_ages[i] = np.nanmean(data.stars.birth_scale_factors[star_mask])

    return average_ages, masses[radial_mask]#, stellar_masses[radial_mask]


def test_subhalo_ages(sim, snap_id=36): ##slower than before :( get rid?
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=36)
    catalogue = load_catalogue(path+catalogue_name+".properties")

    R200crit = catalogue.radii.r_200crit[0]
    masses = catalogue.masses.mass_tot[1:]

    xc = catalogue.positions.xcmbp / catalogue.scale_factor
    yc = catalogue.positions.ycmbp / catalogue.scale_factor
    zc = catalogue.positions.zcmbp / catalogue.scale_factor
    halo_centre = [xc[0], yc[0], zc[0]]

    # Calculate distance from main halo
    dx = xc[1:] - xc[0]
    dy = yc[1:] - yc[0]
    dz = zc[1:] - zc[0]
    dR = np.sqrt(dx**2 + dy**2 + dz**2)
    subhalo_centres = np.vstack((dx,dy,dz)).T
    subhalo_centres = unyt.unyt_array(subhalo_centres, 'kpc')
    subhalo_centres.convert_to_units('Mpc')

    data = load(snapshot_path)
    DM_mass = data.dark_matter.masses[0]
    min_mass = 100 * DM_mass/5 #approx mass of 100 baryonic particles
    aperture_size = unyt.unyt_quantity(50, 'kpc') / catalogue.scale_factor
    data.stars.coordinates = data.stars.coordinates - halo_centre
    
    radial_mask = np.where((dR < R200crit) & (masses >= min_mass))[0]
    subhalo_cenres = subhalo_centres[radial_mask]
    N_subs = len(radial_mask)
    average_ages = np.zeros(N_subs)

    distance, index = spatial.KDTree(subhalo_centres).query(data.stars.coordinates)
    aperture_cut = np.where(distance <= 0.05)[0] #only take star particles within 50kpc of a subhalo
    index = index[aperture_cut]
    ages = data.stars.birth_scale_factors[aperture_cut]
    print(ages)
    for i in range(N_subs):
        subhalo_cut = np.where(index == i)[0]
        average_ages[i] = np.nanmean(ages[subhalo_cut])
    return average_ages, masses[radial_mask]


def plot_all_ages():
    sims = np.array([["R1", "R2", "R3", "R4"], 
                     ["M4", "K2s", "K3s", "R4"],
                     ["R1", "M2", "M3", "M4"]])
    cmR = plt.cm.winter(np.linspace(0,1,4))
    cmK = plt.cm.autumn(np.linspace(0,1,4))
    cmM = plt.cm.summer(np.linspace(0,1,4))

    N_bins = 50
    bins = np.linspace(0,1,N_bins)
    bin_mids = (bins[1:] + bins[:-1]) / 2

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(3.3, 7),
                           sharex=True, gridspec_kw={'hspace' : 0, 'wspace' : 0})
    for row in range(3):
        if row == 0:
            cm = cmR
        elif row == 1:
            cm = cmK
        else:
            cm = cmM
        for sim in range(4):
            ages = get_all_star_ages(sims[row,sim], bins)
            ax[row].plot(bin_mids, ages, color=cm[sim], label=sims[row,sim])
        ax[row].legend()
    ax[2].set_xlabel("$a$")
    ax[1].set_ylabel("$N$")#change
    filename = "stellar_ages.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_star_ages(*sims, snap_id=36):
    N_bins = 50
    bins = np.linspace(0,1,N_bins)
    bin_mids = (bins[1:] + bins[:-1]) / 2

    cm = plt.cm.viridis(np.linspace(0,1,len(sims)))
    plt.figure()
    for i, sim in enumerate(sims):
        ages = get_star_age_hist(sim, bins)
        plt.plot(bin_mids, ages, color=cm[i], label=sim)
    plt.legend()
    plt.xlabel("$a$")
    plt.ylabel("")
    plt.show()

def plot_subhalo_stellar_ages(*sims, snap_id=36):
    N_sims = len(sims)
    fig, ax = plt.subplots(nrows=1, ncols=N_sims, figsize=(N_sims*2.5, 3), sharey=True)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    for i, sim in enumerate(sims):
        avg_ages, masses = get_subhalo_stellar_ages(sim, snap_id=snap_id)
        print(avg_ages)
        age_hist, edges = np.histogram(avg_ages[np.isfinite(avg_ages)], bins=20)
        print(edges)
        ax[i].semilogx(masses, avg_ages, marker="o", linewidth=0)
        ax[i].set_title(sim)
        ax2.plot(edges[1:], age_hist/len(avg_ages), label=sim)
    ax2.set_xlabel("$a$")
    ax2.set_ylabel("$N_{\\rm{sub}} / N_{\\rm{sub,tot}}$")
    ax2.legend()
    print(datetime.now()-start)
    plt.show()


#plot_star_ages("R1", "R2", "R3")
#plot_subhalo_stellar_ages("R3", "M3")
#avg_ages = test_subhalo_ages("R2")
plot_all_ages()
