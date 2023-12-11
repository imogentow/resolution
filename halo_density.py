import numpy as np
from swiftsimio import load
from swiftsimio import mask
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
import unyt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.style.use("mnras.mplstyle")

def make_paths(sim, snap_id=36, sim_type="dmo"):
    path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/stf/snap_"+sim+"_"+str(snap_id).zfill(4)+"/"
    catalogue_name = "snap_"+sim+"_" + str(snap_id).zfill(4)
    snapshot_path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/snapshots/snap_"+sim+"_"+str(snap_id).zfill(4)+".hdf5"
    return path, catalogue_name, snapshot_path


def get_redshift(sim, snap_id):
    _, _, snapshot_path = make_paths(sim, snap_id=snap_id)
    data = load(snapshot_path)
    redshift = data.metadata.z
    return redshift


def halo_density_profile(sim, 
                         conv="none",
                         snap_id=36):

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    R200crit = catalogue.radii.r_200crit[0] / catalogue.scale_factor

    #Define range for density bins
    N_bins = 30
    if conv == "none":
        radii = np.logspace(-1, np.log10(3), N_bins+1) * R200crit
    else:
        radii = np.logspace(np.log10(conv), np.log10(3), N_bins+1) * R200crit

    # Get cluster centre
    xc = catalogue.positions.xcmbp[0]
    yc = catalogue.positions.ycmbp[0]
    zc = catalogue.positions.zcmbp[0]
    centre = [xc, yc, zc] / catalogue.scale_factor *unyt.Mpc

    # Define region for swiftsimio to read in
    max_region = radii[-1]
    cluster_mask = mask(snapshot_path)
    region = [[centre[0] - max_region, centre[0] + max_region],
              [centre[1] - max_region, centre[1] + max_region],
              [centre[2] - max_region, centre[2] + max_region]]
    cluster_mask.constrain_spatial(region)

    # Load data
    data = load(snapshot_path, mask=cluster_mask)

    # Sort particle data
    data.dark_matter.coordinates = data.dark_matter.coordinates - centre
    dx = data.dark_matter.coordinates[:,0] 
    dy = data.dark_matter.coordinates[:,1]
    dz = data.dark_matter.coordinates[:,2]
    DM_radii = np.sqrt(dx**2 + dy**2 + dz**2)
    DM_mass = data.dark_matter.masses

    # Caclulate critical density
    Omega_m0 = data.metadata.cosmology.Om0
    Omega_de0 = data.metadata.cosmology.Ode0
    H0 = data.metadata.cosmology.H0
    a = data.metadata.a
    H = np.sqrt(H0**2 * (Omega_m0 * a**-3 + Omega_de0)).value * unyt.km / unyt.s / unyt.Mpc
    rho_crit = 3 * H**2 / (8*np.pi*unyt.G)
    
    # Calculate density
    DM_density = np.zeros(N_bins)
    tot_mass = 0
    tot_vol = 0
    for i in range(N_bins):
        DM_mask = np.where((DM_radii > radii[i]) & (DM_radii <= radii[i+1]))[0]
        volume = 4/3 * np.pi * (radii[i+1]**3 - radii[i]**3)
        bin_mass = np.sum(DM_mass[DM_mask])
        bin_mass.convert_to_units('Msun')
        DM_density[i] = bin_mass / volume #Msun/kpc**3
    rho_crit.convert_to_units("Msun/kpc**3")
    return DM_density/rho_crit, radii/R200crit


def get_avg_profiles(sim, snap_id, conv="none"):
    if snap_id > 34: #edge case
        snap_range = np.arange(34, 36+1) #range of snap_ids to average over
    else:
        snap_range = np.arange(snap_id-2, snap_id+2+1)
    N_snap = len(snap_range)
    
    N_bins = 30
    densities = np.zeros(N_bins)
    z = np.array([])
    for snap in snap_range:
        rho, rad = halo_density_profile(sim, conv=conv*0.9, snap_id=snap)
        densities = densities + rho
        z = np.append(z, get_redshift(sim, snap))
    return rad, densities/N_snap, z


def measure_convergence_radius(sim, 
                               catalogue="none",
                               snap_id=36):
    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    if catalogue == "none":
        catalogue = load_catalogue(path+catalogue_name+".properties")

    groups = load_groups(path+catalogue_name+".catalog_groups", catalogue=catalogue)
    
    M200crit = catalogue.masses.mass_200crit[0]
    xc = catalogue.positions.xcmbp[0]
    yc = catalogue.positions.ycmbp[0]
    zc = catalogue.positions.zcmbp[0]
    centre = [xc, yc, zc] / catalogue.scale_factor *unyt.Mpc
    #print(xc, M200crit)
    R200crit = catalogue.radii.r_200crit[0] / catalogue.scale_factor

    data = load(snapshot_path)
    data.dark_matter.coordinates = data.dark_matter.coordinates - centre
    dx = data.dark_matter.coordinates[:,0] 
    dy = data.dark_matter.coordinates[:,1]
    dz = data.dark_matter.coordinates[:,2]
    DM_radii = np.sqrt(dx**2 + dy**2 + dz**2)
    DM_mass = data.dark_matter.masses[0]

    Omega_m0 = data.metadata.cosmology.Om0
    Omega_de0 = data.metadata.cosmology.Ode0
    H0 = data.metadata.cosmology.H0
    a = data.metadata.a
    H = np.sqrt(H0**2 * (Omega_m0 * a**-3 + Omega_de0)).value * unyt.km / unyt.s / unyt.Mpc
    rho_crit = 3 * H**2 / (8*np.pi*unyt.G)

    sorted_particles = np.argsort(DM_radii)
    N_particles = len(DM_radii)
    particle_indices = np.arange(1, N_particles+1)
    particle_mass = DM_mass * particle_indices
    particle_volume = 4/3 * np.pi * DM_radii[sorted_particles]**3
    particle_avg_density = particle_mass / particle_volume
    conv_ratio = np.sqrt(200) / 8 * particle_indices / np.log(particle_indices) * (particle_avg_density / rho_crit)**(-1/2)
    interpolation = interp1d(conv_ratio, DM_radii[sorted_particles])
    conv_radius = interpolation(1) * DM_radii[0].units
    return conv_radius/R200crit


def plot_halo_density(list_of_sims, snap_id=36):
    N_sims = len(list_of_sims)

    cm = plt.cm.viridis(np.linspace(0,1,N_sims))

    for i in range(N_sims):
        conv_radius = measure_convergence_radius(list_of_sims[i], snap_id=snap_id)
        halo_density, radii = halo_density_profile(list_of_sims[i], conv=conv_radius, snap_id=snap_id)
        halo_density = halo_density / unyt.kpc**3 * unyt.Msun
        plt.semilogx(radii[1:], halo_density*radii[1:]**2,
                     label=list_of_sims[i], color=cm[i])
        if i == 0:
            ylim = plt.gca().get_ylim()
        plt.plot((conv_radius, conv_radius), ylim,
                 color=cm[i], linestyle="--")
    if snap_id != 36:
        z = get_redshift(list_of_sims[0], snap_id)
        plt.title("$z=$"+str(np.round(z,2)))
    plt.ylim(ylim)
    plt.xlabel("$r/R_{\\rm{200c}}$")
    plt.ylabel("$\\rho/\\rho_{\\rm{crit}} (r/R_{\\rm{200crit}})^2$")
    plt.legend()
    plt.show()


def calc_and_plot(sim_series, ax, 
                  snap_id=36,
                  sim_type="dmo"):
    if sim_series == "R":
        sims = ["R1", "R2", "R3", "R4", "R5"]
        cm = plt.cm.winter(np.linspace(0,1,len(sims)))
    elif sim_series == "K":
        sims = ["K1", "K2", "K3", "K4", "R5"]
        cm = plt.cm.autumn(np.linspace(0,1,len(sims)))
    elif sim_series == "M":
        sims = ["R1", "M2", "M3", "M4", "K1"]
        cm = plt.cm.summer(np.linspace(0,1,len(sims)))

    for i, sim in enumerate(sims):
        print(sim)
        conv_radius = measure_convergence_radius(sim, snap_id=snap_id)
        rad, halo_density, z = get_avg_profiles(sim, snap_id, conv=conv_radius)
        rad_mids = (rad[1:] + rad[:-1]) / 2

        ax.loglog(rad_mids, halo_density*rad_mids**2, ###need to fix y axis units
                     label=sim, color=cm[i])
        if i == 0:
            ylim = plt.gca().get_ylim()
        ax.plot((conv_radius, conv_radius), ylim,
                 color=cm[i], linestyle="--")
    ax.legend()
    plt.ylim(ylim)
    if sim_series == "R":
        plt.suptitle(str(np.round(z[-1],2)) + "$ \leq z \leq $"+str(np.round(z[0],2)))


def plot_avg_halo_density(snap_id=36):
    fig, ax = plt.subplots(nrows=3, ncols=1,
                           figsize=(3.3, 6),
                           sharex=True,
                           sharey=True,
                           gridspec_kw={'hspace' : 0, 'wspace' : 0})
    calc_and_plot("R", ax[0])
    calc_and_plot("K", ax[1])
    calc_and_plot("M", ax[2])

    #plt.ylim(ylim)
    ax[2].set_xlabel("$r/R_{\\rm{200c}}$")
    ax[1].set_ylabel("$\\rho/\\rho_{\\rm{crit}} (r/R_{\\rm{200crit}})^2$")
    plt.subplots_adjust(left=0.17, bottom=0.06, top=0.95)
    filename = "halo_density.png"
    plt.savefig(filename, dpi=300)
    plt.show()

list_of_sims = ["R4", "R3", "R2", "R1"]
#plot_halo_density(list_of_sims, snap_id=36)
plot_avg_halo_density()
