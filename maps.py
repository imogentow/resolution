import numpy as np
from swiftsimio import load
from swiftsimio import mask
from swiftsimio.visualisation.projection_backends import backends_parallel
from swiftsimio.visualisation.smoothing_length_generation import generate_smoothing_lengths
scatter = backends_parallel["subsampled"]
from velociraptor import load as load_catalogue
from velociraptor.particles import load_groups
import unyt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["mathtext.fontset"] = "stix"

def make_paths(sim, sim_type="dmo", snap_id=36):
    path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/stf/snap_"+sim+"_"+str(snap_id).zfill(4)+"/"
    catalogue_name = "snap_"+sim+"_" + str(snap_id).zfill(4)
    snapshot_path = "../"+sim_type+"/L0300N0564_VR18_"+sim+"/snapshots/snap_"+sim+"_"+str(snap_id).zfill(4)+".hdf5"
    return path, catalogue_name, snapshot_path


def make_smooth_map(data, weighting, no_pixels, side_length, projection="z"):
    """
    Uses swiftsimio to make smoothed maps.
    side_size: size of image in multiples of R_200m
    """

    pixel_length = side_length / no_pixels #make sure unyt matches
    proj_length = side_length
    half_side_length = 0.5*side_length
    half_proj_length = 0.5*proj_length
    
    pixel_area = pixel_length.to('m')**2

    if projection == "z":
        ax = 2
    elif projection == "x":
        ax = 0
    elif projection == "y":
        ax = 1
    else:
        print("Invalid projection")
        return

    dx = data.coordinates[:,ax-2]
    dy = data.coordinates[:,ax-1]

    x_coords = dx
    y_coords = dy

    x_bounds = (x_coords + half_side_length) / side_length #put within bounds [0,1]
    y_bounds = (y_coords + half_side_length) / side_length

    x_bounds = np.asarray(x_bounds, dtype=np.float64) #increases speed of swiftsimio calculation
    y_bounds = np.asarray(y_bounds, dtype=np.float64)

    smooth = data.smoothing_lengths
    smoothing_lengths = smooth / side_length
    smoothing_lengths = np.asarray(smoothing_lengths, dtype=np.float32)

    weighting = np.asarray(weighting, dtype=np.float32)

    smoothed_map = scatter(x=x_bounds, y=y_bounds, m=weighting, h=smoothing_lengths, res=no_pixels).T / no_pixels**2
    
    return smoothed_map


def get_DM_map_data(sim, snap_id=36, no_pixels=600):
    side_factor = 3

    path, catalogue_name, snapshot_path = make_paths(sim, snap_id=snap_id)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    groups = load_groups(path+catalogue_name+".catalog_groups", catalogue=catalogue)
    particles, _ = groups.extract_halo(halo_id=0) #find most massive halo in simulation
    
    # Get some halo properties
    centre = [particles.x_mbp, particles.y_mbp, particles.z_mbp]*unyt.Mpc / catalogue.scale_factor
    R200crit = catalogue.radii.r_200crit[0] / catalogue.scale_factor
    side_length = side_factor * R200crit
    half_side = side_length * 0.5

    # Define region for swiftsimio to read in
    cluster_mask = mask(snapshot_path)
    region = [[centre[0] - half_side, centre[0] + half_side],
              [centre[1] - half_side, centre[1] + half_side],
              [centre[2] - half_side, centre[2] + half_side]]
    cluster_mask.constrain_spatial(region)

    # Load data
    data = load(snapshot_path, mask=cluster_mask)

    # Sort particle data
    data.dark_matter.smoothing_lengths = generate_smoothing_lengths(data.dark_matter.coordinates,
                                                                    data.metadata.boxsize,
                                                                    1.8,
                                                                    neighbours=57)

    data.dark_matter.coordinates = data.dark_matter.coordinates - centre
    subset = np.where((data.dark_matter.coordinates[:,0]**2 < half_side**2) &
                      (data.dark_matter.coordinates[:,1]**2 < half_side**2) &
                      (data.dark_matter.coordinates[:,2]**2 < half_side**2))[0] 
    data.dark_matter.coordinates = data.dark_matter.coordinates[subset,:]
    data.dark_matter.masses = data.dark_matter.masses[subset]
    data.dark_matter.smoothing_lengths = data.dark_matter.smoothing_lengths[subset]

    DM_mass_map = make_smooth_map(data.dark_matter, data.dark_matter.masses, no_pixels, side_length)

    return DM_mass_map


def plot_maps_grid(*sims, snap_id=36):
    N_sims = len(sims) #must be 5, first given will be larger than the rest
    if N_sims != 5:
        return

    fig = plt.figure(figsize=(6, 13.5))
    gs = GridSpec(4,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2:,0:])
    axes = [ax1, ax2, ax3, ax4, ax5]

    for i, sim in enumerate(sims):
        DM_mass_map = get_DM_map_data(sim, snap_id=snap_id)
        axes[i].imshow(LogNorm()(DM_mass_map), cmap="plasma")
        axes[i].axis('off')
        axes[i].text(0.05, 0.05, sim, transform=axes[i].transAxes, color="white")
    filename = "K_series_maps.png"
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_maps(*sims, snap_id=36):
    N_sims = len(sims) 

    fig, axes = plt.subplots(nrows=1, ncols=N_sims,
                             figsize=(N_sims*2, 3))

    for i, sim in enumerate(sims):
        DM_mass_map = get_DM_map_data(sim, snap_id=snap_id)
        if N_sims > 1:
            axes[i].imshow(LogNorm()(DM_mass_map), cmap="plasma")
            axes[i].axis('off')
            axes[i].text(0.05, 0.05, sim, transform=axes[i].transAxes, color="white")
        else:
            axes.imshow(LogNorm()(DM_mass_map), cmap="plasma")
            axes.axis('off')
            axes.text(0.05, 0.05, sim, transform=axes.transAxes, color="white")
    plt.show()

#plot_maps("R4", "R3", "R2", "R1", snap_id=25)
plot_maps_grid("K1", "K2", "K3", "K4", "R5")
