import numpy as np
from swiftsimio import load
import swiftsimio.metadata.particle as swp
from velociraptor import load as load_catalogue
import unyt
import matplotlib.pyplot as plt


def make_paths(sim, snap_id=36):
    path = "../L0300N0564_VR18_"+sim+"/stf/snap_"+sim+"_"+str(snap_id).zfill(4)+"/"
    catalogue_name = "snap_"+sim+"_" + str(snap_id).zfill(4)
    snapshot_path = "../L0300N0564_VR18_"+sim+"/snapshots/snap_"+sim+"_"+str(snap_id).zfill(4)+".hdf5"
    return path, catalogue_name, snapshot_path


def contaminating_particles(sim):
    path, catalogue_name, snapshot_path = make_paths(sim)
    catalogue = load_catalogue(path+catalogue_name+".properties")
    
    # Get centre of main halo
    xc = catalogue.positions.xcmbp[0]
    yc = catalogue.positions.ycmbp[0]
    zc = catalogue.positions.zcmbp[0]
    centre = [xc, yc, zc]
    R200crit = catalogue.radii.r_200crit[0]

    #Add second particle type to be read in
    swp.particle_name_underscores[2] = "dm_bg"
    swp.particle_name_class[2] = "DM_bg"
    swp.particle_name_text[2] = "DM_bg"

    # Load data and find radii of background particles
    data = load(snapshot_path)
    data.dm_bg.coordinates = data.dm_bg.coordinates - centre
    dx = data.dm_bg.coordinates[:,0] 
    dy = data.dm_bg.coordinates[:,1]
    dz = data.dm_bg.coordinates[:,2]
    bg_radii = np.sqrt(dx**2 + dy**2 + dz**2)

    min_dist = np.min(bg_radii)
    return min_dist/R200crit


#contaminating_particles("R5")
