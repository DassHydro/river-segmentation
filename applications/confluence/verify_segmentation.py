import json
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import os
import tqdm
from shapely.ops import substring
import sys

from HiVDI.core.confluence.ConfluenceNetCDF import ConfluenceSwotNetCDF
from HiVDI.core.sword.SwordNetCDF import SwordNetCDF

from rt_river_segmentation.denoising import hydraulic_filtering
from rt_river_segmentation.segmentation_methods import *


def load_multiples_reaches_data(dataset, inputdir):
    
    # Retrieve SWORD file from first reach parameters
    sword_file = dataset[0]["sword"]
    
    # Retrieve multiple reaches IDs and observations files
    reachids = []
    swot_files = []
    for single_reach in dataset:

        # Retrieve current reach ID and observation file
        reachids.append(single_reach["reach_id"])
        swot_files.append(single_reach["swot"])
        
        # Check that SWORD and SoS file remain the same
        if single_reach["sword"] != sword_file:
            raise RuntimeError("Different SWORD files detected inside multiple reaches set")
            
    reachid_up = reachids[0]
    reachid_dn = reachids[-1]
    reachid = None
    
    # Load SWORD file
    print("Loading SWORD data")
    fname = os.path.expandvars(os.path.join(inputdir, "sword", sword_file))
    sword = SwordNetCDF(fname, reaches_subset=reachids)
    
    # Load observations
    print("Loading SWOT data")
    
    # Load each single reach observations
    obs = None
    for swot_file in swot_files:
        fname = os.path.expandvars(os.path.join(inputdir, "swot", swot_file))
        reach_obs = ConfluenceSwotNetCDF(fname, sword, time_subset=None, remove_nan=False)
        if obs is None:
            obs = reach_obs
        else:
            obs.concatenate(reach_obs)
                
    # Remove nan observations
    obs.remove_non_valid_observations()
    
    print("Dimensions: P=%i, N=%i, R=%i" % (obs.nodes.t.size, obs.nodes.x.size, 
                                            obs.reaches.x.size))
    print("widths: mean=%.2f m, std=%.2f m" % (np.nanmean(obs.nodes.W.flatten()),
                                               np.nanstd(obs.nodes.W.flatten())))
    print("slopes: mean=%.2f m/km, std=%.2f m/km" % (np.nanmean(obs.reaches.S.flatten()) * 1000.0,
                                                     np.nanstd(obs.reaches.S.flatten()) * 1000.0))
    
    return obs



def load_shp_reaches_data(dataset, sword_shp_dir):
    
    # Retrieve SWORD file from first reach parameters
    sword_file = dataset[0]["sword"]
    
    # Retrieve continent ID
    continentID = sword_file.split("_")[0]
    
    # Retrieve reaches IDs
    reachids = []
    for single_reach in dataset:

        # Retrieve current reach ID and observation file
        reachids.append(str(single_reach["reach_id"]))
            
    # Retrieve shp name
    previous_shp_fname = None
    for reachid in reachids:
        shpID = reachid[0:2]
        shp_fname = "%s_sword_reaches_hb%s_v2.shp" % (continentID, shpID)
        if previous_shp_fname is not None:
            if shp_fname != previous_shp_fname:
                raise RuntimeError("Different shp file")
        previous_shp_fname = shp_fname
        
    shp_dataset = gpd.read_file(os.path.join(sword_shp_dir, continentID.upper(), shp_fname))
    reaches_dataset = shp_dataset[shp_dataset["reach_id"].isin(reachids)]
    
    reaches_dataset = reaches_dataset.sort_values(by="dist_out", ascending=False)
    
    return reaches_dataset

#def get_product_data(data_dir, product, source="nominal", debug=False):
    
    #reaches_fname = os.path.join(data_dir, product, "riverobs_%s_20201105" % source, "river_data", "reaches.shp")
    #nodes_fname = os.path.join(data_dir, product, "riverobs_%s_20201105" % source, "river_data", "nodes.shp")
    
    #if debug: 
        #print("Load reaches dataset")
    #reaches_dataset = gpd.read_file(reaches_fname)
    
    #if debug: 
        #print("Load nodes dataset")
    #nodes_dataset = gpd.read_file(nodes_fname)
        
    #return reaches_dataset, nodes_dataset


#def find_segments_upstream_reaches(reaches_dataset):
    
    #reaches_dataset["upstream_reach"] = pd.Series(data=np.zeros(reaches_dataset.shape[0], dtype=int), index=reaches_dataset.index)
    
    ##reaches_dataset[reaches_dataset["n_rch_up"] != 1]["upstream_reach" = 1
    #for index in reaches_dataset.index:
        #current_reach = reaches_dataset.loc[index, :]
        #if current_reach["type"] == 1 and current_reach["n_rch_up"] != 1:
            #reaches_dataset.loc[index, "upstream_reach"] = 1
        #elif current_reach["type"] != 1 and current_reach["n_rch_dn"] == 1:
            #next_reach_id = current_reach["rch_id_dn"]
            #next_index = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].index
            #if next_index.size > 0:
            ##next_reach_subset = reaches_dataset.loc[next_index, :])
            ##print("next_index=", next_index)
            ##print("next_subset=", reaches_dataset.loc[next_index, :])
            ##if 
                #next_reach = reaches_dataset.loc[next_index, :].iloc[0]
                ##print(next_reach["type"])
                #if next_reach["type"] == 1:
                    #reaches_dataset.loc[next_index, "upstream_reach"] = 1
    
    #upstream_reaches_dataset = reaches_dataset[reaches_dataset["upstream_reach"] == 1]
    #upstream_reaches_dataset.sort_values(by=["dist_out"], ascending=False, inplace=True)
    
    #return upstream_reaches_dataset


#def get_segment(reaches_dataset, upstream_reach, debug=False):
    
    #index = reaches_dataset[reaches_dataset["reach_id"] == upstream_reach].index[0]
    #current_reach = reaches_dataset.iloc[index, :]
    #reaches = [current_reach]
    #if debug:
        #print("Start reach: %s" % current_reach["reach_id"])
    #while current_reach["n_reach_dn"] == 1:
        #next_reach_id = current_reach["rch_id_dn"].split(" ")[0]
        #next_index = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].index
        #if next_index.size == 0:
            #break
        #next_reach = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].iloc[0]
        #if debug:
            #print("- Next reach: %s" % next_reach_id)
        ##if next_reach["type"] != 1:
            ##break
        #if next_reach["n_reach_up"] > 1:
            #break
        #reaches.append(next_reach)
        #current_reach = next_reach
        
    #return reaches
        
        
def run_segmentation(x, heights, widths, lc=5.0, segmentation_method="baseline"):
    
    lc *= 1000.0
    
    xc = x[np.logical_and(np.isfinite(heights), np.isfinite(widths))]
    Z = heights[np.logical_and(np.isfinite(heights), np.isfinite(widths))]
    W = widths[np.logical_and(np.isfinite(heights), np.isfinite(widths))]
    
    #print("x=", x)
    x0 = x[0] - xc
    #print("x0=", x0)
    #Z = Hc
    #print("Z=", Z)
    #W = Wc
    
    pas = 25
    N = int(np.floor(x0[-1] / pas))
    xi = np.linspace(x0[0], x0[0] + N * pas, N+1, endpoint=True)
    Zi = np.interp(xi, x0, Z)
    Wi = np.interp(xi, x0, W)
    
    if True:
        plt.plot(xi, Zi, "b-+")
        plt.plot(x0, Z, "r--")
        plt.show()
    
    print("-" * 80)
    print("Hydraulic filtering")
    print("-" * 80)
    Zfiltered = hydraulic_filtering(xi, Zi, x_direction="downstream", plot_steps=False)
    Zi = Zfiltered
    print("-" * 80)
    
    if True:
        plt.plot(xi, Zi, "b-+")
        plt.plot(x0, Z, "r--")
        plt.show()

    if segmentation_method == "baseline":
        d2xZ_pos, d2xZ_neg, segmentation_bounds, wavelet = segmentation_baseline(xi, Zi, None, lc, min_length=1000.0)
    else:
        _, _, _, _, segmentation_bounds, wavelet_Z, wavelet_W = segmentation_advanced(xi, Zi, None, Wi, None, lc, min_length=1000.0)
        
    if True:
        plt.plot(xi, Zi, "b-")
        if segmentation_method == "baseline":
            plt.plot(d2xZ_pos[0], d2xZ_pos[1], "r-")
            plt.plot(d2xZ_neg[0], d2xZ_neg[1], "g-")
        for bound in segmentation_bounds:
            plt.axvline(bound, color="k", ls="--")
        plt.show()

    # Retransform segmentation bounds to outlet distances
    segmentation_bounds = x[0] - segmentation_bounds
    
    return segmentation_bounds

        
        
def compute_geometry(reaches_data, segmentation_bounds):
    
    ir = 0
    
    # Sort reaches by decreasing outlet distances
    reaches_data_sorted = reaches_data.sort_values(by=["dist_out"], ascending=False)
    
    print(reaches_data_sorted["dist_out"])
    
    xup = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "dist_out"]
    reach_length = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "reach_len"]
    print("First reach: xup=", xup)
    xdn = xup - reach_length
    print("             xdn=", xdn)
    reach_geometry = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "geometry"]
    choice = input()
    
    bound_points = []
    for bound in segmentation_bounds:
        
        while xdn > bound:
            ir += 1
            xup = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "dist_out"]
            reach_length = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "reach_len"]
            xdn = xup - reach_length
            reach_geometry = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "geometry"]
            
        print("bound:", bound)
        print("xup:", xup)
        print("xdn:", xdn)
        alpha = (bound - xdn) / (xup - xdn)
        print("alpha:", alpha)
        bound_point = substring(reach_geometry, start_dist=alpha, end_dist=alpha, normalized=True)
        bound_points.append(bound_point)
        print("point:", bound_point)
        choice = input
        
    bounds_dataset = gpd.GeoDataFrame(data={"id" : np.arange(0, len(segmentation_bounds))}, geometry=bound_points)
    
    return bounds_dataset


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser("Generate a pseudo PEPSI from a segmented PEPSI original case")
    parser.add_argument("verify_dir", type=str, 
                        help="Path to the Verify directory")
    parser.add_argument("sword_shp_dir", type=str, 
                        help="Path to the directory containing SWORD data in Shapefile format")
    parser.add_argument("-json", dest="json_file", type=str, default="reaches_sets_curated_v4.json",
                        help="Json file containing the reaches sets")
    args = parser.parse_args()
    
    verify_dir = args.verify_dir 
    sword_shp_dir = args.sword_shp_dir
    
    # Open curated reaches json
    reachjson = os.path.join(verify_dir, "input", args.json_file)
    with open(reachjson) as jsonfile:
        reaches_sets = json.load(jsonfile)
        
    for set_index, reaches_set in enumerate(reaches_sets):
        
        if isinstance(reaches_set, list):
            
            if len(reaches_set) < 8:
                print("- Skip small reach set %03i : %i reaches" % (set_index, len(reaches_set)))
                continue
            
            obs = load_multiples_reaches_data(reaches_set, os.path.join(verify_dir, "input"))
            reaches_dataset = load_shp_reaches_data(reaches_set, sword_shp_dir)
            print(reaches_dataset["dist_out"])
            choice = input()
            
            dataH = obs.nodes.H
            dataW = obs.nodes.W
            
            flowlines_sorted = np.argsort(dataH[:, 0])
            low_flow_index = flowlines_sorted[dataH.shape[0]*10//100]
            median_flow_index = flowlines_sorted[dataH.shape[0]//2]
            
            H = dataH[low_flow_index, ::-1]
            search = 0
            while np.any(np.isnan(H)) and (low_flow_index - search >= 0 or low_flow_index + search < dataH.shape[0]*30//100):
                search += 1
                below_index = max(0, low_flow_index-search)
                above_index = min(dataH.shape[0]*30//100-1, low_flow_index+search)
                H1 = dataH[below_index, ::-1]
                H2 = dataH[above_index, ::-1]
                
                if np.all(np.isfinite(H1)):
                    H = H1
                else:
                    H = H2
            if np.any(np.isnan(H)):
                print("Unable to find a complete low-flow profile")
                continue

            W = dataW[median_flow_index, ::-1]
            search = 0
            while np.any(np.isnan(W)) and (median_flow_index - search >= dataH.shape[0]*40//100 or median_flow_index + search < dataH.shape[0]*60//100):
                search += 1
                below_index = max(dataH.shape[0]*40//100-1, median_flow_index-search)
                above_index = min(dataH.shape[0]*60//100-1, median_flow_index+search)
                W1 = dataW[below_index, ::-1]
                W2 = dataW[above_index, ::-1]
                
                if np.all(np.isfinite(W1)):
                    W = W1
                else:
                    W = W2
            if np.any(np.isnan(W)):
                print("Unable to find a complete median profile")
                continue
            
            if True:
                fig, axes = plt.subplots(2, 1, sharex=True)
                axes[0].plot(obs.nodes.x * 0.001, dataH.T, color="grey", alpha=0.5)
                axes[0].plot(obs.nodes.x * 0.001, H, color="blue", label="selected")
                axes[1].plot(obs.nodes.x * 0.001, dataW.T, color="grey", alpha=0.5)
                axes[1].plot(obs.nodes.x * 0.001, W, color="blue", label="selected")
                axes[0].set_ylabel("H (m)")
                axes[1].set_xlabel("x (km)")
                axes[1].set_ylabel("W (m)")
                plt.tight_layout()
                plt.show()
            
            x = obs.nodes.x
            H = H[::-1]
            W = W[::-1]
            segmentation_bounds = run_segmentation(x, H, W, segmentation_method="advanced")
            print("segmentation_bounds=", segmentation_bounds)
            if segmentation_bounds[0] < reaches_dataset.loc[reaches_dataset.index[0], "dist_out"]:
                segmentation_bounds[0] = reaches_dataset.loc[reaches_dataset.index[0], "dist_out"]
            if segmentation_bounds[-1] > reaches_dataset.loc[reaches_dataset.index[-1], "dist_out"]:
                segmentation_bounds[-1] = reaches_dataset.loc[reaches_dataset.index[-1], "dist_out"]
                
            
            bounds_dataset = compute_geometry(reaches_dataset, segmentation_bounds)
            bounds_dataset.to_file("out/verify_set%03i_bounds.shp" % (set_index+1))

