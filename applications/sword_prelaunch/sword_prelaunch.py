import argparse
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import os
import tqdm
from shapely.ops import substring
import sys

from rt_river_segmentation.denoising import hydraulic_filtering
from rt_river_segmentation.segmentation_methods import *


def load_nc_sword_dataset(reaches_fname, nodes_fname=None, debug=False):

    if debug: 
        print("Load reaches dataset")
    reaches_dataset = gpd.read_file(reaches_fname)
    
    if nodes_fname is not None:
        if debug: 
            print("Load nodes dataset")
        nodes_dataset = gpd.read_file(nodes_fname)
    else:
        nodes_dataset = None
        
    return reaches_dataset, nodes_dataset


def find_segments_upstream_reaches(reaches_dataset):
    
    reaches_dataset["upstream_reach"] = pd.Series(data=np.zeros(reaches_dataset.shape[0], dtype=int), index=reaches_dataset.index)
    
    #reaches_dataset[reaches_dataset["n_rch_up"] != 1]["upstream_reach" = 1
    for index in reaches_dataset.index:
        current_reach = reaches_dataset.loc[index, :]
        if current_reach["type"] == 1 and current_reach["n_rch_up"] != 1:
            reaches_dataset.loc[index, "upstream_reach"] = 1
        elif current_reach["type"] != 1 and current_reach["n_rch_dn"] == 1:
            next_reach_id = current_reach["rch_id_dn"]
            next_index = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].index
            if next_index.size > 0:
            #next_reach_subset = reaches_dataset.loc[next_index, :])
            #print("next_index=", next_index)
            #print("next_subset=", reaches_dataset.loc[next_index, :])
            #if 
                next_reach = reaches_dataset.loc[next_index, :].iloc[0]
                #print(next_reach["type"])
                if next_reach["type"] == 1:
                    reaches_dataset.loc[next_index, "upstream_reach"] = 1
    
    upstream_reaches_dataset = reaches_dataset[reaches_dataset["upstream_reach"] == 1]
    upstream_reaches_dataset.sort_values(by=["dist_out"], ascending=False, inplace=True)
    
    return upstream_reaches_dataset


def get_segment(reaches_dataset, upstream_reaches_dataset, index, debug=False):
    
    
    reaches = [upstream_reaches_dataset.iloc[index, :]]
    current_reach = upstream_reaches_dataset.iloc[index, :]
    if debug:
        print("Start reach: %s" % current_reach["reach_id"])
    while current_reach["n_rch_dn"] == 1:
        next_reach_id = current_reach["rch_id_dn"]
        next_index = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].index
        if next_index.size == 0:
            break
        next_reach = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].iloc[0]
        if debug:
            print("- Next reach: %s" % next_reach_id)
        if next_reach["type"] != 1:
            break
        if next_reach["n_rch_up"] > 1:
            break
        reaches.append(next_reach)
        current_reach = next_reach
        
    return reaches



def get_segment_nodes(nodes_dataset, segment_reaches, debug=False):
    
    segment_reaches_ids = [reach["reach_id"] for reach in segment_reaches]
    
    segment_nodes = nodes_dataset[nodes_dataset["reach_id"].isin(segment_reaches_ids)]
    segment_nodes.sort_values(by=["dist_out"], ascending=False, inplace=True)
    #print(segment_nodes.loc[:, ["node_id", "reach_id"]])
    
    return segment_nodes



def segment_reaches_bounds(segment_nodes, debug=False):
    
    prev_node = segment_nodes.iloc[0, :]
    xb = [prev_node["dist_out"]]
    for index in segment_nodes.index[1:]:
        if segment_nodes.loc[index, "reach_id"] != prev_node["reach_id"]:
            xb.append(0.5 * (prev_node["dist_out"] + segment_nodes.loc[index, "dist_out"]))
        prev_node = segment_nodes.loc[index, :]
        
    last_node = segment_nodes.iloc[-1, :]
    xb.append(last_node["dist_out"])
    
    return xb


def segment_fixed_reaches_bounds(segment_nodes, debug=False):
    
    prev_node = segment_nodes.iloc[0, :]
    xb = [prev_node["dist_out"]]
    for index in segment_nodes.index[1:]:
        if segment_nodes.loc[index, "reach_id"] != prev_node["reach_id"]:
            if segment_nodes.loc[index, "type"] == 4 or prev_node["type"] == 4:
                xb.append(0.5 * (prev_node["dist_out"] + segment_nodes.loc[index, "dist_out"]))
        prev_node = segment_nodes.loc[index, :]
        
    last_node = segment_nodes.iloc[-1, :]
    xb.append(last_node["dist_out"])
    
    return xb


def new_segment_reaches(segment_nodes, segment_index, d2xZ_pos, d2xZ_neg, debug=False):
    
    x = segment_nodes["dist_out"].values
    x0 = x[0] - x
    pos_flag = np.interp(x0, d2xZ_pos[0], np.isnan(d2xZ_pos[1]))
    pos_flag[np.isnan(pos_flag)] = 1.0
    neg_flag = np.interp(x0, d2xZ_neg[0], np.isnan(d2xZ_neg[1]))
    neg_flag[np.isnan(neg_flag)] = 1.0
    
    reaches = []
    prev_node_reach_id = "NA"
    new_reach_id = None
    new_reach_index = 0
    #new_reach_id = "%08i%02i" % (segment_index, new_reach_index)
    segment_nodes["new_reach_id"] = pd.Series(data=["NA"] * segment_nodes.shape[0], index=segment_nodes.index)
    segment_nodes["pos_flag"] = pd.Series(data=pos_flag, index=segment_nodes.index)
    segment_nodes["neg_flag"] = pd.Series(data=neg_flag, index=segment_nodes.index)
    for i in range(0, segment_nodes.shape[0]):
        node = segment_nodes.iloc[i, :]
        if node["type"] != 1:
            #print(node)
            segment_nodes.loc[segment_nodes.index[i], "new_reach_id"] = segment_nodes.loc[segment_nodes.index[i], "reach_id"]
        else:
            if prev_node_reach_id == new_reach_id:
                if pos_flag[i-1] != pos_flag[i]:
                    
                    # New reach
                    new_reach_index += 1
                    new_reach_id = "%08i%02i" % (segment_index+1, new_reach_index)
                        
            else:
                
                # New reach
                new_reach_index += 1
                new_reach_id = "%08i%02i" % (segment_index+1, new_reach_index)
                
            segment_nodes.loc[segment_nodes.index[i], "new_reach_id"] = new_reach_id
        prev_node_reach_id = segment_nodes.loc[segment_nodes.index[i], "new_reach_id"]
        

def new_segment_reaches2(segment_nodes, segment_index, segmentation_bounds, debug=False):
    
    x = segment_nodes["dist_out"].values
    
    # Compute segmentation index for each node
    # Compute segmentation index for each node
    segmentation_indices = np.zeros(segment_nodes.shape[0], dtype=int)
    index = 0
    for i in range(0, x.size):
        if index < len(segmentation_bounds) - 1:
            while x[i] < segmentation_bounds[index+1]:
                index += 1
                if index >= len(segmentation_bounds) - 1:
                    index = len(segmentation_bounds) - 2
                    break
        segmentation_indices[i] = index
    
    reaches = []
    prev_node_reach_id = "NA"
    new_reach_id = None
    new_reach_index = 0
    #new_reach_id = "%08i%02i" % (segment_index, new_reach_index)
    segment_nodes["new_reach_id"] = pd.Series(data=["NA"] * segment_nodes.shape[0], index=segment_nodes.index)
    segment_nodes["seg_index"] = pd.Series(data=segmentation_indices, index=segment_nodes.index)
    for i in range(0, segment_nodes.shape[0]):
        node = segment_nodes.iloc[i, :]
        if node["type"] != 1:
            #print(node)
            segment_nodes.loc[segment_nodes.index[i], "new_reach_id"] = segment_nodes.loc[segment_nodes.index[i], "reach_id"]
        else:
            if prev_node_reach_id == new_reach_id:
                if segmentation_indices[i-1] != segmentation_indices[i]:
                    
                    # New reach
                    new_reach_index += 1
                    new_reach_id = "%08i%02i" % (segment_index+1, new_reach_index)
                        
            else:
                
                # New reach
                new_reach_index += 1
                new_reach_id = "%08i%02i" % (segment_index+1, new_reach_index)
                
            segment_nodes.loc[segment_nodes.index[i], "new_reach_id"] = new_reach_id
        prev_node_reach_id = segment_nodes.loc[segment_nodes.index[i], "new_reach_id"]


def compute_geometry(segment_reaches, segmentation_bounds):
    
    ir = 0
    
    # Sort reaches by decreasing outlet distances
    #reaches_data_sorted = reaches_data.sort_values(by=["p_dist_out"], ascending=False)
    
    #print(reaches_data_sorted["p_dist_out"])
    
    xup = segment_reaches[ir]["dist_out"]
    reach_length = segment_reaches[ir]["reach_len"]
    #print("First reach: xup=", xup)
    xdn = xup - reach_length
    #print("             xdn=", xdn)
    reach_geometry = segment_reaches[ir]["geometry"]
    #choice = input()
    
    bound_points = []
    for bound in segmentation_bounds:
        
        while xdn > bound:
            ir += 1
            xup = segment_reaches[ir]["dist_out"]
            reach_length = segment_reaches[ir]["reach_len"]
            xdn = xup - reach_length
            reach_geometry = segment_reaches[ir]["geometry"]
            
        #print("bound:", bound)
        #print("xup:", xup)
        #print("xdn:", xdn)
        alpha = (bound - xdn) / (xup - xdn)
        #print("alpha:", alpha)
        bound_point = substring(reach_geometry, start_dist=alpha, end_dist=alpha, normalized=True)
        bound_points.append(bound_point)
        #print("point:", bound_point)
        #choice = input
        
    bounds_dataset = gpd.GeoDataFrame(data={"id" : np.arange(0, len(segmentation_bounds))}, geometry=bound_points)
    
    return bounds_dataset
        
        
def run_method(subset_index, reaches_fname, nodes_fname, segmentation_method="baseline", lc=5.0, min_length=1.0):
    
    reaches_dataset, nodes_dataset = load_nc_sword_dataset(reaches_fname, nodes_fname, debug=False)
    
    upstream_reaches = find_segments_upstream_reaches(reaches_dataset)

    #reaches_type1_subset = reaches_dataset[reaches_dataset["type"] == 1]
    #print("- Number of reaches of type 1 in SWORD: %i" % reaches_type1_subset.shape[0])
    #return 0, reaches_type1_subset.shape[0]
    #choice = input()
    
    #print(upstream_reaches)
        
    lc *= 1000.0
    min_length *= 1000.0
    
    sum_sword = 0
    sum_rtseg = 0
    for iseg in range(0, upstream_reaches.shape[0]):
        
        print("=" * 80)
        print(" SEGMENT %06i/%06i" % (iseg + 1, upstream_reaches.shape[0]))
        print("=" * 80)
        
        print("-" * 80)
        print("Initial segmentation (SWORD)")
        print("-" * 80)
        segment_reaches = get_segment(reaches_dataset, upstream_reaches, iseg, debug=False)
        sum_sword += len(segment_reaches)
        print("- Number of reaches: %i" % len(segment_reaches))
        
        segment_nodes = get_segment_nodes(nodes_dataset, segment_reaches, debug=False)
        print("- Number of nodes: %i" % segment_nodes.shape[0])
        if segment_nodes.shape[0] < 5:
            segment_nodes["new_reach_id"] = pd.Series(data=segment_nodes.loc[:, "reach_id"], index=segment_nodes.index)
            segment_nodes["pos_flag"] = pd.Series(data=np.zeros(segment_nodes.shape[0]),
                                                  index=segment_nodes.index)
            segment_nodes["neg_flag"] = pd.Series(data=np.zeros(segment_nodes.shape[0]),
                                                  index=segment_nodes.index)
            continue
            
        initial_bounds = []
        for reach in segment_reaches:
            initial_bounds.append(reach["dist_out"])
        initial_bounds.append(segment_reaches[-1]["dist_out"] + segment_reaches[-1]["reach_len"])
        reaches_bounds = segment_reaches_bounds(segment_nodes)
        fixed_reaches_bounds = segment_fixed_reaches_bounds(segment_nodes)
        print("-" * 80)
        print("")
        
        x = segment_nodes["dist_out"].values
        x0 = x[0] - x
        # => x = x[0] - x0
        Z = segment_nodes["wse"].values
        W = segment_nodes["width"].values
        
        pas = 50
        N = int(np.floor(x0[-1] / pas))
        xi = np.linspace(x0[0], x0[0] + N * pas, N+1, endpoint=True)
        Zi = np.interp(xi, x0, Z)
        Wi = np.interp(xi, x0, W)
        
        print("-" * 80)
        print("Hydraulic filtering")
        print("-" * 80)
        Zfiltered = hydraulic_filtering(xi, Zi, x_direction="downstream", plot_steps=False)
        Zi = Zfiltered
        print("-" * 80)

        #plt.plot(x0 * 0.001, Z)
        #plt.plot(xi * 0.001, Zi, "--")
        #plt.show()
        
        if segmentation_method == "baseline":
            d2xZ_pos, d2xZ_neg, segmentation_bounds, wavelet = segmentation_baseline(xi, Zi, None, lc, min_length=min_length)
        else:
            _, _, _, _, segmentation_bounds, wavelet_Z, wavelet_W = segmentation_advanced(xi, Zi, None, Wi, None, lc, min_length=min_length)
        #test = np.interp(x0, d2xZ_pos[0], d2xZ_pos[1])
        #print(test)

        # Retransform segmentation bounds to outlet distances
        segmentation_bounds = x[0] - segmentation_bounds
        
        #plt.plot((x[0] - d2xZ_pos[0]) * 0.001, d2xZ_pos[1], linewidth = 4, label = r'$Z(\partial_x^2 > 0)$', color = 'g', linestyle = '-')
        #plt.plot((x[0] - d2xZ_neg[0]) * 0.001, d2xZ_neg[1], linewidth = 4, label = r'$Z(\partial_x^2 < 0)$', color = 'r', linestyle = '-')
        #for reach_bound in reaches_bounds:
            #plt.axvline(reach_bound * 0.001, c="k", ls="--")
        #for reach_bound in fixed_reaches_bounds:
            #plt.axvline(reach_bound * 0.001, c="b", ls="-")
        #plt.show()
        
        #if sword_reaches_length is None:
            #sword_reaches_length = np.abs(np.diff(fixed_reaches_bounds))
            #segmentation_reaches_length = np.abs(np.diff(segmentation_bounds))
        #else:
            #sword_reaches_length = np.concatenate((sword_reaches_length, np.abs(np.diff(fixed_reaches_bounds))))
            #segmentation_reaches_length = np.concatenate((segmentation_reaches_length, np.abs(np.diff(segmentation_bounds))))
        fout1 = open("out/sword_reaches_lengths.csv", "a")
        #print("initial_bounds=", initial_bounds)
        #choice = input()
        for ir in range(0, len(initial_bounds)-1):
            fout1.write("%i;%i;%i;%f\n" % (subset_index+1, iseg+1, ir+1, abs(initial_bounds[ir+1] - initial_bounds[ir])))
        fout1.close()
        fout2 = open("out/rtseg_reaches_lengths.csv", "a")
        for ir in range(0, len(fixed_reaches_bounds)-1):
            fout2.write("%i;%i;%i;%f\n" % (subset_index+1, iseg+1, ir+1, abs(segmentation_bounds[ir+1] - segmentation_bounds[ir])))
        fout2.close()
        
        print("-" * 80)
        print("Final segmentation")
        print("-" * 80)
        #new_segment_reaches(segment_nodes, iseg, d2xZ_pos, d2xZ_neg)
        new_segment_reaches2(segment_nodes, iseg, segmentation_bounds)
        #print(segment_nodes)
        #choice = input()
        sum_rtseg += len(segment_nodes["new_reach_id"].unique())
        print("- Number of reaches: %i" % len(segment_nodes["new_reach_id"].unique()))
        print("-" * 80)
        print("")
        
        if len(segment_reaches) > 10:
            print("-" * 80)
            print("Export new segmentation")
            print("-" * 80)
            #fname = "out/segment_%s_%s_%s.shp" % (segment_reaches[0], segment_reaches[-1], segmentation_method)
            #print(" - Export segment to %s" % fname)
            #segment_nodes.to_file(fname)
            
            # TODO
            bounds_dataset = compute_geometry(segment_reaches, segmentation_bounds)
            fname = "out/subset%i_segment_%s_%s_%s.shp" % (subset_index+1, segment_reaches[0]["reach_id"], segment_reaches[-1]["reach_id"], segmentation_method)
            bounds_dataset.to_file(fname)
            print("saved to %s" % fname)
            #choice = input()
            
            print("-" * 80)
            print("")

        print("=" * 80)
        print("")
    
    print("=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print("- Number of reaches: %i (%i in SWORD)" % (sum_rtseg, sum_sword))
    reaches_type1_subset = reaches_dataset[reaches_dataset["type"] == 1]
    sum_reaches_type1 = reaches_type1_subset.shape[0]
    print("- Number of reaches of type 1 in SWORD: %i" % sum_reaches_type1)
    print("=" * 80)
    print("")
    
    return sum_rtseg, sum_sword, sum_reaches_type1

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Compute pre-launch segmentation on SWORD datasets")
    parser.add_argument("continentID", type=str, choices=["AF", "AS", "EU", "NA", "OC", "SA"],
                        help="Continent ID")
    parser.add_argument("-segmentation", dest="segmentation_method", type=str, choices=["baseline", "advanced"], 
                        default="baseline", help="Segmentation method")
    parser.add_argument("-lambda", dest="lambdac", type=float, default=5.0,
                        help="Segmentation characteristic length in km")
    parser.add_argument("-min-length", dest="min_length", type=float, default=1.0,
                        help="Minimal reach length in km")
    parser.add_argument("-sword-dir", dest="sword_dir", type=str, default="$SWORD_DIR",
                        help="Path to the SWORD directory (default: from environnement variable SWORD_DIR)")
    args = parser.parse_args()
    
    #reaches_fname = "/home/kevin/Documents/SWOT/DAWG/CONFLUENCE/Verify/OhioVerify_V2/input/sword/na_sword_reaches_hb74_v1.shp"
    #nodes_fname = "/home/kevin/Documents/SWOT/DAWG/CONFLUENCE/Verify/OhioVerify_V2/input/sword/na_sword_nodes_hb74_v1.shp"
    
    sword_dir = os.path.expandvars(args.sword_dir)
    continent = args.continentID
    
    nodes_files = glob.glob(os.path.join(sword_dir, continent, "%s_sword_nodes_*.shp" % continent.lower()))
    
    sum_rtseg = 0
    sum_sword = 0
    sum_type1 = 0

    fout1 = open("out/sword_reaches_lengths.csv", "w")
    fout1.write("Subset;Segment;Reach;Length\n")
    fout1.close()
    fout2 = open("out/rtseg_reaches_lengths.csv", "w")
    fout2.write("Subset;Segment;Reach;Length\n")
    fout2.close()
    
    for i in range(0, len(nodes_files)):
        
        nodes_fname = nodes_files[i]
        reaches_fname = nodes_fname.replace("nodes", "reaches")
        
        #nodes_fname = "/home/kevin/Documents/SWOT/CNES/DATA/SWORD/shp/EU/eu_sword_nodes_hb2%i_v2.shp" % i
        #reaches_fname = "/home/kevin/Documents/SWOT/CNES/DATA/SWORD/shp/EU/eu_sword_reaches_hb2%i_v2.shp" % i

        print("*" * 80)
        print(" DATASET %1i" % (i+1))
        print("*" * 80)
        
        res = run_method(i, reaches_fname, nodes_fname, segmentation_method=args.segmentation_method, lc=args.lambdac, min_length=args.min_length)
        
        sum_rtseg += res[0]
        sum_sword += res[1]
        sum_type1 += res[2]
        
    
    print("=" * 80)
    print(" TOTAL")
    print("=" * 80)
    print("- Number of reaches: %i (%i in SWORD / %i type 1)" % (sum_rtseg, sum_sword, sum_type1))
    print("=" * 80)
    print("")
        
            

