import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import os
import tqdm
from shapely.ops import substring
import sys

from rt_river_segmentation.denoising import hydraulic_filtering
from rt_river_segmentation.segmentation_methods import *


def get_product_data(data_dir, product, source="nominal", debug=False):
    
    reaches_fname = os.path.join(data_dir, product, "riverobs_%s_20201105" % source, "river_data", "reaches.shp")
    nodes_fname = os.path.join(data_dir, product, "riverobs_%s_20201105" % source, "river_data", "nodes.shp")
    
    if debug: 
        print("Load reaches dataset")
    reaches_dataset = gpd.read_file(reaches_fname)
    
    if debug: 
        print("Load nodes dataset")
    nodes_dataset = gpd.read_file(nodes_fname)
        
    return reaches_dataset, nodes_dataset


def get_segment(reaches_dataset, upstream_reach, debug=False):
    
    index = reaches_dataset[reaches_dataset["reach_id"] == upstream_reach].index[0]
    current_reach = reaches_dataset.iloc[index, :]
    reaches = [current_reach]
    if debug:
        print("Start reach: %s" % current_reach["reach_id"])
    while current_reach["n_reach_dn"] == 1:
        next_reach_id = current_reach["rch_id_dn"].split(" ")[0]
        next_index = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].index
        if next_index.size == 0:
            break
        next_reach = reaches_dataset[reaches_dataset["reach_id"] == next_reach_id].iloc[0]
        if debug:
            print("- Next reach: %s" % next_reach_id)
        if next_reach["n_reach_up"] > 1:
            break
        reaches.append(next_reach)
        current_reach = next_reach
        
    return reaches
        
        
def run_segmentation(x, heights, widths, lc=5.0, segmentation_method="baseline"):
    
    lc *= 1000.0
    
    xc = x[np.logical_and(np.isfinite(heights), np.isfinite(widths))]
    Z = heights[np.logical_and(np.isfinite(heights), np.isfinite(widths))]
    W = widths[np.logical_and(np.isfinite(heights), np.isfinite(widths))]
    
    x0 = x[0] - xc
    
    pas = 25
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
    
    if False:
        plt.plot(xi, Zi, "b-+")
        plt.plot(x0, Z, "r--")
        plt.show()

    if segmentation_method == "baseline":
        d2xZ_pos, d2xZ_neg, segmentation_bounds, wavelet = segmentation_baseline(xi, Zi, None, lc, min_length=1000.0)
    else:
        _, _, _, _, segmentation_bounds, wavelet_Z, wavelet_W = segmentation_advanced(xi, Zi, None, Wi, None, lc, min_length=1000.0)
        
    if True:
        plt.plot(x0, Z, "r--")
        plt.plot(xi, Zi, "b-")
        for bound in segmentation_bounds:
            plt.axvline(bound, color="k", ls="--")
        plt.xlabel("x* (km)")
        plt.xlabel("H (m)")
        plt.show()

    # Retransform segmentation bounds to outlet distances
    segmentation_bounds = x[0] - segmentation_bounds
    
    return segmentation_bounds

        
        
def compute_geometry(reaches_data, segmentation_bounds):
    
    ir = 0
    
    # Sort reaches by decreasing outlet distances
    reaches_data_sorted = reaches_data.sort_values(by=["p_dist_out"], ascending=False)
    
    xup = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "p_dist_out"]
    reach_length = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "p_length"]
    xdn = xup - reach_length
    reach_geometry = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "geometry"]
    
    bound_points = []
    for bound in segmentation_bounds:
        
        while xdn > bound:
            ir += 1
            xup = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "p_dist_out"]
            reach_length = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "p_length"]
            xdn = xup - reach_length
            reach_geometry = reaches_data_sorted.loc[reaches_data_sorted.index[ir], "geometry"]
            
        alpha = (bound - xdn) / (xup - xdn)
        bound_point = substring(reach_geometry, start_dist=alpha, end_dist=alpha, normalized=True)
        bound_points.append(bound_point)
        
    bounds_dataset = gpd.GeoDataFrame(data={"id" : np.arange(0, len(segmentation_bounds))}, geometry=bound_points)
    
    return bounds_dataset


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("Compute segmentation on Sacramento SWOTHR dataset")
    parser.add_argument("data_dir", type=str, 
                        help="Path to the directory of the Sacramento SWOTHR dataset")
    parser.add_argument("-product", type=str, choices=["truth", "nominal"], default="nominal",
                        help="Data product on which segmentation is to be performed")
    args = parser.parse_args()
    
    data_dir = args.data_dir
    segment1 = ["77449300011", "77449100221", "77449100211", "77449100201", "77449100191", "77449100181", 
                "77449100171", "77449100161", "77449100151", "77449100141", "77449100131", "77449100121",
                "77449100111", "77449100101", "77449100091", "77449100081", "77449100071", "77449100061",
                "77449100051", "77449100041", "77449100031", "77449100021", "77449100011", "77445000021"]
    segment = ["77449100161", "77449100151", "77449100141", "77449100131", "77449100121",
               "77449100111", "77449100101", "77449100091"]

    # Create output directory is necessary
    if not os.path.isdir("out"):
        os.mkdir("out")
    
    # List all products
    products = os.listdir(data_dir)
    print(products)
    
    # Plot all flow lines (segment 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for index, product in enumerate(products):
        
        reaches_data, nodes_data = get_product_data(data_dir, product, source=args.product)
        
        segment_reaches_data = reaches_data[reaches_data["reach_id"].isin(segment1)]
        segment_nodes_data = nodes_data[nodes_data["reach_id"].isin(segment1)]
        indices = segment_nodes_data[segment_nodes_data["wse"] < -9999].index
        segment_nodes_data.loc[indices, "wse"] = np.nan
        
        H = segment_nodes_data["wse"].values
        H[H < -9999.0] = np.nan
        W = segment_nodes_data["width"].values
        W[W < -9999.0] = np.nan
        
        l1, = ax1.plot(segment_nodes_data["p_dist_out"] * 0.001, H)
        ax2.plot(segment_nodes_data["p_dist_out"] * 0.001, W, color=l1.get_color())
        
    segment_reach_data = reaches_data[reaches_data["reach_id"] == segment[0]]
    xr = segment_reach_data["p_dist_out"].iloc[0]
    ax1.axvline(xr * 0.001, c="k", ls="--")
    ax2.axvline(xr * 0.001, c="k", ls="--")
    segment_reach_data = reaches_data[reaches_data["reach_id"] == segment[-1]]
    xr = segment_reach_data["p_dist_out"].iloc[0] - segment_reach_data["p_length"].iloc[0]
    ax1.axvline(xr * 0.001, c="k", ls="--")
    ax2.axvline(xr * 0.001, c="k", ls="--")
    ax1.set_ylabel("H (m)")
    ax2.set_ylabel("W (m)")
    ax2.set_xlabel("x (km)")
    plt.tight_layout()
    plt.savefig("out/sacramento_flow_lines.png")
    plt.close(plt.gcf())
    
    dataH = None
    dataW = None
    for index, product in enumerate(products):
        
        reaches_data, nodes_data = get_product_data(data_dir, product, source=source)
        
        segment_reaches_data = reaches_data[reaches_data["reach_id"].isin(segment)]
        segment_nodes_data = nodes_data[nodes_data["reach_id"].isin(segment)]
        indices = segment_nodes_data[segment_nodes_data["wse"] < -9999].index
        segment_nodes_data.loc[indices, "wse"] = np.nan
        
        if dataH is None:
            dataH = np.zeros((len(products), segment_nodes_data.shape[0]))
            dataW = np.zeros((len(products), segment_nodes_data.shape[0]))
        dataH[index, :] = segment_nodes_data["wse"].values
        dataH[index, dataH[index, :] < -9999.0] = np.nan
        dataW[index, :] = segment_nodes_data["width"].values
        dataW[index, dataW[index, :] < -9999.0] = np.nan
        #segment_nodes_data[segment_nodes_data["wse"] < -9999] = np.nan
        
        print("product %s: Hdn=" % product, segment_nodes_data["wse"].values[:30])
        if segment_nodes_data["wse"].values[-1] == np.nan:
            print("- Removing product %s (partial)")
        
    # Select flow lines (low flow for heights, median for width)
    flowlines_sorted = np.argsort(dataH[:, 0])
    low_flow_index = flowlines_sorted[len(products)*10//100]
    median_flow_index = flowlines_sorted[len(products)//2]
    
    H = dataH[low_flow_index, :]
    W = dataW[median_flow_index, :]
    
    if True:
        fig, axes = plt.subplots(2, 1, sharex=True)
        axes[0].plot(segment_nodes_data["p_dist_out"] * 0.001, dataH.T, color="grey", alpha=0.5)
        axes[0].plot(segment_nodes_data["p_dist_out"] * 0.001, H, color="blue", label="selected")
        axes[1].plot(segment_nodes_data["p_dist_out"] * 0.001, dataW.T, color="grey", alpha=0.5)
        axes[1].plot(segment_nodes_data["p_dist_out"] * 0.001, W, color="blue", label="selected")
        axes[0].set_ylabel("H (m)")
        axes[1].set_xlabel("x (km)")
        axes[1].set_ylabel("W (m)")
        plt.tight_layout()
        plt.show()
    
    x = segment_nodes_data["p_dist_out"].values[::-1]
    H = H[::-1]
    W = W[::-1]
    segmentation_bounds = run_segmentation(x, H, W, segmentation_method="baseline")
    print("segmentation_bounds=", segmentation_bounds)
    
    segment_reaches_data = segment_reaches_data.sort_values(by="p_dist_out", ascending=False)
    bounds_dataset = compute_geometry(segment_reaches_data, segmentation_bounds)
    bounds_dataset.to_file("out/sacramento_%s_segment_bounds.shp" % source)
