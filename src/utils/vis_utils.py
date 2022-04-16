import os
import numpy as np
import matplotlib.pyplot as plt
# no X forwarding on remote machine using ssh & screen & tmux
plt.switch_backend("agg")
from matplotlib import gridspec

from src.utils import utils, io_utils

try:
    import seaborn as sns
    sns.set_style("whitegrid", {"axes.grid": False})
except ImportError:
    print("Install seaborn to colorful visualization!")
except:
    print("Unknown error")

FONTSIZE = 4
plt.rc('xtick', labelsize=FONTSIZE)
plt.rc('ytick', labelsize=FONTSIZE)

""" helper functions for visualization """

def add_attention_to_figure(fig, gc, row, col, row_height, col_width, att,
                            x_labels=None, y_labels=None, aspect="auto", # "equal"
                            show_colorbar=False, vmin=None, vmax=None, yrotation=90):
    ax = fig.add_subplot(gc[row:row+row_height, col:col+col_width])
    iax = ax.imshow(att, interpolation="nearest", aspect=aspect,
                     cmap="binary", vmin=vmin, vmax=vmax)
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=90, fontsize=2)
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, rotation=yrotation, fontsize=FONTSIZE)
    else:
        ax.get_yaxis().set_visible(False)
    if show_colorbar:
        fig.colorbar(iax)
    #ax.grid()

def visualize_PLRN(config, vis_data, itow, prefix):

    # fetching data
    qids = vis_data["qids"]
    qr_label = vis_data["query_labels"] # [B, L_q] == [5,25]
    gt = vis_data["grounding_gt"] # [B, L_v] == [5,128]
    pred = vis_data["grounding_pred"] # [B, L_v]
    vid_nfeats = vis_data["nfeats"]
    watt_w = vis_data["se_attw"] # [B,A,Lq]
    matt_w = vis_data["t_attw"] # [B,A,Lv]
    gatt_w = vis_data["s_attw"] # [B,A,Lv]

    # constants
    B, num_seg = qr_label.shape[0], pred.shape[1]

    # prepare xaxis labels for visualization
    query = [utils.label2string(itow, qr_label[idx], end_idx=0).split(" ")
             for idx in range(B)]
    vid_idx = []
    for idx in range(B):
        if len(query[idx]) < qr_label.shape[1]:
            for i in range(qr_label.shape[1]-len(query[idx])):
                # add null token
                query[idx].append("-")
        vid_idx.append([])
        for i in range(num_seg):
            if i >= vid_nfeats[idx]: vlabel = "$"
            elif i % 10 == 0: vlabel = str(i)
            else: vlabel = ""
            vid_idx[idx].append(vlabel)


    # create figure
    figsize = [5, B] # (col, row)
    fig = plt.figure(figsize=figsize)
    gc = gridspec.GridSpec(figsize[1], figsize[0])

    # create figure
    ngates = [str(i+1) for i in range(gatt_w.shape[1])]
    for idx in range(B):
        # word attention weight
        add_attention_to_figure(fig, gc, idx, 0, 1, 1, watt_w[idx],
                                query[idx], ["watt"], show_colorbar=True)
        # MM attentive pooling
        add_attention_to_figure(fig, gc, idx, 1, 1, 1, matt_w[idx],
                                vid_idx[idx], ["matt"], show_colorbar=True)
        # MM attentive pooling
        add_attention_to_figure(fig, gc, idx, 2, 1, 1, gatt_w[idx][np.newaxis,:],
                                ngates, ["gatt"], show_colorbar=True)
        # localization
        add_attention_to_figure(fig, gc, idx, 3, 1, 1,
                                gt[idx][np.newaxis, :], vid_idx[idx], ["GT"])
        add_attention_to_figure(fig, gc, idx, 4, 1, 1,
                                pred[idx][np.newaxis, :], vid_idx[idx], ["Pred"])

    # save figure
    save_dir = os.path.join(config["misc"]["result_dir"], "qualitative", "Train")
    save_path = os.path.join(save_dir, prefix + "_qrn.png")
    io_utils.check_and_create_dir(save_dir)
    plt.tight_layout(pad=0.1, h_pad=0.1)
    plt.savefig(save_path, bbox_inches="tight", dpi=450)
    print("Visualization of LGI is saved in {}".format(save_path))
    plt.close()
