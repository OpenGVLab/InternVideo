import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def draw_curves():
    fig = plt.figure(figsize=(8,5))
    plt.rcParams["font.family"] = "Arial"
    fontsize = 15
    markersize = 80

    # I3D
    I3D_DNN_HMDB = [94.69, 75.07]  # (closed-set ACC, open-set AUC)
    I3D_DNN_MiT = [94.69, 79.14]
    I3D_DEAR_HMDB = [94.34, 77.08]
    I3D_DEAR_MiT = [94.34, 81.54]
    # TSM
    TSM_DNN_HMDB = [95.11, 73.85]
    TSM_DNN_MiT = [95.11, 78.35]
    TSM_DEAR_HMDB = [94.45, 78.65]
    TSM_DEAR_MiT = [94.45, 83.92]
    # TPN
    TPN_DNN_HMDB = [95.41, 74.13]
    TPN_DNN_MiT = [95.41, 77.76]
    TPN_DEAR_HMDB = [96.42, 79.23]
    TPN_DEAR_MiT = [96.42, 81.80]
    # SlowFast
    SlowFast_DNN_HMDB = [96.72, 75.41]
    SlowFast_DNN_MiT = [96.72, 78.49]
    SlowFast_DEAR_HMDB = [96.66, 82.94]
    SlowFast_DEAR_MiT = [96.66, 86.99]

    # Line: DNN for HMDB
    plt.plot([I3D_DNN_HMDB[0], TSM_DNN_HMDB[0], TPN_DNN_HMDB[0], SlowFast_DNN_HMDB[0]],
             [I3D_DNN_HMDB[1], TSM_DNN_HMDB[1], TPN_DNN_HMDB[1], SlowFast_DNN_HMDB[1]], 'r-', linewidth=2, label='HMDB')
    # Line: DEAR for HMDB
    plt.plot([I3D_DEAR_HMDB[0], TSM_DEAR_HMDB[0], TPN_DEAR_HMDB[0], SlowFast_DEAR_HMDB[0]],
             [I3D_DEAR_HMDB[1], TSM_DEAR_HMDB[1], TPN_DEAR_HMDB[1], SlowFast_DEAR_HMDB[1]], 'r-', linewidth=2)
    # Line: DNN for MiT
    plt.plot([I3D_DNN_MiT[0], TSM_DNN_MiT[0], TPN_DNN_MiT[0], SlowFast_DNN_MiT[0]],
             [I3D_DNN_MiT[1], TSM_DNN_MiT[1], TPN_DNN_MiT[1], SlowFast_DNN_MiT[1]], 'b-', linewidth=2, label='MiT')
    # Line: DEAR for MiT
    plt.plot([I3D_DEAR_MiT[0], TSM_DEAR_MiT[0], TPN_DEAR_MiT[0], SlowFast_DEAR_MiT[0]],
             [I3D_DEAR_MiT[1], TSM_DEAR_MiT[1], TPN_DEAR_MiT[1], SlowFast_DEAR_MiT[1]], 'b-', linewidth=2)
    

    # Draw all I3D points
    # HMDB
    plt.scatter(I3D_DNN_HMDB[0], I3D_DNN_HMDB[1], marker='^', s=markersize, color='r', label='Dropout BALD')
    plt.text(I3D_DNN_HMDB[0], I3D_DNN_HMDB[1], 'I3D', fontsize=fontsize)
    plt.scatter(I3D_DEAR_HMDB[0], I3D_DEAR_HMDB[1], marker='*', s=markersize, color='r', label='DEAR EU')
    plt.text(I3D_DEAR_HMDB[0], I3D_DEAR_HMDB[1], 'I3D', fontsize=fontsize)
    plt.plot([I3D_DNN_HMDB[0], I3D_DEAR_HMDB[0]], [I3D_DNN_HMDB[1], I3D_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # plt.arrow(I3D_DNN_HMDB[0]+1, I3D_DNN_HMDB[1], I3D_DEAR_HMDB[0]-I3D_DNN_HMDB[0]-2, I3D_DEAR_HMDB[1]-I3D_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    # # MiT
    plt.scatter(I3D_DNN_MiT[0], I3D_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(I3D_DNN_MiT[0], I3D_DNN_MiT[1], 'I3D', fontsize=fontsize)
    plt.scatter(I3D_DEAR_MiT[0], I3D_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(I3D_DEAR_MiT[0], I3D_DEAR_MiT[1], 'I3D', fontsize=fontsize)
    plt.plot([I3D_DNN_MiT[0], I3D_DEAR_MiT[0]], [I3D_DNN_MiT[1], I3D_DEAR_MiT[1]], 'b--', linewidth=0.5)
    # plt.arrow(I3D_DNN_MiT[0]+1, I3D_DNN_MiT[1], I3D_DEAR_MiT[0]-I3D_DNN_MiT[0]-3, I3D_DEAR_MiT[1]-I3D_DNN_MiT[1]-2,head_width=0.8, fc='grey',ec='grey', head_length=0.8)

    # Draw all TSM points
    # HMDB
    plt.scatter(TSM_DNN_HMDB[0], TSM_DNN_HMDB[1], marker='^', s=markersize, color='r')
    plt.text(TSM_DNN_HMDB[0], TSM_DNN_HMDB[1], 'TSM', fontsize=fontsize)
    plt.scatter(TSM_DEAR_HMDB[0], TSM_DEAR_HMDB[1], marker='*', s=markersize, color='r')
    plt.text(TSM_DEAR_HMDB[0], TSM_DEAR_HMDB[1], 'TSM', fontsize=fontsize)
    plt.plot([TSM_DNN_HMDB[0], TSM_DEAR_HMDB[0]], [TSM_DNN_HMDB[1], TSM_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # plt.arrow(TSM_DNN_HMDB[0]+1, TSM_DNN_HMDB[1], TSM_DEAR_HMDB[0]-TSM_DNN_HMDB[0]-2, TSM_DEAR_HMDB[1]-TSM_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    # # MiT
    plt.scatter(TSM_DNN_MiT[0], TSM_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(TSM_DNN_MiT[0], TSM_DNN_MiT[1], 'TSM', fontsize=fontsize)
    plt.scatter(TSM_DEAR_MiT[0], TSM_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(TSM_DEAR_MiT[0], TSM_DEAR_MiT[1], 'TSM', fontsize=fontsize)
    plt.plot([TSM_DNN_MiT[0], TSM_DEAR_MiT[0]], [TSM_DNN_MiT[1], TSM_DEAR_MiT[1]], 'b--', linewidth=0.5)

    # Draw all TPN points
    # HMDB
    plt.scatter(TPN_DNN_HMDB[0], TPN_DNN_HMDB[1], marker='^', s=markersize, color='r')
    plt.text(TPN_DNN_HMDB[0], TPN_DNN_HMDB[1], 'TPN', fontsize=fontsize)
    plt.scatter(TPN_DEAR_HMDB[0], TPN_DEAR_HMDB[1], marker='*', s=markersize, color='r')
    plt.text(TPN_DEAR_HMDB[0], TPN_DEAR_HMDB[1], 'TPN', fontsize=fontsize)
    plt.plot([TPN_DNN_HMDB[0], TPN_DEAR_HMDB[0]], [TPN_DNN_HMDB[1], TPN_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # plt.arrow(TPN_DNN_HMDB[0]+1, TPN_DNN_HMDB[1], TPN_DEAR_HMDB[0]-TPN_DNN_HMDB[0]-2, TPN_DEAR_HMDB[1]-TPN_DNN_HMDB[1]-1,head_width=0.8, fc='skyblue',ec='skyblue', head_length=0.8)
    plt.scatter(TPN_DNN_MiT[0], TPN_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(TPN_DNN_MiT[0], TPN_DNN_MiT[1], 'TPN', fontsize=fontsize)
    plt.scatter(TPN_DEAR_MiT[0], TPN_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(TPN_DEAR_MiT[0], TPN_DEAR_MiT[1], 'TPN', fontsize=fontsize)
    plt.plot([TPN_DNN_MiT[0], TPN_DEAR_MiT[0]], [TPN_DNN_MiT[1], TPN_DEAR_MiT[1]], 'b--', linewidth=0.5)

    # Draw all SlowFast points
    # HMDB
    plt.scatter(SlowFast_DNN_HMDB[0], SlowFast_DNN_HMDB[1], marker='^', s=markersize, color='r')
    plt.text(SlowFast_DNN_HMDB[0], SlowFast_DNN_HMDB[1], 'SlowFast', fontsize=fontsize)
    plt.scatter(SlowFast_DEAR_HMDB[0], SlowFast_DEAR_HMDB[1], marker='*', s=markersize, color='r')
    plt.text(SlowFast_DEAR_HMDB[0], SlowFast_DEAR_HMDB[1], 'SlowFast', fontsize=fontsize)
    plt.plot([SlowFast_DNN_HMDB[0], SlowFast_DEAR_HMDB[0]], [SlowFast_DNN_HMDB[1], SlowFast_DEAR_HMDB[1]], 'r--', linewidth=0.5)
    # MiT
    plt.scatter(SlowFast_DNN_MiT[0], SlowFast_DNN_MiT[1], marker='^', s=markersize, color='b')
    plt.text(SlowFast_DNN_MiT[0], SlowFast_DNN_MiT[1], 'SlowFast', fontsize=fontsize)
    plt.scatter(SlowFast_DEAR_MiT[0], SlowFast_DEAR_MiT[1], marker='*', s=markersize, color='b')
    plt.text(SlowFast_DEAR_MiT[0], SlowFast_DEAR_MiT[1], 'SlowFast', fontsize=fontsize)
    plt.plot([SlowFast_DNN_MiT[0], SlowFast_DEAR_MiT[0]], [SlowFast_DNN_MiT[1], SlowFast_DEAR_MiT[1]], 'b--', linewidth=0.5)
    
    plt.xlim(94, 97.5)
    plt.ylim(65, 90)
    plt.xlabel('Closed-Set Accuracy (%)', fontsize=fontsize)
    plt.ylabel('Open-Set AUC Score (%)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc='lower left', fontsize=fontsize)
    plt.grid('on', linestyle='--')
    plt.tight_layout()

    plt.savefig('../temp/compare_gain.png')
    plt.savefig('../temp/compare_gain.pdf')


def draw_one_curve(data_dict, markers, markercolor='g', markersize=80, fontsize=10, label='I3D', linestyle='g-', add_marker_text=False, text_offset=[0,0]):

    sorted_dict = dict(sorted(data_dict.items(),key=lambda x:x[1][1]))

    x_data, y_data = [], []
    for k, v in sorted_dict.items():
        x_data.append(v[1])
        y_data.append(v[0])
        # Marker: OpenMax
        if k == 'DEAR (Ours)':
            plt.scatter(v[1], v[0], marker=markers[k], s=markersize*4, color=markercolor)
        else:
            plt.scatter(v[1], v[0], marker=markers[k], s=markersize, color=markercolor)
        if add_marker_text:
            # plt.text(v[1] + text_offset[0], v[0]+ text_offset[1], k, fontsize=fontsize)
            pass
    # Line: I3D for MiT
    line_hd, = plt.plot(x_data, y_data, linestyle, linewidth=2, label=label, markersize=1)
    return line_hd
    

def draw_mit_curves():
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams["font.family"] = "Arial"
    fontsize = 25
    markersize = 80

    # (open maF1, open-set AUC)
    # I3D
    I3D_OpenMax = [66.22, 77.76]
    I3D_Dropout = [68.11, 79.14]
    I3D_BNNSVI = [68.65, 79.50]
    I3D_SoftMax = [68.84, 79.94]
    I3D_RPL = [68.11, 79.16]
    I3D_DEAR = [69.98, 81.54]
    # TSM
    TSM_OpenMax = [71.81, 83.05]
    TSM_Dropout = [65.32, 78.35]
    TSM_BNNSVI = [64.28, 77.39]
    TSM_SoftMax = [71.68, 82.38]
    TSM_RPL = [63.92, 77.28]
    TSM_DEAR = [70.15, 83.92]
    # TPN
    TPN_OpenMax = [64.80, 76.26]
    TPN_Dropout = [65.77, 77.76]
    TPN_BNNSVI = [61.40, 75.32]
    TPN_SoftMax = [70.82, 81.35]
    TPN_RPL = [66.21, 78.21]
    TPN_DEAR = [71.18, 81.80]
    # SlowFast
    SlowFast_OpenMax = [72.48, 80.62]
    SlowFast_Dropout = [67.53, 78.49]
    SlowFast_BNNSVI = [65.22, 77.39]
    SlowFast_SoftMax = [74.42, 82.88]
    SlowFast_RPL = [66.33, 77.42]
    SlowFast_DEAR = [77.28, 86.99]

    markers = {'DEAR (Ours)': '*', 'SoftMax': 'o', 'OpenMax': '^', 'RPL': 'd', 'MC Dropout': 's', 'BNN SVI': 'P'}
    
    # Line: I3D for MiT
    data_dict = {'OpenMax': I3D_OpenMax, 'MC Dropout': I3D_Dropout, 'BNN SVI': I3D_BNNSVI, 'SoftMax': I3D_SoftMax, 'RPL': I3D_RPL, 'DEAR (Ours)': I3D_DEAR}
    line1_hd = draw_one_curve(data_dict, markers=markers, markercolor='g', markersize=markersize, fontsize=fontsize, label='I3D', linestyle='g-')

    data_dict = {'OpenMax': TSM_OpenMax, 'MC Dropout': TSM_Dropout, 'BNN SVI': TSM_BNNSVI, 'SoftMax': TSM_SoftMax, 'RPL': TSM_RPL, 'DEAR (Ours)': TSM_DEAR}
    line2_hd = draw_one_curve(data_dict, markers=markers, markercolor='k', markersize=markersize, fontsize=fontsize, label='TSM', linestyle='k-')

    data_dict = {'OpenMax': TPN_OpenMax, 'MC Dropout': TPN_Dropout, 'BNN SVI': TPN_BNNSVI, 'SoftMax': TPN_SoftMax, 'RPL': TPN_RPL, 'DEAR (Ours)': TPN_DEAR}
    line3_hd = draw_one_curve(data_dict, markers=markers, markercolor='b', markersize=markersize, fontsize=fontsize, label='TPN', linestyle='b-')

    data_dict = {'OpenMax': SlowFast_OpenMax, 'MC Dropout': SlowFast_Dropout, 'BNN SVI': SlowFast_BNNSVI, 'SoftMax': SlowFast_SoftMax, 'RPL': SlowFast_RPL, 'DEAR (Ours)': SlowFast_DEAR}
    line4_hd = draw_one_curve(data_dict, markers=markers, markercolor='r', markersize=markersize, fontsize=fontsize, label='SlowFast', linestyle='r-', 
                    add_marker_text=True, text_offset=[-2.2, -0.2])
    
    marker_elements = []
    for k, v in markers.items():
        msize = 18 if k == 'DEAR (Ours)' else 12
        elem = Line2D([0], [0], marker=v, label=k, markersize=msize, linestyle="None")
        marker_elements.append(elem)
    marker_legend = ax.legend(handles=marker_elements, fontsize=fontsize-3, loc='lower right', ncol=1, handletextpad=0.05, columnspacing=0.05, borderaxespad=0.1)
    ax.add_artist(marker_legend)

    plt.ylim(60, 78)
    plt.xlim(75, 88)
    plt.ylabel('Open maF1 (%)', fontsize=fontsize)
    plt.xlabel('Open-Set AUC Score (%)', fontsize=fontsize)
    plt.xticks(np.arange(75, 89, 4), fontsize=fontsize)
    plt.yticks(np.arange(60, 79, 4), fontsize=fontsize)
    plt.legend(handles=[line1_hd, line2_hd, line3_hd, line4_hd], loc='upper left', fontsize=fontsize-3, handletextpad=0.5, borderaxespad=0.1)
    plt.title('MiT-v2 as Unknown', fontsize=fontsize)
    plt.grid('on', linestyle='--')
    plt.tight_layout()
    plt.savefig('../temp/compare_gain_mit.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig('../temp/compare_gain_mit.pdf', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)



def draw_hmdb_curves():
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams["font.family"] = "Arial"
    fontsize = 25
    markersize = 80

    # (open maF1, open-set AUC)
    # I3D
    I3D_OpenMax = [67.85, 74.34]
    I3D_Dropout = [71.13, 75.07]
    I3D_BNNSVI = [71.57, 74.66]
    I3D_SoftMax = [73.19, 75.68]
    I3D_RPL = [71.48, 75.20]
    I3D_DEAR = [77.24, 77.08]
    # TSM
    TSM_OpenMax = [74.17, 77.07]
    TSM_Dropout = [71.52, 73.85]
    TSM_BNNSVI = [69.11, 73.42]
    TSM_SoftMax = [78.27, 77.99]
    TSM_RPL = [69.34, 73.62]
    TSM_DEAR = [84.69, 78.65]
    # TPN
    TPN_OpenMax = [65.27, 74.12]
    TPN_Dropout = [68.45, 74.13]
    TPN_BNNSVI = [63.81, 72.68]
    TPN_SoftMax = [76.23, 77.97]
    TPN_RPL = [70.31, 75.32]
    TPN_DEAR = [81.79, 79.23]
    # SlowFast
    SlowFast_OpenMax = [73.57, 78.76]
    SlowFast_Dropout = [70.55, 75.41]
    SlowFast_BNNSVI = [69.19, 74.78]
    SlowFast_SoftMax = [78.04, 79.16]
    SlowFast_RPL = [68.32, 74.23]
    SlowFast_DEAR = [85.48, 82.94]

    markers = {'DEAR (Ours)': '*', 'SoftMax': 'o', 'OpenMax': '^', 'RPL': 'd', 'MC Dropout': 's', 'BNN SVI': 'P'}
    
    # Line: I3D for HMDB
    data_dict = {'OpenMax': I3D_OpenMax, 'MC Dropout': I3D_Dropout, 'BNN SVI': I3D_BNNSVI, 'SoftMax': I3D_SoftMax, 'RPL': I3D_RPL, 'DEAR (Ours)': I3D_DEAR}
    line1_hd = draw_one_curve(data_dict, markers=markers, markercolor='g', markersize=markersize, fontsize=fontsize, label='I3D', linestyle='g-')

    data_dict = {'OpenMax': TSM_OpenMax, 'MC Dropout': TSM_Dropout, 'BNN SVI': TSM_BNNSVI, 'SoftMax': TSM_SoftMax, 'RPL': TSM_RPL, 'DEAR (Ours)': TSM_DEAR}
    line2_hd = draw_one_curve(data_dict, markers=markers, markercolor='k', markersize=markersize, fontsize=fontsize, label='TSM', linestyle='k-')

    data_dict = {'OpenMax': TPN_OpenMax, 'MC Dropout': TPN_Dropout, 'BNN SVI': TPN_BNNSVI, 'SoftMax': TPN_SoftMax, 'RPL': TPN_RPL, 'DEAR (Ours)': TPN_DEAR}
    line3_hd = draw_one_curve(data_dict, markers=markers, markercolor='b', markersize=markersize, fontsize=fontsize, label='TPN', linestyle='b-')

    data_dict = {'OpenMax': SlowFast_OpenMax, 'MC Dropout': SlowFast_Dropout, 'BNN SVI': SlowFast_BNNSVI, 'SoftMax': SlowFast_SoftMax, 'RPL': SlowFast_RPL, 'DEAR (Ours)': SlowFast_DEAR}
    line4_hd = draw_one_curve(data_dict, markers=markers, markercolor='r', markersize=markersize, fontsize=fontsize, label='SlowFast', linestyle='r-', 
                    add_marker_text=True, text_offset=[0.2, -1.5])

    
    marker_elements = []
    for k, v in markers.items():
        msize = 18 if k == 'DEAR (Ours)' else 12
        elem = Line2D([0], [0], marker=v, label=k, markersize=msize, linestyle="None")
        marker_elements.append(elem)
    marker_legend = ax.legend(handles=marker_elements, fontsize=fontsize-3, loc='lower right', ncol=1, handletextpad=0.3, columnspacing=0.2, borderaxespad=0.1)
    ax.add_artist(marker_legend)

    plt.ylim(60, 88)
    plt.xlim(72, 85)
    plt.ylabel('Open maF1 (%)', fontsize=fontsize)
    plt.xlabel('Open-Set AUC Score (%)', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(handles=[line1_hd, line2_hd, line3_hd, line4_hd], loc='upper left', fontsize=fontsize-3, handletextpad=0.5, borderaxespad=0.1)
    plt.grid('on', linestyle='--')
    plt.title('HMDB-51 as Unknown', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('../temp/compare_gain_hmdb.png', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
    plt.savefig('../temp/compare_gain_hmdb.pdf', bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)


if __name__ == '__main__':

    models = ['I3D', 'TPN', 'TSM', 'SlowFast']

    # draw_curves()

    draw_mit_curves()
    draw_hmdb_curves()
