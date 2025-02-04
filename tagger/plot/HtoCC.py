import os, json
from argparse import ArgumentParser

from qkeras.utils import load_qmodel
import awkward as ak
import numpy as np
import uproot
import hist
from hist import Hist

import matplotlib.pyplot as plt
import matplotlib
import mplhep as hep


#Interpolation of working point
from scipy.interpolate import interp1d

#Imports from other modules
import tagger.plot.style as style
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, get_bar_patch_data

style.set_style()

def nn_score_sum(model, jet_nn_inputs, indices, n_jets=2):

    #Get the inputs for the first n_jets
    btag_inputs = [np.asarray(jet_nn_inputs[:, i]) for i in range(0, n_jets)]

    #Get the nn outputs
    nn_outputs = [model.predict(nn_input) for nn_input in btag_inputs]

    #Sum them together
    b_index = indices['b']
    c_index = indices['charm']
    # bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])
    cscore_sum = sum([pred_score[0][:, c_index] for pred_score in nn_outputs])
    # bcscore_sum = bscore_sum + cscore_sum

    return cscore_sum

def pick_and_plot(rate_list, ht_list, nn_list, model_dir, target_rate = 14):
    """
    Pick the working points and plot
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/cc')
    os.makedirs(plot_dir, exist_ok=True)

    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    im = ax.scatter(nn_list, ht_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'cc rate [kHZ]')

    ax.set_ylabel(r"HT [GeV]")
    ax.set_xlabel(r"$\sum_{2~leading~jets}$ c scores")

    ax.set_xlim([0,2.5])
    ax.set_ylim([10,500])

    #plus, minus range
    RateRange = 0.5

    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Get the coordinates
    target_rate_NN = [nn_list[i] for i in target_rate_idx] # NN cut dimension
    target_rate_HT = [ht_list[i] for i in target_rate_idx] # HT cut dimension

    # Create an interpolation function
    interp_func = interp1d(target_rate_HT, target_rate_NN, kind='linear', fill_value='extrapolate')

    # Interpolate the NN value for the desired HT
    working_point_NN = interp_func(WPs_CMSSW['btag_l1_ht'] / 2)

    # Export the working point
    working_point = {"HT": WPs_CMSSW['btag_l1_ht'] / 2, "NN": float(working_point_NN)}
    with open(os.path.join(plot_dir, "working_point.json"), "w") as f:
        json.dump(working_point, f, indent=4)

    ax.plot(target_rate_NN, target_rate_HT,
                linewidth=5,
                color ='firebrick',
                label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    ax.legend(loc='upper right')
    plt.savefig(f"{plot_dir}/cc_rate.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/cc_rate.png", bbox_inches='tight')

def derive_cc_WPs(model_dir, minbias_path, target_rate=14, n_entries=100, tree='outnano/Jets'):
    """
    Derive the HH->4b working points
    """

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    #Load input/ouput variables of the NN
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_inputs, num_elements=2)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_nn_inputs = grouped_arrays

    #btag input list for first 2 jets
    nn_outputs = [model.predict(np.asarray(jet_nn_inputs[:, i])) for i in range(0,2)]

    #Calculate the output sum
    b_index = class_labels['b']
    c_index = class_labels['charm']
    # bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])
    cscore_sum = sum([pred_score[0][:, c_index] for pred_score in nn_outputs])
    # bcscore_sum = bscore_sum + cscore_sum
    ht = ak.sum(jet_pt, axis=1)

    assert(len(cscore_sum) == len(ht))

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(0,250,1)) + [5000] #Make sure to capture everything
    NN_edges = list([round(i,3) for i in np.arange(0, 1.25, 0.005)]) + [2.0]

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = ht, nn = cscore_sum)

    #Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []
    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:

            #Calculate the rate
            rate = RateHist[{"ht": slice(ht*1j, None, sum)}][{"nn": slice(NN*1.00j, None, sum)}]/n_events
            rate_list.append(rate*MINBIAS_RATE)

            #Append the results
            ht_list.append(ht)
            nn_list.append(NN)

    #Pick target rate and plot it
    pick_and_plot(rate_list, ht_list, nn_list, model_dir, target_rate=target_rate)

    return

def cc_eff_HT(model_dir, signal_path, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    ht_egdes = list(np.arange(0,800,20))
    ht_axis = hist.axis.Variable(ht_egdes, name = r"$HT^{gen}$")

    #Working points for CMSSW
    cmssw_btag = WPs_CMSSW['btag'] / 2
    cmssw_btag_ht =  WPs_CMSSW['btag_l1_ht'] / 2

    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, "plots/physics/cc/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        btag_wp = WPs['NN']
        btag_ht_wp = WPs['HT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    #Load the signal data
    signal = uproot.open(signal_path)[tree]

    # Calculate the truth HT
    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt_phys', n_entries)
    raw_cmssw_bscore = extract_array(signal, 'jet_bjetscore', n_entries)

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    raw_inputs = extract_nn_inputs(signal, input_vars, n_entries=n_entries)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_genpt, raw_jet_pt, raw_cmssw_bscore, raw_inputs, num_elements=2)
    jet_genpt, jet_pt, cmssw_bscore, jet_nn_inputs = grouped_arrays

    #Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    #B score from cmssw emulator
    cmsssw_bscore_sum = ak.sum(cmssw_bscore[:,:2], axis=1) #Only sum up the first two
    model_score_sum = nn_score_sum(model, jet_nn_inputs, class_labels)

    cmssw_selection = (jet_ht > cmssw_btag_ht) & (cmsssw_bscore_sum > cmssw_btag)
    model_selection = (jet_ht > btag_ht_wp) & (model_score_sum > btag_wp)

    #PLot the efficiencies
    #Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    cmssw_selected_events = Hist(ht_axis)
    model_selected_events = Hist(ht_axis)

    all_events.fill(jet_genht)
    cmssw_selected_events.fill(jet_genht[cmssw_selection])
    model_selected_events.fill(jet_genht[model_selection])

    #Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)

    #Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)

    #Now plot all
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    ax.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, c=style.color_cycle[0], fmt='o', linewidth=3, label=r'btag CMSSW Emulator @ 14 kHz (L1 $HT$ > {} GeV, $\sum$ bb > {})'.format(cmssw_btag_ht, cmssw_btag))
    ax.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3, label=r'Multiclass @ 14 kHz (L1 $HT$ > {} GeV, $\sum$ cc > {})'.format(btag_ht_wp, round(btag_wp,2)))

    #Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.1])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(H $\to$ cc trigger rate at 14 kHz)")
    plt.legend(loc='upper left')

    #Save plot
    plot_path = os.path.join(model_dir, "plots/physics/cc/HH_eff_HT")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')
    plt.show(block=False)


if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python HtoCC.py --deriveWPs
    2. Run efficiency based on the derived working points: python HtoCC.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-s', '--sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/VBFHToCC_PU200.root' , help = 'Signal sample for VBF->H->cc')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for c-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF->H->cc')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=1000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    if args.deriveWPs:
        derive_cc_WPs(args.model_dir, args.minbias, n_entries=args.n_entries)
    elif args.eff:
        cc_eff_HT(args.model_dir, args.sample, n_entries=args.n_entries)
