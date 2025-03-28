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
import tagger.plot.style as style

style.set_style()

#Interpolation of working point
from scipy.interpolate import interp1d

#Imports from other modules
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values, get_input_mask
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, get_bar_patch_data

def nn_bscore_sum(model, jet_nn_inputs, n_jets=4, b_index = 1):

    N_FILTERS = model.get_layer('avgpool').output_shape[1]

    #Get the inputs and masks for the first n_jets
    btag_inputs = [np.asarray(jet_nn_inputs[:, i]) for i in range(0, n_jets)]
    input_masks = [get_input_mask(btag_inp, N_FILTERS) for btag_inp in btag_inputs]
    nn_inputs = [[inp, mask] for inp, mask in zip(btag_inputs, input_masks)]

    #Get the nn outputs
    nn_outputs = [model.predict(nn_inp) for nn_inp in nn_inputs]

    #Sum them together
    bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])

    return bscore_sum

def pick_and_plot(rate_list, ht_list, nn_list, model_dir, target_rate = 14):
    """
    Pick the working points and plot
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/bbbb')
    os.makedirs(plot_dir, exist_ok=True)

    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    im = ax.scatter(nn_list, ht_list, c=rate_list, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'4-b rate [kHZ]')

    ax.set_ylabel(r"HT [GeV]")
    ax.set_xlabel(r"$\sum_{4~leading~jets}$ b scores")
<<<<<<< HEAD

    ax.set_xlim([0, 1.3])
=======

    ax.set_xlim([0,2.5])
>>>>>>> masking_model
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
    working_point_NN = interp_func(WPs_CMSSW['btag_l1_ht'])

    # Export the working point
    working_point = {"HT": WPs_CMSSW['btag_l1_ht'], "NN": float(working_point_NN)}
    with open(os.path.join(plot_dir, "working_point.json"), "w") as f:
        json.dump(working_point, f, indent=4)

    ax.plot(target_rate_NN, target_rate_HT,
                linewidth=5,
                color ='firebrick',
                label = r"${} \pm {}$ kHz".format(target_rate, RateRange))

    ax.legend(loc='upper right')
    plt.savefig(f"{plot_dir}/bbbb_rate.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/bbbb_rate.png", bbox_inches='tight')

<<<<<<< HEAD
def derive_HT_WP(RateHist, ht_edges, n_events, model_dir, target_rate = 14, RateRange=0.8):
    """
    Derive the HT only working points (without bb cuts)
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/bbbb')

    #Derive the rate
    rate_list = []
    ht_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:

        #Calculate the rate
        rate = RateHist[{"ht": slice(ht*1j, None, sum)}][{"nn": slice(0.0j, None, sum)}]/n_events
        rate_list.append(rate*MINBIAS_RATE)

        #Append the results
        ht_list.append(ht)

    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Read WPs dict and add HT cut
    WP_json = os.path.join(plot_dir, "working_point.json")
    working_point = json.load(open(WP_json, "r"))
    working_point["ht_only_cut"] = float(ht_list[target_rate_idx[0]])
    json.dump(working_point, open(WP_json, "w"), indent=4)


def derive_bbbb_WPs(model_dir, minbias_path, target_rate=14, n_entries=100, tree='outnano/Jets'):
=======
def derive_bbbb_WPs(model_dir, minbias_path, target_rate=14, n_entries=100, tree='jetntuple/Jets'):
>>>>>>> masking_model
    """
    Derive the HH->4b working points
    """

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))
    N_FILTERS = model.get_layer('avgpool').output_shape[1]

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
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_nn_inputs = grouped_arrays

    #Btag input list for first 4 jets
    jet_inputs = [np.asarray(jet_nn_inputs[:, i]) for i in range(0, 4)]
    input_masks = [get_input_mask(jet_inp, N_FILTERS) for jet_inp in jet_inputs]
    nn_inputs = [[inp, mask] for inp, mask in zip(jet_inputs, input_masks)]
    nn_outputs = [model.predict(nn_inp) for nn_inp in nn_inputs]

    #Calculate the output sum
    b_index = class_labels['b']
    bscore_sum = sum([pred_score[0][:, b_index] for pred_score in nn_outputs])
    ht = ak.sum(jet_pt, axis=1)

    assert(len(bscore_sum) == len(ht))

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(0,500,2)) + [10000] #Make sure to capture everything
    NN_edges = list([round(i,2) for i in np.arange(0, 2.5, 0.01)]) + [4.0]

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = ht, nn = bscore_sum)

    #Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:

            #Calculate the rate
            rate = RateHist[{"ht": slice(ht*1j, None, sum)}][{"nn": slice(NN*1.0j, None, sum)}]/n_events
            rate_list.append(rate*MINBIAS_RATE)

            #Append the results
            ht_list.append(ht)
            nn_list.append(NN)

    #Pick target rate and plot it
    pick_and_plot(rate_list, ht_list, nn_list, model_dir, target_rate=target_rate)
    derive_HT_WP(RateHist, ht_edges, n_events, model_dir, target_rate=target_rate)

    return

def load_bbbb_WPs(model_dir):
    """
    Check and lodad all bbbb working points
    """

    #Check if the working point have been derived
    WP_path = os.path.join(model_dir, "plots/physics/bbbb/working_point.json")

    #Get derived working points
    if os.path.exists(WP_path):
        with open(WP_path, "r") as f:  WPs = json.load(f)
        btag_wp = WPs['NN']
        btag_ht_wp = WPs['HT']
        ht_only_wp = WPs['ht_only_cut']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    return btag_wp, btag_ht_wp, ht_only_wp

def bbbb_eff(model_dir, signal_path, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    ht_edges = list(np.arange(0,800,20))
    ht_axis = hist.axis.Variable(ht_edges, name = r"$HT^{gen}$")

    #Working points for CMSSW
    cmssw_btag = WPs_CMSSW['btag']
    cmssw_btag_ht =  WPs_CMSSW['btag_l1_ht']

    btag_wp, btag_ht_wp, ht_only_wp =  load_bbbb_WPs(model_dir)

    #Load the signal data
    signal = uproot.open(signal_path)[tree]

    # Calculate the truth HT
    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt_phys', n_entries)
    raw_cmssw_bscore = extract_array(signal, 'jet_bjetscore', n_entries)

    # Try to extract genHH_mass, set to None if not found
    try:
        raw_gen_mHH = extract_array(signal, 'genHH_mass', n_entries)
    except (KeyError, ValueError, uproot.exceptions.KeyInFileError) as e:
        print(f"Warning: 'genHH_mass' not found in signal file: {e}")
        raw_gen_mHH = None

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)
    raw_inputs = extract_nn_inputs(signal, input_vars, n_entries=n_entries)

    #Group event_id, gen_mHH, and genpt separately
    if raw_gen_mHH is not None:
        event_id, grouped_gen_arrays = group_id_values(raw_event_id, raw_gen_mHH, raw_jet_genpt, num_elements=0)
        all_event_gen_mHH = ak.firsts(grouped_gen_arrays[0])
    else:
        # If genHH_mass doesn't exist, only group event_id and genpt
        event_id, grouped_gen_arrays = group_id_values(raw_event_id, raw_jet_genpt, num_elements=0)
        all_event_gen_mHH = None

    all_jet_genht = ak.sum(grouped_gen_arrays[-1], axis=1)  # Last element will always be genpt

    #Group these attributes by event id, and filter out groups that don't have at least 4 elements
    if raw_gen_mHH is not None:
        event_id, grouped_arrays = group_id_values(raw_event_id, raw_gen_mHH, raw_jet_genpt, raw_jet_pt, raw_cmssw_bscore, raw_inputs, num_elements=4)
        event_gen_mHH, jet_genpt, jet_pt, cmssw_bscore, jet_nn_inputs = grouped_arrays

        #Just pick the first entry of jet mHH arrays
        event_gen_mHH = ak.firsts(event_gen_mHH)
    else:
        # Handle case where genHH_mass doesn't exist
        event_id, grouped_arrays = group_id_values(raw_event_id, raw_jet_genpt, raw_jet_pt, raw_cmssw_bscore, raw_inputs, num_elements=4)
        jet_genpt, jet_pt, cmssw_bscore, jet_nn_inputs = grouped_arrays
        event_gen_mHH = None

    #Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    #B score from cmssw emulator
    cmsssw_bscore_sum = ak.sum(cmssw_bscore[:,:4], axis=1) #Only sum up the first four
    model_bscore_sum = nn_bscore_sum(model, jet_nn_inputs, b_index=class_labels['b'])

    cmssw_selection = (jet_ht > cmssw_btag_ht) & (cmsssw_bscore_sum > cmssw_btag)
    model_selection = (jet_ht > btag_ht_wp) & (model_bscore_sum > btag_wp)
    ht_only_selection = jet_ht > ht_only_wp

    #Plot the efficiencies w.r.t mHH, only if genHH_mass exists
    if all_event_gen_mHH is not None and event_gen_mHH is not None:
        bbbb_eff_mHH(model_dir,
                    all_event_gen_mHH,
                    event_gen_mHH,
                    cmssw_selection, model_selection, ht_only_selection)
    else:
        print("Skipping mHH efficiency plots because 'genHH_mass' is not available")

    #PLot the efficiencies
    #Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    cmssw_selected_events = Hist(ht_axis)
    model_selected_events = Hist(ht_axis)
    ht_only_selected_events = Hist(ht_axis)

    all_events.fill(all_jet_genht)
    cmssw_selected_events.fill(jet_genht[cmssw_selection])
    model_selected_events.fill(jet_genht[model_selection])
    ht_only_selected_events.fill(jet_genht[ht_only_selection])

    #Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)

    #Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    #Now plot all
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    ax.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, c=style.color_cycle[0], fmt='o', linewidth=3, label=r'BTag CMSSW Emulator @ 14 kHz (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(cmssw_btag_ht, cmssw_btag))
    ax.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3, label=r'Multiclass @ 14 kHz (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(btag_ht_wp, round(btag_wp,2)))

    #Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.1])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(HH $\to$ 4b)")
    plt.legend(loc='upper left')

    #Save plot
    plot_path = os.path.join(model_dir, "plots/physics/bbbb/HH_eff_HT")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')
    plt.show(block=False)

    #Plot a different plot comparing the multiclass with ht only selection
    fig2, ax2 = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax2, fontsize=style.MEDIUM_SIZE-2)

    ax2.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3,
                label=r'Multiclass @ 14 kHz (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(btag_ht_wp, round(btag_wp, 2)))
    ax2.errorbar(ht_only_x, ht_only_y, yerr=ht_only_err, c=style.color_cycle[2], fmt='o', linewidth=3,
                label=r'HT-only + QuadJets @ 14 kHz (L1 $HT$ > {} GeV)'.format(ht_only_wp))

    # Common plot settings for second plot
    ax2.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax2.grid(True)
    ax2.set_ylim([0., 1.1])
    ax2.set_xlim([0, 800])
    ax2.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax2.set_ylabel(r"$\epsilon$(HH $\to$ 4b)")
    ax2.legend(loc='upper left')

    # Save second plot
    ht_compare_path = os.path.join(model_dir, "plots/physics/bbbb/HH_eff_HT_vs_HTonly")
    plt.savefig(f'{ht_compare_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{ht_compare_path}.png', bbox_inches='tight')

    plt.show(block=False)


def bbbb_eff_mHH(model_dir,
                all_event_gen_mHH,
                event_gen_mHH,
                cmssw_selection, model_selection, ht_only_selection):
    """
    Plot HH->4b w.r.t gen m_HH
    """

    #Define the histogram edges
    mHH_edges = list(np.arange(0,1000,20))
    mHH_axis = hist.axis.Variable(mHH_edges, name = r"$HT^{gen}$")

    #Create the histograms
    all_events = Hist(mHH_axis)
    cmssw_selected_events = Hist(mHH_axis)
    model_selected_events = Hist(mHH_axis)
    ht_only_selected_events = Hist(mHH_axis)

    all_events.fill(all_event_gen_mHH)
    cmssw_selected_events.fill(event_gen_mHH[cmssw_selection])
    model_selected_events.fill(event_gen_mHH[model_selection])
    ht_only_selected_events.fill(event_gen_mHH[ht_only_selection])

    #Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_model = plot_ratio(all_events, model_selected_events)

    #Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    model_x, model_y, model_err = get_bar_patch_data(eff_model)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    #Load the working point from model directory
    btag_wp, btag_ht_wp, ht_only_wp =  load_bbbb_WPs(model_dir)

    #Plot a plot comparing the multiclass with ht only selection
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE-2)

    ax.errorbar(model_x, model_y, yerr=model_err, c=style.color_cycle[1], fmt='o', linewidth=3,
                label=r'Multiclass @ 14 kHz (L1 $HT$ > {} GeV, $\sum$ 4b > {})'.format(btag_ht_wp, round(btag_wp, 2)))
    ax.errorbar(ht_only_x, ht_only_y, yerr=ht_only_err, c=style.color_cycle[2], fmt='o', linewidth=3,
                label=r'HT-only + QuadJets @ 14 kHz (L1 $HT$ > {} GeV)'.format(ht_only_wp))

    # Common plot settings for second plot
    ax.hlines(1, 0, 1000, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.1])
    ax.set_xlim([0, 1000])
    ax.set_xlabel(r"$m_{HH}^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(HH $\to$ 4b)")
    ax.legend(loc='upper left')

    # Save second plot
    ht_compare_path = os.path.join(model_dir, "plots/physics/bbbb/HH_eff_mHH")
    plt.savefig(f'{ht_compare_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{ht_compare_path}.png', bbox_inches='tight')

    plt.show(block=False)

    return

if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python bbbb.py --deriveWPs
    2. Run efficiency based on the derived working points: python bbbb.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-s', '--sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125_addGenH/GluGluHHTo4B_PU200.root' , help = 'Signal sample for HH->bbbb')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for b-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for HH->4b')

    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=1000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    if args.deriveWPs:
        derive_bbbb_WPs(args.model_dir, args.minbias, n_entries=args.n_entries,tree=args.tree)
    elif args.eff:
        bbbb_eff(args.model_dir, args.sample, n_entries=args.n_entries,tree=args.tree)
