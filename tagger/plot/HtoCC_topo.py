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
import rate_configurations
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, get_bar_patch_data
from topo_helpers import topo_input

style.set_style()

def return_sum_score(c_score, b_score, tag_sum):
    # Return the sum score based on the input
    if tag_sum == 'cb':
        return c_score + b_score
    elif tag_sum == 'c':
        return c_score
    elif tag_sum == 'b':
        return b_score

def nn_score_sum(nn_outputs, class_labels, tag_sum, njets=2):
    b_index = class_labels['b']
    c_index = class_labels['charm']

    sorted_bscores = ak.sort([pred_score[:, b_index] for pred_score in nn_outputs], axis=0)[::-1]
    bscore_sum = np.sum(sorted_bscores[:njets,:], axis=0)
    sorted_cscores = np.sort([pred_score[:, c_index] for pred_score in nn_outputs], axis=0)[::-1]
    cscore_sum = np.sum(sorted_cscores[:njets,:], axis=0)
    sum_score = return_sum_score(cscore_sum, bscore_sum, tag_sum)

    return sum_score

def pick_and_plot(rate_list, ht_list, nn_list, model_dir, tag_sum, fuse, rate):
    """
    Pick the working points and plot
    """
    plot_dir = os.path.join(model_dir, 'plots/physics/wps')
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
    target_rate_idx = find_rate(rate_list, target_rate = rate, RateRange=RateRange)

    #Get the coordinates
    target_rate_NN = [nn_list[i] for i in target_rate_idx] # NN cut dimension
    target_rate_HT = [ht_list[i] for i in target_rate_idx] # HT cut dimension

    # Create an interpolation function
    interp_func = interp1d(target_rate_HT, target_rate_NN, kind='linear', fill_value='extrapolate')

    # Interpolate the NN value for the desired HT
    working_point_NN = interp_func(WPs_CMSSW['btag_l1_ht'])

    # Export the working point
    working_point = {"HT": WPs_CMSSW['btag_l1_ht'], "NN": float(working_point_NN)}
    rate = round(rate)

    # for unified category: wps are identical
    if fuse:
        for tag_sum in ['c', 'b', 'cb']:
            with open(os.path.join(plot_dir, f"working_point_{rate}_{tag_sum}.json"), "w") as f:
                json.dump(working_point, f, indent=4)

    ax.plot(target_rate_NN, target_rate_HT,
                linewidth=5,
                color ='firebrick',
                label = r"${} \pm {}$ kHz".format(rate, RateRange))

    ax.legend(loc='upper right')
    plt.savefig(f"{plot_dir}/cc_{rate}_{tag_sum}.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/cc_rate_{rate}_{tag_sum}.png", bbox_inches='tight')


def derive_cc_WPs(tagger_dir, topo_dir, minbias_path, seed_name, tag_sum,
    nn_type, n_entries, tree='outnano/Jets'):
    """
    Derive the HH->4b working points
    """
    with open(os.path.join(f"tagger/output/physics/cc_rates/{seed_name}_rate.json"), "r") as f: rate = json.load(f)
    target_rate = rate['rate']

    tagger_model = load_qmodel(os.path.join(tagger_dir, "model/saved_model.h5"))
    topo_model = load_qmodel(os.path.join(topo_dir, "model.h5"))

    #Load input/ouput variables of the NN
    with open(os.path.join(tagger_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(tagger_dir, "class_label.json"), "r") as f: tagger_labels = json.load(f)
    with open(os.path.join(topo_dir, "class_label.json"), "r") as f: topo_labels = json.load(f)

    try:
        cc_topo_idx = topo_labels['VBFHToCC_PU200']
        bb_topo_idx = topo_labels['VBFHToBB_PU200']
        fuse = False
    except:
        cc_topo_idx = topo_labels['VBFHToBBorCC_PU200']
        bb_topo_idx = topo_labels['VBFHToBBorCC_PU200']
        fuse = True

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 4 elements
    # 2 VBF Jets + 2 Higgs Decay Jets
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_nn_inputs = grouped_arrays

    # configurations for the nn type used for the WPs
    if nn_type == "topo":
        n_features = topo_model.get_layer('avgpool').input_shape[-1]
        n_jets = ak.num(jet_nn_inputs, axis=1)
        flat_jets = np.asarray(ak.flatten(jet_nn_inputs, axis=1))
        tagger_preds = tagger_model.predict(flat_jets)[0]
        jet_preds = ak.unflatten(tagger_preds, n_jets)
        topo_inputs = topo_input(minbias, jet_preds, tagger_labels, n_features, n_entries)
        topo_outputs = topo_model.predict(topo_inputs)
        nn_score = return_sum_score(topo_outputs[:, cc_topo_idx], topo_outputs[:, bb_topo_idx], tag_sum)
        model_dir = topo_dir
        max_score = 1
    elif nn_type == "tagger":
        nn_outputs = [tagger_model.predict(np.asarray(jet_nn_inputs[:, i]))[0] for i in range(0,4)]
        nn_score = nn_score_sum(nn_outputs, tagger_labels, tag_sum, 2)
        model_dir = tagger_dir
        max_score = 2

    #Calculate the output sum
    ht = ak.sum(jet_pt, axis=1)
    assert(len(nn_score) == len(ht))

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(0,250,1)) + [5000] #Make sure to capture everything
    NN_edges = list([round(i,3) for i in np.arange(0, 0.75 * max_score, 0.002 * max_score)])
    NN_edges.append(max_score)

    RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                    hist.axis.Variable(NN_edges, name="nn", label="nn"))

    RateHist.fill(ht = ht, nn = nn_score)

    #Derive the rate
    rate_list = []
    ht_list = []
    nn_list = []
    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:
        for NN in NN_edges[:-1]:

            #Calculate the rate
            counts = RateHist[{"ht": slice(ht*1j, None, sum)}][{"nn": slice(NN*1.00j, None, sum)}]
            rate_list.append((counts / n_events) * MINBIAS_RATE)

            #Append the results
            ht_list.append(ht)
            nn_list.append(NN)

    #Pick target rate and plot it
    pick_and_plot(rate_list, ht_list, nn_list, model_dir, tag_sum, fuse, target_rate)

    return

def derive_rate(minbias_path, seed_name, n_entries=100000, tree='outnano/Jets'):

    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_cmssw_bscore = extract_array(minbias, 'jet_bjetscore', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta', n_entries)

    array_fields = [raw_cmssw_bscore, raw_jet_pt, raw_jet_eta]

    event_id, grouped_arrays  = group_id_values(raw_event_id, *array_fields, num_elements=4)
    jet_btag, jet_pt, jet_eta = grouped_arrays
    n_events = len(np.unique(raw_event_id))

    # Apply the cuts
    seeds_function = getattr(rate_configurations, seed_name)
    _, n_passed = seeds_function(jet_pt, jet_eta, jet_btag, 2)

    # convert to rate [kHz]
    rate = {'rate': (n_passed / n_events) * MINBIAS_RATE}

    rate_dir = os.path.join('tagger/output/physics/cc_rates')
    os.makedirs(rate_dir, exist_ok=True)
    with open(os.path.join(rate_dir, f"{seed_name}_rate.json"), "w") as f:
        json.dump(rate, f, indent=4)

    print(f'Derived rate for seed {seed_name}:', rate)

    return

def cc_eff_HT(tagger_dir, topo_dir, signal_path, seed_name, tag_sum, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """

    with open(os.path.join(f"tagger/output/physics/cc_rates/{seed_name}_rate.json"), "r") as f: rate = json.load(f)
    rate = np.round(rate['rate'], 1)
    tagger_model=load_qmodel(os.path.join(tagger_dir, "model/saved_model.h5"))
    topo_model=load_qmodel(os.path.join(topo_dir, "model.h5"))

    ht_egdes = list(np.arange(0,800,20))
    ht_axis = hist.axis.Variable(ht_egdes, name = r"$HT^{gen}$")

    #Check if the working point have been derived
    final_state = os.path.basename(signal_path).replace('_PU200.root', '').split('To')[-1].lower()
    WP_tagger = os.path.join(tagger_dir, f"plots/physics/wps/working_point_{round(rate)}_{tag_sum}.json")
    WP_topo = os.path.join(topo_dir, f"plots/physics/wps/working_point_{round(rate)}_{tag_sum}.json")


    #Get derived working points
    if os.path.exists(WP_tagger) & os.path.exists(WP_topo):
        with open(WP_tagger, "r") as f:  WPs = json.load(f)
        tagger_wp = WPs['NN']
        tagger_ht_wp = WPs['HT']
        with open(WP_topo, "r") as f:  WPs = json.load(f)
        topo_wp = WPs['NN']
        topo_ht_wp = WPs['HT']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    #Load the signal data
    signal = uproot.open(signal_path)[tree]

    # Calculate the truth HT
    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt_phys', n_entries)
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)
    raw_cmssw_bscore = extract_array(signal, 'jet_bjetscore', n_entries)

    # Load the inputs
    with open(os.path.join(tagger_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(tagger_dir, "class_label.json"), "r") as f: tagger_labels = json.load(f)
    with open(os.path.join(topo_dir, "class_label.json"), "r") as f: topo_labels = json.load(f)
    try:
        cc_topo_idx = topo_labels['VBFHToCC_PU200']
        bb_topo_idx = topo_labels['VBFHToBB_PU200']
        topo_model_type = 'decoupled'
    except:
        cc_topo_idx = topo_labels['VBFHToBBorCC_PU200']
        bb_topo_idx = topo_labels['VBFHToBBorCC_PU200']
        topo_model_type = 'fuse'

    raw_inputs = extract_nn_inputs(signal, input_vars, n_entries=n_entries)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_genpt, raw_jet_pt, raw_jet_eta, raw_cmssw_bscore, raw_inputs, num_elements=4)
    jet_genpt, jet_pt, jet_eta, cmssw_bscore, jet_nn_inputs = grouped_arrays

    #Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    # Topo scores
    n_features = topo_model.get_layer('avgpool').input_shape[-1]
    n_jets = ak.num(jet_nn_inputs, axis=1)
    flat_jets = np.asarray(ak.flatten(jet_nn_inputs, axis=1))
    tagger_preds = tagger_model.predict(flat_jets)[0]
    jet_preds = ak.unflatten(tagger_preds, n_jets)
    topo_inputs = topo_input(signal, jet_preds, tagger_labels, n_features, n_entries)
    topo_outputs = topo_model.predict(topo_inputs)
    topo_score = return_sum_score(topo_outputs[:, cc_topo_idx], topo_outputs[:, bb_topo_idx], tag_sum)

    # Tagger scores
    nn_outputs_tagger = [jet_preds[:, i] for i in range(0,4)]
    tagger_score = nn_score_sum(nn_outputs_tagger, tagger_labels, tag_sum, 2)

    seeds_function = getattr(rate_configurations, seed_name)
    cmssw_selection, _ = seeds_function(jet_pt, jet_eta, cmssw_bscore, 2)
    from IPython import embed; embed()
    tagger_selection = (jet_ht > tagger_ht_wp) & (tagger_score > tagger_wp)
    topo_selection = (jet_ht > topo_ht_wp) & (topo_score > topo_wp)

    #PLot the efficiencies
    #Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    cmssw_selected_events = Hist(ht_axis)
    tagger_selected_events = Hist(ht_axis)
    topo_selected_events = Hist(ht_axis)

    all_events.fill(jet_genht)
    cmssw_selected_events.fill(jet_genht[cmssw_selection])
    tagger_selected_events.fill(jet_genht[tagger_selection])
    topo_selected_events.fill(jet_genht[topo_selection])

    #Plot the ratio
    eff_cmssw = plot_ratio(all_events, cmssw_selected_events)
    eff_tagger = plot_ratio(all_events, tagger_selected_events)
    eff_topo = plot_ratio(all_events, topo_selected_events)

    #Get data from handles
    cmssw_x, cmssw_y, cmssw_err = get_bar_patch_data(eff_cmssw)
    tagger_x, tagger_y, tagger_err = get_bar_patch_data(eff_tagger)
    topo_x, topo_y, topo_err = get_bar_patch_data(eff_topo)

    #Now plot all
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    ax.errorbar(cmssw_x, cmssw_y, yerr=cmssw_err, c=style.color_cycle[0], fmt='o', linewidth=3,
        label=r'btag CMSSW Emulator @ {} kHz ({} seed)'.format(rate, seed_name))
    ax.errorbar(tagger_x, tagger_y, yerr=tagger_err, c=style.color_cycle[1], fmt='o', linewidth=3,
        label=r'Multiclass Jet Tagger @ {} kHz (L1 $HT$ > {} GeV, $\sum$ {} > {})'.format(rate, tagger_ht_wp, tag_sum, round(tagger_wp,2)))
    ax.errorbar(topo_x, topo_y, yerr=topo_err, c=style.color_cycle[2], fmt='o', linewidth=3,
        label=r'Multiclass Topo Tagger @ {} kHz (L1 $HT$ > {} GeV, {} > {})'.format(rate, topo_ht_wp, tag_sum, round(topo_wp,2)))

    #Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.1])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(H $\to$ cc trigger rate at 14 kHz)")
    plt.legend(loc='upper left')

    #Save plot
    tagger_path = os.path.join(tagger_dir, f"plots/physics/{final_state}/Hcc_eff_{seed_name}_{tag_sum}_{topo_model_type}")
    topo_path = os.path.join(topo_dir, f"plots/physics/{final_state}/Hcc_eff_{seed_name}_{tag_sum}_{topo_model_type}")
    for plot_path in [tagger_path, topo_path]:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
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
    parser.add_argument('-tagger','--tagger_dir', default='/eos/user/s/stella/TrainTagger/output/baseline', help='Jet tagger model')
    parser.add_argument('-topo','--topo_dir', default='/eos/user/s/stella/nn_models/MinBias_PU200_VBFHToBB_PU200_VBFHToCC_PU200_VBFHToInvisible_PU200_VBFHToTauTau_PU200/fold1of3/model_ds_bg4', help='Topo tagger model')
    parser.add_argument('-seed', '--seed_name', default='ht_btag', help='Decide which seed to compare to')
    parser.add_argument('-s', '--sample', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/VBFHToBB_PU200.root', help='Signal sample for VBF->H->bb')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_jettuples_090125/MinBias_PU200.root', help='Minbias sample for deriving rates')

    #Different modes
    parser.add_argument('--deriveRate', action='store_true', help='Derive the fixed rate')
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for c-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF->H->cc')

    #Other controls
    parser.add_argument('-tag-sum', '--tag-sum', default='c', help='Decide which score to sum for the c-tagging (c, b, cb)')
    parser.add_argument('-nn', '--nn_type', default='topo', help='Decide which NN to use for the WPs (topo, tagger)')
    parser.add_argument('-n','--n_entries', type=int, default=1000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    if args.deriveRate:
        derive_rate(args.minbias, args.seed_name, n_entries=args.n_entries)
    elif args.deriveWPs:
        derive_cc_WPs(args.tagger_dir, args.topo_dir, args.minbias, args.seed_name, args.tag_sum,
            args.nn_type, args.n_entries)
    elif args.eff:
        cc_eff_HT(args.tagger_dir, args.topo_dir, args.sample, args.seed_name, args.tag_sum, n_entries=args.n_entries)
