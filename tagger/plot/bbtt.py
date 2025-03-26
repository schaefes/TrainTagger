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
from tagger.data.tools import extract_array, extract_nn_inputs, group_id_values
from common import MINBIAS_RATE, WPs_CMSSW, find_rate, plot_ratio, get_bar_patch_data

# Helpers
def bbtt_seed(jet_pt, tau_pt):
    # tau pt thresholds L1_DoubleIsoTau34er2p1
    tau_pt = ak.sort(tau_pt, axis=1, ascending=False)
    tau_pt1 = (tau_pt[:, 0] >= 34)
    tau_pt2 = (tau_pt[:, 1] >= 34)
    tau_pt_mask = tau_pt1 & tau_pt2

    ht = ak.sum(jet_pt, axis=1)
    ht_mask = ht > 220

    mask = tau_pt_mask & ht_mask
    n_passed = np.sum(mask)

    return mask, n_passed

def x_vs_y(x, y):
    s = x / (x + y)

    return s

def nn_score_sums(model, jet_nn_inputs, class_labels, n_jets=4):
    #Btag input list for first 4 jets
    nn_outputs = [model.predict(np.asarray(jet_nn_inputs[:, i]))[0] for i in range(0,n_jets)]

    #Calculate the output sum
    b_index = class_labels['b']
    taup_index = class_labels['taup']
    taum_index = class_labels['taum']
    light_index = class_labels['light']

    # vs light preds
    taup_vs_l = np.transpose([x_vs_y(pred_score[:, taup_index], pred_score[:, light_index]) for pred_score in nn_outputs])
    taum_vs_l = np.transpose([x_vs_y(pred_score[:, taum_index], pred_score[:, light_index]) for pred_score in nn_outputs])
    b_vs_l = np.transpose([x_vs_y(pred_score[:, b_index], pred_score[:, light_index]) for pred_score in nn_outputs])

    # raw preds
    taup_preds = np.transpose([pred_score[:, taup_index] for pred_score in nn_outputs])
    taum_preds = np.transpose([pred_score[:, taum_index] for pred_score in nn_outputs])
    b_preds = np.transpose([pred_score[:, b_index] for pred_score in nn_outputs])

    bscore_sums, tscore_sums = [], []
    for b, taup, taum in zip([b_preds, b_vs_l], [taup_preds, taup_vs_l], [taum_preds, taum_vs_l]):
        tscore_sum, tau_indices = max_tau_sum(taup, taum)
        tscore_sums.append(tscore_sum)

        # use b scores from the remaining 2 jets
        indices = np.tile(np.arange(0,b.shape[1]),(b.shape[0],1))
        indices[np.arange(len(indices)).reshape(-1, 1), tau_indices] = -1
        b_indices = np.sort(indices, axis=1)[:, -2:]
        bscore_sum = b[np.arange(len(b)).reshape(-1, 1), b_indices].sum(axis=1)
        bscore_sums.append(bscore_sum)

    return bscore_sums, tscore_sums

def default_selection(jet_pt, jet_eta):
    mask = (jet_pt[:,:4] > 10) & (np.abs(jet_eta[:,:4]) < 2.4)
    event_mask = np.sum(mask, axis=1) == 4

    return event_mask

def pick_and_plot(rate_list, ht_list, bb_list, tt_list, ht, raw_score, model_dir, signal_path, n_entries, target_rate):
    """
    Pick the working points and plot
    """
    #plus, minus range
    RateRange = 0.5

    #Find the target rate points, plot them and print out some info as well
    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Get the coordinates
    target_rate_bb = np.array([bb_list[i] for i in target_rate_idx]) # NN cut dimension
    target_rate_tt = np.array([tt_list[i] for i in target_rate_idx]) # NN cut dimension
    target_rate_ht = np.array([ht_list[i] for i in target_rate_idx]) # HT cut dimension

    # Get the signal predictions and class labels
    signal_preds, n_events, signal_pt = make_predictions(signal_path, model_dir, n_entries)
    event_ht = ak.sum(signal_pt, axis=1)

    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Calculate the output sum
    b_index = class_labels['b']
    taup_index = class_labels['taup']
    taum_index = class_labels['taum']
    light_index = class_labels['light']

    if raw_score:
        taup_preds = np.transpose([pred_score[:, taup_index] for pred_score in signal_preds])
        taum_preds = np.transpose([pred_score[:, taum_index] for pred_score in signal_preds])
        b_preds = np.transpose([pred_score[:, b_index] for pred_score in signal_preds])
    else:
        taup_preds = np.transpose([x_vs_y(pred_score[:, taup_index], pred_score[:, light_index]) for pred_score in signal_preds])
        taum_preds = np.transpose([x_vs_y(pred_score[:, taum_index], pred_score[:, light_index]) for pred_score in signal_preds])
        b_preds = np.transpose([x_vs_y(pred_score[:, b_index], pred_score[:, light_index]) for pred_score in signal_preds])


    tscore_sum, tau_indices = max_tau_sum(taup_preds, taum_preds)
    indices = np.tile(np.arange(0,b_preds.shape[1]),(b_preds.shape[0],1))
    indices[np.arange(len(indices)).reshape(-1, 1), tau_indices] = -1
    b_indices = np.sort(indices, axis=1)[:, -2:]
    bscore_sum = b_preds[np.arange(len(b_preds)).reshape(-1, 1), b_indices].sum(axis=1)

    # Calculate the efficiency
    target_rate_eff = np.zeros(len(target_rate_ht))
    for i, (ht, bb, tt) in enumerate(zip(target_rate_ht, target_rate_bb, target_rate_tt)):
        ht_mask = event_ht > ht
        bb_mask = bscore_sum > bb
        tt_mask = tscore_sum > tt
        mask = ht_mask & bb_mask & tt_mask
        eff = np.sum(mask) / n_events
        target_rate_eff[i] = eff

    # Efficiency at target rate and HT WP
    target_bb = target_rate_bb[target_rate_ht==ht] # target rate and HT
    target_tt = target_rate_tt[target_rate_ht==ht] # target rate and HT
    target_eff = target_rate_eff[target_rate_ht==ht]

    # get max efficiency at target rate and HT WP
    wp_ht_eff_idx = np.argmax(target_eff)

    # Export the working point
    fixed_ht_wp = {"HT": float(ht), "BB": float(target_bb[wp_ht_eff_idx]),
        "TT": float(target_tt[wp_ht_eff_idx])}

    # save WPs
    score_type = 'raw' if raw_score else 'vs_light'
    plot_dir = os.path.join(model_dir, 'plots/physics/bbtt')
    os.makedirs(plot_dir, exist_ok=True)
    with open(os.path.join(plot_dir, f"bbtt_fixed_{ht}_wp_{score_type}.json"), "w") as f:
        json.dump(fixed_ht_wp, f, indent=4)

    # plot
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    im = ax.scatter(target_bb, target_tt, c=target_eff, s=500, marker='s',
                    cmap='Spectral_r',
                    linewidths=0,
                    norm=matplotlib.colors.LogNorm())

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"Efficiency at HT={ht}, {target_rate} [kHZ]")

    ax.set_ylabel(r"$\sum_{\tau^{+}_{max}\tau^{-}_{max}}$ scores")
    ax.set_xlabel(r"$\sum_{2~leading~jets}$ b scores")

    ax.set_xlim([0,2])
    ax.set_ylim([0,2])

    ax.legend(loc='upper right')
    plt.savefig(f"{plot_dir}/bbtt_rate.pdf", bbox_inches='tight')
    plt.savefig(f"{plot_dir}/bbtt_rate.png", bbox_inches='tight')

def make_predictions(data_path, model_dir, n_entries, tree='jetntuple/Jets', njets=4):
    data = uproot.open(data_path)[tree]

    model = load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    #Load input/ouput variables of the NN
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    raw_event_id = extract_array(data, 'event', n_entries)
    raw_jet_pt = extract_array(data, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(data, 'jet_eta_phys', n_entries)
    raw_inputs = extract_nn_inputs(data, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_nn_inputs = grouped_arrays
    default_sel = default_selection(jet_pt, jet_eta)
    jet_nn_inputs = jet_nn_inputs[default_sel]
    jet_pt = jet_pt[default_sel]

    #nn input list for first 4 jets
    nn_outputs = [model.predict(np.asarray(jet_nn_inputs[:, i]))[0] for i in range(0,njets)]

    return nn_outputs, n_events, jet_pt

def max_tau_sum(taup_preds, taum_preds):
    """
    Calculate the sum of the two highest tau scores
    """
    taup_argsort, taum_argsort = np.argsort(taup_preds, axis=1)[:,::-1][:,:2], np.argsort(taum_preds, axis=1)[:,::-1][:,:2]
    rows = np.arange(len(taup_preds)).reshape((-1, 1))
    alt_scores = taup_preds[rows, taup_argsort] + taum_preds[rows, taum_argsort][:,::-1]
    tau_alt_idxs = np.stack((taup_argsort[:,0], taum_argsort[:,1]), axis=-1)
    tau_alt2_idxs = np.stack((taup_argsort[:,1], taum_argsort[:,0]), axis=-1)
    tau_alt_idxs[ak.argmax(alt_scores, axis=1) == 1] = tau_alt2_idxs[ak.argmax(alt_scores, axis=1) == 1]
    tau_scores = taup_preds[rows, taup_argsort[:,0].reshape(-1,1)] + taum_preds[rows, taum_argsort[:,0].reshape(-1,1)]
    tau_scores = tau_scores.flatten()
    tau_scores[taup_argsort[:,0] == taum_argsort[:,0]] = ak.max(alt_scores, axis=1)[taup_argsort[:,0] == taum_argsort[:,0]]
    tau_idxs = np.stack((taup_argsort[:,0], taum_argsort[:,0]), axis=-1)
    tau_idxs[taup_argsort[:,0] == taum_argsort[:,0]] =  tau_alt_idxs[taup_argsort[:,0] == taum_argsort[:,0]]

    return tau_scores, tau_idxs

# Callables for studies
def derive_rate(minbias_path, model_dir, n_entries=100000, tree='jetntuple/Jets'):

    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_tau_pt = extract_array(minbias, 'jet_taupt', n_entries)

    array_fields = [raw_jet_pt, raw_jet_eta, raw_tau_pt]

    event_id, grouped_arrays  = group_id_values(raw_event_id, *array_fields, num_elements=4)
    jet_pt, jet_eta, tau_pt = grouped_arrays
    n_events = len(np.unique(raw_event_id))

    # Apply the cuts
    _, n_passed = bbtt_seed(jet_pt, tau_pt)

    # convert to rate [kHz]
    rate = {'rate': (n_passed / n_events) * MINBIAS_RATE}

    rate_dir = os.path.join(model_dir, "plots/physics/bbtt")
    os.makedirs(rate_dir, exist_ok=True)
    with open(os.path.join(rate_dir, f"bbtt_seed_rate.json"), "w") as f:
        json.dump(rate, f, indent=4)

    return

def derive_HT_WP(RateHist, ht_edges, n_events, model_dir, target_rate, RateRange=0.75):
    """
    Derive the HT only working points (without bb cuts)
    """

    plot_dir = os.path.join(model_dir, 'plots/physics/bbtt')

    #Derive the rate
    rate_list = []
    ht_list = []

    #Loop through the edges and integrate
    for ht in ht_edges[:-1]:

        #Calculate the rate
        rate = RateHist[{"ht": slice(ht*1j, None, sum)}][{"nn_bb": slice(0.0j, None, sum)}][{"nn_tt": slice(0.0j, None, sum)}]/n_events
        rate_list.append(rate*MINBIAS_RATE)

        #Append the results
        ht_list.append(ht)

    target_rate_idx = find_rate(rate_list, target_rate = target_rate, RateRange=RateRange)

    #Read WPs dict and add HT cut
    WP_json = os.path.join(plot_dir, "ht_working_point.json")
    working_point = {"ht_only_cut": float(ht_list[target_rate_idx[0]])}
    json.dump(working_point, open(WP_json, "w"), indent=4)

def derive_bbtt_WPs(model_dir, minbias_path, ht_cut, signal_path, n_entries=100, tree='outnano/Jets'):
    """
    Derive the HH->4b working points
    """

    model = load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    with open(os.path.join(model_dir, f"plots/physics/bbtt/bbtt_seed_rate.json"), "r") as f: rate = json.load(f)
    rate = np.round(rate['rate'], 1)

    #Load input/ouput variables of the NN
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    #Load the minbias data
    minbias = uproot.open(minbias_path)[tree]

    raw_event_id = extract_array(minbias, 'event', n_entries)
    raw_jet_pt = extract_array(minbias, 'jet_pt', n_entries)
    raw_jet_eta = extract_array(minbias, 'jet_eta_phys', n_entries)
    raw_inputs = extract_nn_inputs(minbias, input_vars, n_entries=n_entries)

    #Count number of total event
    n_events = len(np.unique(raw_event_id))
    print("Total number of minbias events: ", n_events)

    #Group these attributes by event id, and filter out groups that don't have at least 2 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_pt, raw_jet_eta, raw_inputs, num_elements=4)

    # Extract the grouped arrays
    # Jet pt is already sorted in the producer, no need to do it here
    jet_pt, jet_eta, jet_nn_inputs = grouped_arrays
    default_sel = default_selection(jet_pt, jet_eta)
    jet_nn_inputs = jet_nn_inputs[default_sel]

    bscore_sums, tscore_sums = nn_score_sums(model, jet_nn_inputs, class_labels)

    jet_ht = ak.sum(jet_pt, axis=1)[default_sel]

    assert(len(bscore_sums[0]) == len(jet_ht))
    assert(len(bscore_sums[1]) == len(jet_ht))

    #Define the histograms (pT edge and NN Score edge)
    ht_edges = list(np.arange(150,500,1)) + [10000] #Make sure to capture everything
    NN_edges = list([round(i,2) for i in np.arange(0, 1.2, 0.02)]) + [2.0]

    # for raw and vs light preds
    raw = True
    for bscore_sum, tscore_sum in zip(bscore_sums, tscore_sums):
        RateHist = Hist(hist.axis.Variable(ht_edges, name="ht", label="ht"),
                        hist.axis.Variable(NN_edges, name="nn_bb", label="nn_bb"),
                        hist.axis.Variable(NN_edges, name="nn_tt", label="nn_tt"))
        RateHist.fill(ht = jet_ht, nn_bb = bscore_sum, nn_tt = tscore_sum)

        #Derive the rate
        rate_list = []
        ht_list = []
        bb_list = []
        tt_list = []

        #Loop through the edges and integrate
        for bb in NN_edges[:-1]:
            for tt in NN_edges[:-1]:
                #Calculate the rate
                counts = RateHist[{"ht": slice(ht_cut*1j, None, sum)}][{"nn_bb": slice(bb*1.0j, None, sum)}][{"nn_tt": slice(tt*1.0j, None, sum)}]
                rate_list.append((counts / n_events)*MINBIAS_RATE)

                #Append the results
                ht_list.append(ht_cut)
                bb_list.append(bb)
                tt_list.append(tt)

        #Pick target rate and plot it
        pick_and_plot(rate_list, ht_list, bb_list, tt_list, ht_cut, raw, model_dir, signal_path, n_entries, rate)
        raw = False
    derive_HT_WP(RateHist, ht_edges, n_events, model_dir, rate)


    return

def bbtt_eff_HT(model_dir, signal_path, score_type, n_entries=100000, tree='outnano/Jets'):
    """
    Plot HH->4b efficiency w.r.t HT
    """
    with open(os.path.join(model_dir, f"plots/physics/bbtt/bbtt_seed_rate.json"), "r") as f: rate = json.load(f)
    rate = np.round(rate['rate'], 1)

    model=load_qmodel(os.path.join(model_dir, "model/saved_model.h5"))

    ht_egdes = list(np.arange(0,800,20))
    ht_axis = hist.axis.Variable(ht_egdes, name = r"$HT^{gen}$")

    #Check if the working point have been derived
    WP_path_220 = os.path.join(model_dir, f"plots/physics/bbtt/bbtt_fixed_220_wp_{score_type}.json")
    WP_path_190 = os.path.join(model_dir, f"plots/physics/bbtt/bbtt_fixed_190_wp_{score_type}.json")
    HT_path = os.path.join(model_dir, "plots/physics/bbtt/ht_working_point.json")

    #Get derived working points
    if os.path.exists(WP_path_220):
        with open(WP_path_220, "r") as f:  WPs = json.load(f)
        btag_wp_220 = WPs['BB']
        ttag_wp_220 = WPs['TT']
        ht_wp_220 = WPs['HT']
    if os.path.exists(WP_path_190):
        with open(WP_path_190, "r") as f:  WPs = json.load(f)
        btag_wp_190 = WPs['BB']
        ttag_wp_190 = WPs['TT']
        ht_wp_190 = WPs['HT']
    if os.path.exists(HT_path):
        with open(HT_path, "r") as f:  WPs = json.load(f)
        ht_only_wp = WPs['ht_only_cut']
    else:
        raise Exception("Working point does not exist. Run with --deriveWPs first.")

    #Load the signal data
    signal = uproot.open(signal_path)[tree]

    # Calculate the truth HT
    raw_event_id = extract_array(signal, 'event', n_entries)
    raw_jet_genpt = extract_array(signal, 'jet_genmatch_pt', n_entries)
    raw_jet_pt = extract_array(signal, 'jet_pt_phys', n_entries)
    raw_jet_eta = extract_array(signal, 'jet_eta_phys', n_entries)
    raw_tau_pt = extract_array(signal, 'jet_taupt', n_entries)

    # Load the inputs
    with open(os.path.join(model_dir, "input_vars.json"), "r") as f: input_vars = json.load(f)
    with open(os.path.join(model_dir, "class_label.json"), "r") as f: class_labels = json.load(f)

    raw_inputs = extract_nn_inputs(signal, input_vars, n_entries=n_entries)

    #Group these attributes by event id, and filter out groups that don't have at least 4 elements
    event_id, grouped_arrays  = group_id_values(raw_event_id, raw_jet_genpt, raw_jet_pt, raw_jet_eta, raw_tau_pt, raw_inputs, num_elements=4)
    jet_genpt, jet_pt, jet_eta, tau_pt, jet_nn_inputs = grouped_arrays
    default_sel = default_selection(jet_pt, jet_eta)

    #Calculate the ht
    jet_genht = ak.sum(jet_genpt, axis=1)
    jet_ht = ak.sum(jet_pt, axis=1)

    # Result from the baseline selection, multiclass tagger and ht only working point
    baseline_selection, _ = bbtt_seed(jet_pt, tau_pt)
    model_bscore_sums, model_tscore_sums = nn_score_sums(model, jet_nn_inputs, class_labels)

    # use either raw or vs light scores
    if score_type == 'raw':
        model_bscore_sum = model_bscore_sums[0]
        model_tscore_sum = model_tscore_sums[0]
    else:
        model_bscore_sum = model_bscore_sums[1]
        model_tscore_sum = model_tscore_sums[1]

    model_selection_220 = (jet_ht > ht_wp_220) & (model_bscore_sum > btag_wp_220) & (model_tscore_sum > ttag_wp_220) & default_sel
    model_selection_190 = (jet_ht > ht_wp_190) & (model_bscore_sum > btag_wp_190) & (model_tscore_sum > ttag_wp_190) & default_sel
    ht_only_selection = jet_ht > ht_only_wp

    #PLot the efficiencies
    #Basically we want to bin the selected truth ht and divide it by the overall count
    all_events = Hist(ht_axis)
    baseline_selected_events = Hist(ht_axis)
    model_selected_events_220 = Hist(ht_axis)
    model_selected_events_190 = Hist(ht_axis)
    ht_only_selected_events = Hist(ht_axis)

    all_events.fill(jet_genht)
    baseline_selected_events.fill(jet_genht[baseline_selection])
    model_selected_events_220.fill(jet_genht[model_selection_220])
    model_selected_events_190.fill(jet_genht[model_selection_190])
    ht_only_selected_events.fill(jet_genht[ht_only_selection])

    #Plot the ratio
    eff_baseline = plot_ratio(all_events, baseline_selected_events)
    eff_model_220 = plot_ratio(all_events, model_selected_events_220)
    eff_model_190 = plot_ratio(all_events, model_selected_events_190)

    #Get data from handles
    baseline_x, baseline_y, baseline_err = get_bar_patch_data(eff_baseline)
    model_x_220, model_y_220, model_err_220 = get_bar_patch_data(eff_model_220)
    model_x_190, model_y_190, model_err_190 = get_bar_patch_data(eff_model_190)
    eff_ht_only = plot_ratio(all_events, ht_only_selected_events)
    ht_only_x, ht_only_y, ht_only_err = get_bar_patch_data(eff_ht_only)

    # Plot ht distribution in the background
    counts, bin_edges = np.histogram(np.clip(jet_genht, 0, 800), bins=np.arange(0,800,40))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    normalized_counts = counts / np.sum(counts)

    #Now plot all
    fig,ax = plt.subplots(1,1,figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT,rlabel=style.CMSHEADER_RIGHT,ax=ax,fontsize=style.MEDIUM_SIZE-2)
    ax.bar(bin_centers, normalized_counts, width=bin_width, fill=False, edgecolor='grey')
    ax.errorbar(baseline_x, baseline_y, yerr=baseline_err, c=style.color_cycle[0], fmt='o', linewidth=3, label=r'bb$\tau \tau$ seed@ {} kHz (L1 $HT$ > {} GeV, $\tau 1,2 p_T$ > {} GeV)'.format(rate, 220, 34))
    ax.errorbar(model_x_220, model_y_220, yerr=model_err_220, c=style.color_cycle[1], fmt='o', linewidth=3, label=r'Multiclass @ {} kHz (L1 $HT$ > {} GeV, $\sum$ $\tau\tau$ > {}, $\sum$ bb > {})'.format(rate, ht_wp_220, round(ttag_wp_220,2), round(btag_wp_220,2)))
    ax.errorbar(model_x_190, model_y_190, yerr=model_err_190, c=style.color_cycle[2], fmt='o', linewidth=3, label=r'Multiclass @ {} kHz (L1 $HT$ > {} GeV, $\sum$ $\tau\tau$ > {}, $\sum$ bb > {})'.format(rate, ht_wp_190, round(ttag_wp_190,2), round(btag_wp_190,2)))
    ax.errorbar(ht_only_x, ht_only_y, yerr=ht_only_err, c=style.color_cycle[3], fmt='o', linewidth=3, label=r'HT-only @ {} kHz (L1 $HT$ > {} GeV)'.format(rate, ht_only_wp))

    #Plot other labels
    ax.hlines(1, 0, 800, linestyles='dashed', color='black', linewidth=4)
    ax.grid(True)
    ax.set_ylim([0., 1.1])
    ax.set_xlim([0, 800])
    ax.set_xlabel(r"$HT^{gen}$ [GeV]")
    ax.set_ylabel(r"$\epsilon$(HH $\to$ bb$\tau \tau$ trigger rate at {} kHz)".format(rate))
    plt.legend(loc='upper left')

    #Save plot
    plot_path = os.path.join(model_dir, f"plots/physics/bbtt/HHbbtt_eff_{score_type}_ht_comparison")
    plt.savefig(f'{plot_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{plot_path}.png', bbox_inches='tight')
    plt.show(block=False)


if __name__ == "__main__":
    """
    2 steps:

    1. Derive working points: python bbtt.py --deriveWPs
    2. Run efficiency based on the derived working points: python bbtt.py --eff
    """

    parser = ArgumentParser()
    parser.add_argument('-m','--model_dir', default='output/baseline', help = 'Input model')
    parser.add_argument('-s', '--signal', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_4param_021024/ggHHbbtt_PU200.root' , help = 'Signal sample for HH->bbtt')
    parser.add_argument('--minbias', default='/eos/cms/store/cmst3/group/l1tr/sewuchte/l1teg/fp_ntuples_v131Xv9/extendedTRK_4param_021024/MinBias_PU200.root' , help = 'Minbias sample for deriving rates')

    #Different modes
    parser.add_argument('--deriveRate', action='store_true', help='derive the rate for the bbtt seed')
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for b-tagging')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for HH->bbtt')

    parser.add_argument('--tree', default='jetntuple/Jets', help='Tree within the ntuple containing the jets')

    #Other controls
    parser.add_argument('-n','--n_entries', type=int, default=1000, help = 'Number of data entries in root file to run over, can speed up run time, set to None to run on all data entries')
    args = parser.parse_args()

    if args.deriveRate:
        derive_rate(args.minbias, args.model_dir, n_entries=args.n_entries,tree=args.tree)
    if args.deriveWPs:
        derive_bbtt_WPs(args.model_dir, args.minbias, 220, args.signal, n_entries=args.n_entries,tree=args.tree)
        derive_bbtt_WPs(args.model_dir, args.minbias, 190, args.signal, n_entries=args.n_entries,tree=args.tree)
    elif args.eff:
        bbtt_eff_HT(args.model_dir, args.signal, 'raw', n_entries=args.n_entries,tree=args.tree)
        bbtt_eff_HT(args.model_dir, args.signal, 'vs_light', n_entries=args.n_entries,tree=args.tree)
