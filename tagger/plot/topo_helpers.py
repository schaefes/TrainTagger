import awkward as ak
import numpy as np
from tagger.data.tools import extract_array, group_id_values


def get_input_mask(jets_ds, n_features):
    input_mask = np.where(np.sum(jets_ds, axis=2) == 0, 0, 1)
    n_jets = np.sum(input_mask, axis=1)
    input_mask = input_mask * input_mask.shape[1] / n_jets[:, np.newaxis]
    input_mask = np.repeat(input_mask[:, :,np.newaxis], n_features, axis=2)

    return input_mask

def get_topo_arrays(minbias, n_entries):
    fields = ['event', 'jet_energy', 'jet_pt_phys', 'jet_eta_phys', 'jet_phi_phys']
    data = {}
    for f in fields:
        data[f] = extract_array(minbias, f, n_entries)
    raw_event_id = extract_array(minbias, 'event', n_entries)

    event_sorting = ak.argsort(data['event'])
    _, counts = np.unique(data['event'][event_sorting], return_counts=True)
    event_mask = ak.where(counts < 4, False, True) # mask events with less than 4 jets

    # switch to event shape and apply event mask
    fields_dict = {x: ak.unflatten(data[x][event_sorting], counts)[event_mask] for x in fields}
    del fields_dict['event']

    return fields_dict


def topo_tagger_input(inp_data, tagger_preds, n_features, n_entries):
    sorting_idx = np.argsort(inp_data['event'])
    jet_collection = {'jet_energy': 'energy',
                      'jet_pt_phys': 'pt',
                      'jet_eta_phys': 'eta',
                      'jet_phi_phys': 'phi',
                    }

    fields = ['event',
        'jet_energy', 'jet_pt_phys', 'jet_eta_phys', 'jet_phi_phys',
        ]

    data = {k: inp_data[k].array()[sorting_idx] for k in inp_data.keys()}

    # get counts used for event splitting
    fields_dict = get_topo_arrays(inp_data, n_entries)

    # create jet objects for coffea use
    jets = {v: ak.unflatten(data[k][event_sorting], counts)[event_mask] for k, v in jet_collection.items()}

    # Create LorentzVectors
    jets = ak.zip(jets, behavior=vector.behavior, with_name="LorentzVector")

    n_jets = 15 # number of jets to keep
    jets_padded = {k: ak.fill_none(ak.pad_none(jet_fields[k], n_jets), 0) for k in jet_fields.keys()}
    individual_shape = ak.to_numpy(jets_padded[list(jet_fields.keys())[0]]).shape
    events = np.empty((individual_shape[0], individual_shape[1] * len(jet_fields)))
    for j, k in enumerate(jets_padded.keys()):
        events[:, j::len(jet_fields)] = ak.to_numpy(jets_padded[k])
    jets_ds = events.reshape(individual_shape[0], -1, len(jet_fields))

    jets_mask = get_input_mask(jets_ds, n_features)

   # get event level features
    inv_mass_table = jets.metric_table(jets, axis=1, metric=inv_mass_helper)
    dEta_table = jets.metric_table(jets, axis=1, metric=delta_eta_helper)
    dEta_max1 = ak.max(dEta_table, axis=1)
    dEta_argmax1 = ak.argmax(dEta_table, axis=1)
    idx0 = np.array(range(len(jets)))
    idx2 = np.array(ak.argmax(dEta_max1, axis=1))
    idx1 = np.array(dEta_argmax1[idx0, idx2])
    inv_mass = np.array(inv_mass_table[idx0, idx1, idx2])
    dEta = np.array(ak.max(dEta_max1, axis=1))
    max_inv_mass = ak.max(ak.max(inv_mass_table, axis=1), axis=1)

    # get leading tagging invariant masses
    finalstate_tags = ['jet_multijetscore_b', 'jet_multijetscore_charm', 'jet_multijetscore_taup',
                       'jet_multijetscore_taum']
    tagging_scores = {k.split('_')[-1]: fields_dict[k] for k in finalstate_tags}
    leading_idxs = {k: ak.argsort(v, axis=1, ascending=False)[:,:2] for k, v in tagging_scores.items()}
    leading_b_mass = (jets[leading_idxs['b']][:,0] + jets[leading_idxs['b']][:,1]).mass
    leading_c_mass = (jets[leading_idxs['charm']][:,0] + jets[leading_idxs['charm']][:,1]).mass
    tau_p0m1_scores = tagging_scores['taup'][leading_idxs['taup']][:,0] + tagging_scores['taum'][leading_idxs['taum']][:,1]
    tau_p1m0_scores = tagging_scores['taup'][leading_idxs['taup']][:,1] + tagging_scores['taum'][leading_idxs['taum']][:,0]
    tau_pm_scores = np.stack((tau_p0m1_scores, tau_p1m0_scores), axis=-1)
    tau_p0m1_mass =  (jets[leading_idxs['taup']][:,0] + jets[leading_idxs['taum']][:,1]).mass
    tau_p1m0_mass =  (jets[leading_idxs['taup']][:,1] + jets[leading_idxs['taum']][:,0]).mass
    tau_pm_mass = np.stack((tau_p0m1_mass, tau_p1m0_mass), axis=-1)
    leading_tau_masses = (jets[leading_idxs['taup']][:,0] + jets[leading_idxs['taum']][:,0]).mass
    leading_tau_mass = ak.where(leading_idxs['taup'][:,0] == leading_idxs['taum'][:,0],
        tau_pm_mass[np.arange(len(tau_pm_mass)), np.argmax(ak.to_numpy(tau_pm_scores), axis=1)],
        leading_tau_masses)

    ht = ak.sum(jets.pt, axis=1)
    n_jets = ak.to_numpy(ak.num(jets.pt, axis=1))
    tagging_sums= {f'{k}_sum': ak.sum(v, axis=1) for k, v in tagging_scores.items()}

    event_features_dict = {
        'ht': ht,
        'mjj': inv_mass,
        'max_mjj': max_inv_mass,
        'deta': dEta,
        'njets': n_jets,
        # 'b_mass': leading_b_mass,
        # 'c_mass': leading_c_mass,
        # 'tau_mass': leading_tau_mass,
    }
    # event_features_dict.update(tagging_sums)
    event_features = np.stack(list(event_features_dict.values()), axis=-1)

    return [jets_ds, event_features, jets_mask]
