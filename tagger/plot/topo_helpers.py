import awkward as ak
import numpy as np
import coffea
import os
import json
from coffea.nanoevents.methods import vector
from tagger.data.tools import extract_array, group_id_values

jet_collection = {'jet_energy': 'energy',
                    'jet_pt_phys': 'pt',
                    'jet_eta_phys': 'eta',
                    'jet_phi_phys': 'phi',
                }

tag_collection = {'jet_multijetscore_b': 'b',
                    'jet_multijetscore_charm': 'charm',
                    'jet_multijetscore_gluon': 'gluon',
                    'jet_multijetscore_light': 'light',
                    'jet_multijetscore_taup': 'taup',
                    'jet_multijetscore_taum': 'taum'
                }

def get_jet_mass(eta, pt, energy):
    p = np.sqrt(pt**2 * (1 + np.sinh(eta)**2))
    m = np.sqrt(energy**2 - p**2)
    return m

def delta_eta_helper(obj1, obj2):
    obj_eta = abs(obj1.eta - obj2.eta)
    return obj_eta

def inv_mass_helper(obj1, obj2):
    obj_mass = (obj1 + obj2).mass
    return obj_mass

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


def topo_input(inp_data, tagger_preds, tag_idxs, n_features, n_entries):
    sorting_idx = np.argsort(inp_data['event'])

    # get counts used for event splitting
    fields_dict = get_topo_arrays(inp_data, n_entries)
    # Create LorentzVectors
    jets = {v: fields_dict[k] for k, v in jet_collection.items()}
    jets = ak.zip(jets, behavior=vector.behavior, with_name="LorentzVector")

    jet_features = np.load("/eos/user/s/stella/nn_data/MinBias_PU200/jet_feature_names.npy")
    event_features = np.load("/eos/user/s/stella/nn_data/MinBias_PU200/event_feature_names.npy")

    # add the tagger predictions
    tags_list = jet_features[4:]
    for k in tags_list:
        idx = tag_idxs[tag_collection[k]]
        fields_dict[k] = tagger_preds[:,:, idx]

    n_jets = 15 # number of jets to keep
    jets_padded = {k: ak.fill_none(ak.pad_none(fields_dict[k], n_jets, clip=True), 0) for k in fields_dict.keys()}
    individual_shape = ak.to_numpy(jets_padded[list(fields_dict.keys())[0]]).shape
    events = np.empty((individual_shape[0], individual_shape[1] * len(fields_dict)))
    for j, k in enumerate(jet_features):
        events[:, j::len(fields_dict)] = ak.to_numpy(jets_padded[k])
    jets_ds = events.reshape(individual_shape[0], -1, len(fields_dict))
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
    max_inv_mass = ak.to_numpy(ak.max(ak.max(inv_mass_table, axis=1), axis=1))

    # get leading tagging invariant masses
    # finalstate_tags = ['jet_multijetscore_b', 'jet_multijetscore_charm', 'jet_multijetscore_taup',
    #                    'jet_multijetscore_taum']
    # tagging_scores = {k.split('_')[-1]: fields_dict[k] for k in finalstate_tags}
    # leading_idxs = {k: ak.argsort(v, axis=1, ascending=False)[:,:2] for k, v in tagging_scores.items()}
    # leading_b_mass = (jets[leading_idxs['b']][:,0] + jets[leading_idxs['b']][:,1]).mass
    # leading_c_mass = (jets[leading_idxs['charm']][:,0] + jets[leading_idxs['charm']][:,1]).mass
    # tau_p0m1_scores = tagging_scores['taup'][leading_idxs['taup']][:,0] + tagging_scores['taum'][leading_idxs['taum']][:,1]
    # tau_p1m0_scores = tagging_scores['taup'][leading_idxs['taup']][:,1] + tagging_scores['taum'][leading_idxs['taum']][:,0]
    # tau_pm_scores = np.stack((tau_p0m1_scores, tau_p1m0_scores), axis=-1)
    # tau_p0m1_mass =  (jets[leading_idxs['taup']][:,0] + jets[leading_idxs['taum']][:,1]).mass
    # tau_p1m0_mass =  (jets[leading_idxs['taup']][:,1] + jets[leading_idxs['taum']][:,0]).mass
    # tau_pm_mass = np.stack((tau_p0m1_mass, tau_p1m0_mass), axis=-1)
    # leading_tau_masses = (jets[leading_idxs['taup']][:,0] + jets[leading_idxs['taum']][:,0]).mass
    # leading_tau_mass = ak.where(leading_idxs['taup'][:,0] == leading_idxs['taum'][:,0],
    #     tau_pm_mass[np.arange(len(tau_pm_mass)), np.argmax(ak.to_numpy(tau_pm_scores), axis=1)],
    #     leading_tau_masses)

    ht = ak.to_numpy(ak.sum(jets.pt, axis=1))
    n_jets = ak.to_numpy(ak.num(jets.pt, axis=1))
    # tagging_sums= {f'{k}_sum': ak.sum(v, axis=1) for k, v in tagging_scores.items()}

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
    stacked_event_features = np.stack([event_features_dict[k] for k in event_features], axis=-1).data

    return [jets_ds, stacked_event_features, jets_mask]
