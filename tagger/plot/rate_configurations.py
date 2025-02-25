import awkward as ak
import numpy as np


def ht_sum_btag(jet_pt, jet_eta, jet_btag, n_jets):
    # cut all jets with -1 for btag
    jet_btag = jet_btag[jet_btag > -1]
    jet_eta = jet_eta[jet_btag > -1]
    jet_pt = jet_pt[jet_btag > -1]

    # allow only events with jets that contain at least #n_jets btagged jets
    n_mask = ak.num(jet_pt) >= n_jets

    btag_sum = ak.sum(jet_btag[:,:n_jets], axis=1)

    ht = ak.sum(jet_pt, axis=1)

    mask = (ht > 220) & (btag_sum > (0.58 * n_jets))
    mask = mask & n_mask
    n_passed = np.sum(mask)

    return mask, n_passed

def ht_worst_btag(jet_pt, jet_eta, jet_btag, n_jets):
    # cut all jets with -1 for btag
    jet_btag = jet_btag[jet_btag > -1]
    jet_eta = jet_eta[jet_btag > -1]
    jet_pt = jet_pt[jet_btag > -1]

    btag_sum = ak.min(jet_btag[:,:n_jets], axis=1)

    ht = ak.sum(jet_pt, axis=1)

    mask = (ht > 220) & (btag_sum > 0.5)
    n_passed = np.sum(mask)

    return mask, n_passed

def quadjet_ht(jet_pt, jet_eta, jet_btag, n_jets):
    pt_cuts = [70, 55, 40, 40]
    pt_mask = np.array([True]*len(jet_pt))
    for i, pt in enumerate(pt_cuts):
        mask = jet_ptl1(jet_pt[:,i], jet_eta[:,i], pt)
        pt_mask = pt_mask & mask
    eta_mask = abs(jet_eta) < 2.4
    ht_pts = jet_ptl1(jet_pt, jet_eta, 30)
    jets_ht = jet_pt[eta_mask & ht_pts]
    ht_mask = ak.sum(jets_ht, axis=1) > htl1(400)
    n_passed = np.sum((ht_mask & pt_mask))

    return mask, n_passed

# translate offline to online and return mask of matching objects
def barrel_ptl1(jet_pt, jet_eta, offline_pt):
    # 0 to 1.3 eta
    l1pt = (offline_pt - 17.492) / 1.294
    mask_pt = jet_pt > l1pt
    mask_eta = abs(jet_eta) < 1.3
    mask = mask_pt & mask_eta
    return mask

def endcap_ptl1(jet_pt, jet_eta, offline_pt):
    # 1.3 to 3 eta
    l1pt = (offline_pt - 15.296) / 1.773
    mask_pt = jet_pt > l1pt
    mask_eta = (1.3 <= abs(jet_eta)) & (abs(jet_eta) < 3.)
    mask = mask_pt & mask_eta
    return mask

def forward_ptl1(jet_pt, jet_eta, offline_pt):
    # 3 to 5.2 eta
    l1pt = (offline_pt - 61.507) / 1.442
    mask_pt = jet_pt > l1pt
    mask_eta = (3. <= abs(jet_eta)) & (abs(jet_eta) < 5.2)
    mask = mask_pt & mask_eta
    return mask

def jet_ptl1(jet_pt, jet_eta, offline_pt):
    mask_barrel = barrel_ptl1(jet_pt, jet_eta, offline_pt)
    mask_endcap = endcap_ptl1(jet_pt, jet_eta, offline_pt)
    mask_forward = forward_ptl1(jet_pt, jet_eta, offline_pt)
    mask = mask_barrel | mask_endcap | mask_forward
    return mask

def htl1(offline_ht):
    l1ht = (offline_ht - 49.014) / 1.135

    return l1ht
