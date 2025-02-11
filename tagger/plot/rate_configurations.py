def ht_btag(jet_pt, jet_eta, jet_btag):

    ht = ak.sum(jet_pt, axis=1)

    # cut -1 values in btags
    jet_btag = jet_btag[jet_btag > 0]
    btag_sum = ak.sum(ak.sort(jet_btag, ascending=False)[:,:4], axis=1)

    mask = (ht > 220) & (btag_sum > 2.32)
    n_passed = np.sum(mask)

    return n_passed

def quadjet_ht(jet_pt, jet_eta, b_tag):
    # ensure pt sorting
    pt_sort = ak.argsort(jet_pt, axis=1)
    jet_pt = jet_pt[pt_sort]
    jet_eta = jet_eta[pt_sort]
    pt_cuts = [70, 55, 40, 40]

    pt_mask = np.array([True]*len(jet_pt))
    for i, pt in enumerate(pt_cuts):
        mask = jet_ptl1(jet_pt[:,i], jet_eta[:,i], pt)
        pt_mask = pt_mask & mask

    # for ht, consider only jets with pt > 30GeV and abs eta < 2.4
    # 30 GeV prob needs to be converted with barrel endcap forward
    # also, do the vbf topo tagger without tagging info, and check masking in gitlab







    return cuts_dict

# translate offline to online and return mask of matching objects
def barrel_ptl1(jet_pt, offline_pt, jet_eta):
    # 0 to 1.3 eta
    l1pt = (offline_pt - 17.492) / 1.294
    mask_pt = jet_pt > l1pt
    mask_eta = abs(jet_eta) < 1.3
    mask = mask_pt & mask_eta
    return mask

def endcap_ptl1(jet_pt, offline_pt, jet_eta):
    # 1.3 to 3 eta
    l1pt = (offline_pt - 15.296) / 1.773
    mask_pt = jet_pt > l1pt
    mask_eta = (1.3 <= abs(jet_eta)) & (abs(jet_eta) < 3.)
    mask = mask_pt & mask_eta
    return mask

def forward_ptl1(jet_pt, offline_pt, jet_eta):
    # 3 to 5.2 eta
    l1pt = (offline_pt - 61.507) / 1.442
    mask_pt = jet_pt > l1pt
    mask_eta = (3. <= abs(jet_eta)) & (abs(jet_eta) < 5.2)
    mask = mask_pt & mask_eta
    return mask

def jet_ptl1(jet_pt, jet_eta, offline_pt):
    mask_barrel = barrel_ptl1(jet_pt, offline_pt, jet_eta)
    mask_endcap = endcap_ptl1(jet_pt, offline_pt, jet_eta)
    mask_forward = forward_ptl1(jet_pt, offline_pt, jet_eta)

    mask = mask_barrel | mask_endcap | mask_forward
    return mask

def htl1(offline_ht):
    l1ht = (offline_ht - 49.014) / 1.135

    return l1ht
