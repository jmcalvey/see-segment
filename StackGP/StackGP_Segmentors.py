from see.Segmentors import *


# bound a random value between min and max
def boundVal(val, minVal, maxVal):
    if val<minVal:
        return minVal
    elif val>maxVal:
        return maxVal
    else:
        return val

def seg_color_threshold(img, alpha1, alpha2, beta1, beta2, gamma1, gamma2):
    seg = segmentor()
    seg.params["algorithm"] = "ColorThreshold"

    seg.params["alpha1"] = boundVal(alpha1, 0, 1)
    seg.params["alpha2"] = boundVal(alpha2, 0, 1)
    seg.params["beta1"] = boundVal(beta1, 0, 1)
    seg.params["beta2"] = boundVal(beta2, 0, 1)
    seg.params["gamma1"] = boundVal(gamma1, 0, 1)
    seg.params["gamma2"] = boundVal(gamma2, 0, 1)

    return seg.evaluate(img)
seg_color_threshold.__repr__ = lambda: "color_threshold"


def seg_felzenszwalb(img, alpha1, alpha2, beta1):
    seg = segmentor()
    seg.params["algorithm"] = "Felzenszwalb"

    seg.params["alpha1"] = boundVal(alpha1, 0, 1)
    seg.params["alpha2"] = boundVal(alpha2, 0, 1)
    seg.params["beta1"] = boundVal(beta1, 0.01, 1)

    return seg.evaluate(img)
seg_felzenszwalb.__repr__ = lambda: "felzenszwalb"


def seg_slic(img, alpha1, beta1, n_segments, max_num_iter):
    seg = segmentor()
    seg.params["algorithm"] = "Slic"

    seg.params["alpha1"] = boundVal(alpha1, 0, 1)
    seg.params["beta1"] = boundVal(beta1, 0.01, 1)
    seg.params["n_segments"] = max(2, int(n_segments * 200))
    seg.params["max_num_iter"] = max(1, int(max_num_iter * 20))

    return seg.evaluate(img)
seg_slic.__repr__ = lambda: "slic"


def seg_slic0(img, alpha1, beta1, n_segments, max_num_iter):
    seg = segmentor()
    seg.params["algorithm"] = "SlicO"

    seg.params["alpha1"] = boundVal(alpha1, 0, 1)
    seg.params["beta1"] = boundVal(beta1, 0.01, 1)
    seg.params["n_segments"] = max(2, int(n_segments * 200))
    seg.params["max_num_iter"] = max(1, int(max_num_iter * 20))

    return seg.evaluate(img)
seg_slic0.__repr__ = lambda: "slic0"


def seg_quick_shift(img, alpha1):
    seg = segmentor()
    seg.params["algorithm"] = "QuickShift"
    seg.params["alpha1"] = boundVal(alpha1, 0, 1)

    return seg.evaluate(img)
seg_quick_shift.__repr__ = lambda: "quick_shift"


def seg_morph_chan_vese(img, alpha1, beta1, beta2, n_segments, max_num_iter):
    seg = segmentor()
    seg.params["algorithm"] = "Morphological_Chan_Vese"

    seg.params["alpha1"] = boundVal(alpha1, 0, 1)
    seg.params["beta1"] = boundVal(beta1, 0, 1)
    seg.params["beta2"] = boundVal(beta2, 0, 1)
    seg.params["n_segments"] = int(boundVal(n_segments, 0, 1) * 1.999)
    seg.params["max_num_iter"] = max(1, int(boundVal(max_num_iter, 0, 1) * 50))

    return seg.evaluate(img)
seg_morph_chan_vese.__repr__ = lambda: "morph_chan_vese"


def seg_morph_gac(img, alpha1, alpha2, beta1, beta2, n_segments, max_num_iter):
    seg = segmentor()
    seg.params["algorithm"] = "MorphGeodesicActiveContour"

    seg.params["alpha1"] = boundVal(alpha1, 0, 1)
    seg.params["alpha2"] = boundVal(alpha2, 0, 1)

    # Avoid zero or negative gradient parameters
    seg.params["beta1"] = boundVal(beta1, 0.01, 1)
    seg.params["beta2"] = boundVal(beta2, 0.01, 1)

    # SEE uses n_segments % 2 to choose initialization
    seg.params["n_segments"] = int(boundVal(n_segments, 0, 1) * 1.999)

    # Integer iteration count
    seg.params["max_num_iter"] = max(1, int(boundVal(max_num_iter, 0, 1) * 50))

    return seg.evaluate(img)
seg_morph_gac.__repr__ = lambda: "morph_gac"


segmentOps = [
    seg_felzenszwalb,
    seg_color_threshold,
    seg_slic,
    seg_slic0,
    seg_quick_shift,
    seg_morph_chan_vese,
    seg_morph_gac
]