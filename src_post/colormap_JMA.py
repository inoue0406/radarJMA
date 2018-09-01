# custom colormap for rainfall
# after JMA
# https://www.jma.go.jp/jma/kishou/info/colorguide/120524_hpcolorguide.pdf

from matplotlib.colors import LinearSegmentedColormap

def Colormap_JMA(n_bin=50):
    cdict = {'red':   [(0,0.95,0.95),
                       (0.01,0.95,0.63),
                       (0.05,0.63,0.13),
                       (0.1,0.13,0),
                       (0.2,0,0.98),
                       (0.3,0.98,1),
                       (0.5,1,1),
                       (0.8,1,0.71),
                       (1,0.71,0.71)],
             'green': [(0,0.95,0.95),
                       (0.01,0.95,0.82),
                       (0.05,0.82,0.55),
                       (0.1,0.55,0.25),
                       (0.2,0.25,0.96),
                       (0.3,0.96,0.6),
                       (0.5,0.6,0.16),
                       (0.8,0.16,0),
                       (1,0,0)],
             'blue':  [(0,1,1),
                       (0.01,1,1),
                       (0.05,1,1),
                       (0.1,1,1),
                       (0.2,1,0),
                       (0.3,0,0),
                       (0.5,0,0),
                       (0.8,0,0.41),
                       (1,0.41,0.41)]}
    cmap_name="precip"
    cm = LinearSegmentedColormap(cmap_name, cdict, N=n_bin)
    return(cm)

