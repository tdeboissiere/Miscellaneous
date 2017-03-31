'''
Matplotlib 2.0 changes
'''
http://matplotlib.org/2.0.0/users/dflt_style_changes.html

'''
Plot a 2D matrix array
'''

import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt

H = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])  # added some commas and array creation code

fig, ax = plt.subplots()

image = H #np.random.uniform(size=(10, 10))
ax.imshow(image,   interpolation='nearest') #cmap=plt.cm.gray,

ax.set_title('dropped spines')

plt.imshow(H,   interpolation='nearest')
plt.colorbar(orientation='vertical')
plt.show()


'''
Subplots
'''

fig, (ax0, ax1)  = plt.subplots(nrows=2)
xmax=Pulse.TraceLength()

ax0.plot(np.arange(xmax),PulseA.Trace(),color="black")
ax0.set_title('Channel A spike')

ax1.plot(np.arange(xmax),PulseB.Trace(),color="red")
ax1.set_title('Channel B spike')

'''
No white space when saving figure
'''

    plt.gca().xaxis.set_major_locator(mp.ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(mp.ticker.NullLocator())

    plt.savefig(os.path.join(FLAGS.fig_dir, "current_batch_%s.png" % e), bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    # whitout pad_inches=0, small white boundary

'''
Grouped subplots
'''

def groupedGridSpec(ncols, nrows):

    list_gs = []
    list_ax = []

    size = 1. / (ncols + (ncols + 1) / 10.)
    offset = size / 10.

    lowers = [offset + i * (size + offset) for i in range(ncols + 1)]
    uppers = [i * (size + offset) for i in range(1, ncols + 1)]

    for c in range(ncols):

        gs = gridspec.GridSpec(4, 2)
        gs.update(left=lowers[c], right=uppers[c], wspace=0)
        for r in range(nrows):
            ax1 = plt.subplot(gs[r, 0])
            ax1.set_xticks([])
            ax1.set_yticks([])

            ax2 = plt.subplot(gs[r, 1])
            ax2.set_xticks([])
            ax2.set_yticks([])

            list_ax.append([ax1, ax2])

        list_gs.append(gs)

    return list_ax

plt.subplots_adjust(hspace=0.3)

'''
Clearing figures
'''
# cla() clears an axis, i.e. the currently active axis in the current figure. It leaves the other axes untouched.
# clf() clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
# close() closes a window, which will be the current window, if not specified otherwise.
# Which functions suits you best depends thus on your use-case.
# The close() function furthermore allows one to specify which window should be closed. The argument can either be a number or name given to a window when it was created using figure(number_or_name) or it can be a figure instance fig obtained, i.e., usingfig = figure(). If now argument is given to close(), the currently active window will be closed. Furthermore, there is the syntax close('all'), which closes all figures.

'''
Marker size in scatter plot
'''
# => specify s=...
scatter(x, y, s=500, color='green', marker='h')

'''
Move the legend
'''
ax1_1.legend(bbox_to_anchor=[0.33, 0.9])

'''
Properly normalise to 1
'''
weights = np.ones_like(myarray)/len(myarray)
plt.hist(myarray, weights=weights)

'''
Color scale
'''

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('red'), c('red'), 0.33, c('red'), c('green'), 0.76, c('green')])


'''
Nice plotting options
'''     

plt.figure()
plt.scatter(arr_heat_WIMP, arr_ion_fid_WIMP, color="0.6", s=1 )
plt.scatter(arr_heat, arr_ion_fid, s=20, c=arr['NN'], cmap=rvb, vmin = -1, vmax = 1)
cbar = plt.colorbar()
cbar.set_label('BDT output', labelpad = 15, fontsize= 18)
plt.xlabel('Heat (keV)', fontsize = 20)
plt.ylabel('Fiducial Ion (keV)', fontsize = 20)
plt.ylim([0,5])
plt.xlim([1,5])
plt.title(list_title[index] + "   " r'$M_{\chi}=$' + " " + str(mass) + " GeV", y=1.031, fontsize = 24)
plt.grid(True)
plt.savefig(fig_path + list_fig_name[index] + ".png")
plt.close("all")


'''
labels, legend title, nice layout
'''  

plt.xlabel("Heat (keVee)")
plt.legend(loc="lower right", title = "KS p-value", prop={"size":10})
plt.tight_layout()


'''
Use latex
'''  

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

'''
Group boxplot
''' 

bp1 = plt.boxplot(d_train[8],0, "", positions = [1], widths = 0.6, patch_artist=True)
bp2 = plt.boxplot(d_test[8],0, "", positions = [2], widths = 0.6, patch_artist=True)
# # second boxplot pair
plt.boxplot([d_train[7], d_test[7]],0, "", positions = [4, 5], widths = 0.6)

'''
Nice scatter plot with seaborn
''' 

sns.set_style("white")
sns.kdeplot(arr_WIMP, cmap = sns.dark_palette(sns.xkcd_rgb["ocean"], as_cmap = True))
plt.plot([0.5,3], [0,0], "k--")
sns.regplot(arr_heat[:,0], arr_heat[:,1], color =sns.xkcd_rgb["orange red"], fit_reg=False, scatter_kws={"s": 5, "alpha":0.5})
# plt.scatter(arr_heat[:,0], arr_heat[:,1], color = sns.xkcd_rgb["pale red"], s=2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.grid()
plt.xlabel("Combined Heat (keVee)", fontsize = 20)
plt.ylabel("Fiducial ionisation (keVee)", fontsize = 20)
plt.xlim([0.5, 3])
plt.ylim([-0.5, 2])
plt.tick_params(axis='both', which='major', labelsize=15) 
plt.tight_layout()
plt.savefig("plotname.png")

'''
Add arrow next to axis
'''
plt.figure(figsize=(20,20))
plt.imshow()
plt.xticks([])
plt.yticks([])
plt.ylabel("Varying categorical factor", fontsize=28, labelpad=60)

plt.annotate('', xy=(-0.1, 0), xycoords='axes fraction', xytext=(-0.1, 1),
             arrowprops=dict(arrowstyle="-|>", color='k', linewidth=4))
plt.savefig("../../figures/varying_categorical.png")
plt.clf()
plt.close()

'''
Scatter plot of images
'''

from skimage.transform import resize


def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, img.shape[:2])
    if min([w, h]) != size:
        if w <= h:
            img = resize(img, (int(round((h / w) * size)), int(size)))
        else:
            img = resize(img, (int(size), int(round((w / h) * size))))
    return img


def image_scatter(images, img_res, res=300, cval=1.):

    # Load and rescale data

    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    f2d = np.load("../../data/processed/X_tsne.npy")

    xx = f2d[:, 0]
    yy = f2d[:, 1]
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = sx / float(sy) * res
        res_y = res
    else:
        res_x = res
        res_y = sy / float(sx) * res

    canvas = np.ones((res_x + max_width, res_y + max_height, 3)) * cval
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords)**2)
        y_idx = np.argmin((y - y_coords)**2)
        canvas[x_idx:x_idx + w, y_idx:y_idx + h] = image

    plt.figure(figsize=(40, 40))
    plt.imshow(canvas)
    plt.savefig("/home/tmain/Pictures/tsne_scattplot.png")
    plt.tight_layout()
    
'''
Same binning for histograms
'''

bins = np.histogram(np.ravel(y_pred), bins=40)[1]  # get the bin edges
plt.hist(y_pred[y_true == 0][:, 1], bins=bins, alpha=0.5, label="bla")
plt.hist(y_pred[y_true == 1][:, 1], bins=bins, alpha=0.5, label="bli")


'''
Plot decoration
'''

def process_plot(xlabel, ylabel, fontsize, labelsize, save_dir, xlim=None, ylim=None):

    plt.xlabel(xlabel, fontsize=fontsize, labelpad=20)
    plt.ylabel(ylabel, fontsize=fontsize, labelpad=20)
    plt.legend(fontsize=fontsize, loc="best")
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tick_params(axis='both', which='minor', labelsize=labelsize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.savefig(save_dir)
    plt.clf()
    plt.close()

'''
Scatte rlegend size
'''
plt.legened(markerscale=XXX)
