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
Scatter legend size
'''
plt.legened(markerscale=XXX)

'''
Grouped gridspec
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


"""
Tick label size
"""

import matplotlib as mpl
label_size = 8
mpl.rcParams['xtick.labelsize'] = label_size

or

[tick.label.set_fontsize(6) for tick in ax.xaxis.get_major_ticks()]
[tick.label.set_fontsize(6) for tick in ax.yaxis.get_major_ticks()]
[tick.label.set_fontsize(6) for tick in ax.zaxis.get_major_ticks()]


"""
Avoid fading of scatter marker in 3D
"""

ax.scatter(x, y, z, depthshade=0)

"""
Adjust title y position
"""

ax.set_title("bla", y=1.2)

"""
Hollow marker in scatter
"""

ax.scatter(x, y, facecolors="none")

"""
Axis equal aspect + automatic data boundaries
"""
ax.set_aspect('equal', 'box')

"""
Control 3D axis orientation
"""
ax.view_init(elev=-64., azim=97)

"""
Circles in 3D
"""

def rotation_matrix(v1,v2):
    """
    Calculates the rotation matrix that changes v1 into v2.
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    cos_angle = np.dot(v1,v2)
    d = np.cross(v1,v2)
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        M = np.identity(3) if cos_angle > 0. else -np.identity(3)
    else:
        d /= sin_angle

        eye = np.eye(3)
        ddt = np.outer(d, d)
        skew = np.array([[0, d[2], -d[1]],
                         [-d[2], 0, d[0]],
                         [d[1], -d[0], 0]], dtype=np.float64)

        M = ddt + cos_angle * (eye - ddt) + sin_angle * skew

    return M


def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str:  # Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1,0,0), index)

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    M = rotation_matrix(normal,(0, 0, 1))  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta
    
c = Circle((0,0), v.pupil_radius, edgecolor="C0", facecolor='C0', alpha=.2)
ax.add_patch(c)
pathpatch_2d_to_3d(c, z=0, normal=xxx)
pathpatch_translate(c, np.array([x,y,z]))

"""
Set legend line width
"""

leg = plt.legend()
# get the individual lines inside legend and set line width
for line in leg.get_lines():
    line.set_linewidth(4)

"""
Add path effects
"""

ax.plot(x,
        y,
        linewidth=3,
        path_effects=[pe.Stroke(linewidth=3.5, foreground='k'), pe.Normal()])

"""
Matplotlib scatter edgecolor and width
"""
ax.scatter(x, y,
       s=50,
       edgecolor="black",
       linewidth=0.5)
'''
Create legend from custom artist/label lists
'''
l_pred = plt.Line2D((0,1),(0,0), color='C0', linewidth=3)
l_LM = plt.Line2D((0,1),(0,0), color='C2', linewidth=3)
l_gt = plt.Line2D((0,1),(0,0), color='C1', linewidth=3)


bbox_to_anchor = (2.5, 1.6)
list_ax[0].legend([l_pred, l_LM, l_gt], ['NN prediction', "LM prediction", 'Ground truth'],
                  fontsize=18,
                  bbox_to_anchor=bbox_to_anchor)


'''
Create 3D scatter plot with 2D planes
'''

fig = plt.figure()
plt.suptitle(title, y=0.95, fontsize=14)
ax = fig.add_subplot(111, projection='3d')

# X, Y, Z = np.arrays

ax.scatter(X,
           Y,
           Z,
           color="gray",
           alpha=0.3)

# Show the plane found by PCA
# Plane equation is n[0]*x + n[1]*y + n[2]*z + cst = 0
# Where n is a normal vector to the plane
# The plane is expected to roughly pass through the origin, hence
# we set cst = 0

# We compute the normal vector by taking the cross product of the 2 first pca components
u = pca.components_[0]
v = pca.components_[1]

n = np.cross(u, v)

# We can then build the surface corresponding to the plane
X = np.linspace(-0.3, 0.3, 10)
Y = np.linspace(-0.3, 0.3, 10)
X, Y = np.meshgrid(X, Y)

Z = - (n[0] * X + n[1] * Y) / n[2]

ax.plot_surface(X, Y, Z, alpha=0.3, color="C0")
# Patch to plot legend
patch_PCA = mpatches.Patch(color='C0', alpha=0.3, label="Plane found by PCA")

# Compute rotation matrix
pitch, yaw, roll = df[["pitch_deg_truth", "yaw_deg_truth", "roll_deg_truth"]].values[0]
pitch, yaw, roll = np.deg2rad(pitch), np.deg2rad(yaw), np.deg2rad(roll)
Rx = np.array([[1, 0, 0],
               [0, np.cos(pitch), -np.sin(pitch)],
               [0, np.sin(pitch), np.cos(pitch)]])
Ry = np.array([[np.cos(yaw),0, np.sin(yaw)],
               [0, 1, 0],
               [-np.sin(yaw), 0, np.cos(yaw)]])
Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
               [np.sin(roll), np.cos(roll), 0],
               [0, 0, 1]])

wRa = (Ry @ Rx @ Rz).T
nplane = np.dot(wRa, np.array([0, 1, 0]).reshape(3, 1))

# We can then build the surface correpsonding to the plane
X = np.linspace(-0.3, 0.3, 10)
Y = np.linspace(-0.3, 0.3, 10)
X, Y = np.meshgrid(X, Y)

Z = - (nplane[0] * X + nplane[1] * Y) / nplane[2]

ax.plot_surface(X, Y, Z, alpha=0.3, color="C3")
patch_Truth = mpatches.Patch(color='C3', alpha=0.3, label="Truth plane")

blue_line = mlines.Line2D([], [], color='C0', label='Line patch')

'''
Reverse color map
'''
def reverse_colour_map(cm, name="reversed_cm"):
    reverse = []
    k = []
    for key in cm._segmentdata:
        k.append(key)
        channel = cm._segmentdata[key]
        data = []
        for t in channel:
            data.append((1 - t[0],t[2],t[1]))
        reverse.append(sorted(data))
    linearl = dict(zip(k,reverse))
    reversed_cm = matplotlib.colors.LinearSegmentedColormap(name, linearl)
    return reversed_cm

'''
2D scatter
'''
def hexbin_scatter(datasets, d_cmap, with_points=False):

    # Check we have only one dataset
    assert len(datasets.keys()) == 1
    dataset_name = list(datasets.keys())[0]
    gaze_by_point = datasets[dataset_name]

    # Initialize hexbin parameters
    gridsize = None
    num_hexagones = 30

    # HexBin scatter
    for idx, pt in enumerate(gaze_by_point.keys()):

        measures = np.array

        xmin, xmax = measures[:, 0].min(), measures[:, 0].max()
        ymin, ymax = measures[:, 1].min(), measures[:, 1].max()

        # gridsize specifiew how many hexagons will be plotted
        # logic below makes sure that for each pt, we have approximately similar size hexagons
        if gridsize is None:
            grid_width = (xmax - xmin) / num_hexagones
            x_grid_size = num_hexagones
            y_grid_size = int((ymax - ymin) / grid_width)
            gridsize = (x_grid_size, y_grid_size)
        else:
            gridsize = int((xmax - xmin) / grid_width), int((ymax - ymin) / grid_width)

        # mincnt makes sure that bins without data are plotted in white
        plt.hexbin(measures[:, 0], measures[:, 1],
                   gridsize=gridsize, cmap=d_cmap[pt],
                   linewidths=0.1, mincnt=1, norm=LogNorm())



def hist2d_scatter(datasets, d_cmap, with_points=False):

    # Check we have only one dataset
    assert len(datasets.keys()) == 1
    dataset_name = list(datasets.keys())[0]
    gaze_by_point = datasets[dataset_name]

    # Same binning for all 2D hist
    bins_x = np.arange(-1.5, 1, step=2.5 / 200)
    bins_y = np.arange(-0.5, 0.5, step=1. / 200)

    # Hist 2D scatter
    plt.hist2d(measures[:, 0], measures[:, 1], bins=[bins_x, bins_y], cmap=d_cmap[pt], norm=LogNorm())


