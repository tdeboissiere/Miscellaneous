import matplotlib.pyplot as plt


def prettify_boxplot(bp, fill_color, linecolor, linewidth, linestyle):
    """ Prettify box plots

    Detail:
        Change color, line width etc to make it more appealing

    Args:
        bp = (matplotlib box)
        fill_color (str) box fill color
        linecolor (str) = line color for the box, whiskers, medians and caps
        linewidth (int) = line width for box, whiskers , mediansand caps
        linestyle (str) = linestyle for whiskers

    Returns:
        void

    Raises:
        void
    """

    for box in bp['boxes']:
        # change outline color
        box.set(color=linecolor, linewidth=linewidth)
        # change fill color
        box.set(facecolor=fill_color)

    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(
            color=linecolor,
            linewidth=linewidth,
            linestyle=linestyle)

    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color=linecolor, linewidth=linewidth)

    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color=linecolor, linewidth=linewidth)

    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='#e7298a', alpha=0.5)
