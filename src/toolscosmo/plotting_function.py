import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

def study_param_dependence(metric, param_dict, fig_title=None,
                           n_bins=10, metric_scale='linear'):
    """
    Plot the metric by varying the parameters.

    Parameters
    ----------
    metric : function
        A function that takes the keys as input and 
        gives a number that is compared.
    param_dict : dictionary
        A dictionary with paramter names as keys pointing
        to a range to probe. 
    fig_title: string
        The figure title.
    n_bins: int
        Number of bins of each parameter range.
    metric_scale: string
        Scaling the plotted metric.

    Returns
    -------
    matplotlib figure
        The figure plotting the variations.

    Examples
    --------
    >>> def sum_diff_wCDM(w0): return np.sum(wCDM(zs, w0=w0)/wCDM(zs)-1)
    >>> param_dict = {'$\omega_0$': [-2,0]}
    >>> fig, axs = study_param_dependence(sum_diff_wCDM, param_dict, fig_title='$\omega$CDM')
    >>> plt.show()
    """

    if metric_scale == 'linear':
        norm = Normalize()  # Linear scaling
    elif metric_scale == 'logarithmic':
        norm = LogNorm()  # Logarithmic scaling
    elif metric_scale == 'symlog':
        norm = SymLogNorm(linthresh=1e-10, linscale=1.0, vmin=None, vmax=None, base=10)  # Symlog scaling
    else:
        raise ValueError("Invalid metric_scale. Use 'linear', 'logarithmic', or 'symlog'.")

    param_names = list(param_dict.keys())
    n_params = len(param_names)

    if n_params==1:
        fig, axs = plt.subplots(1,1,figsize=(7,6))
        fig.suptitle(fig_title, fontsize=16)
        param_name_i = param_names[0]
        xmin, xmax = param_dict[param_name_i]
        xx = np.linspace(xmin,xmax,n_bins)
        yy = np.vectorize(metric)(xx)
        axs.plot(xx, yy)
        axs.set_yscale(metric_scale)
        axs.set_xlabel(param_name_i, fontsize=15)
        axs.set_ylabel(r'$\sum \left(\frac{\rm Model}{\rm \Lambda CDM}-1\right)$', fontsize=15)
    else:
        fig, axs = plt.subplots(n_params-1,n_params-1,figsize=(8,7))
        fig.suptitle(fig_title, fontsize=16)
        for ii,param_name_i in enumerate(param_names[:-1]):
            for jj,param_name_j in enumerate(param_names[1:]):
                print(ii,jj,param_name_i,param_name_j)
                xmin, xmax = param_dict[param_name_i]
                ymin, ymax = param_dict[param_name_j]
                xx, yy = np.meshgrid(np.linspace(xmin,xmax,n_bins),np.linspace(ymin,ymax,n_bins), indexing='ij')
                zz = np.vectorize(metric)(xx,yy)
                try: ax = axs[ii,jj]
                except: ax = axs
                if ii>jj:
                    ax.set_visible(False)
                else:
                    pcm = ax.pcolor(xx, yy, zz, norm=norm, cmap='viridis')
                    cbar = fig.colorbar(pcm, ax=ax)
                    cbar.set_label(r'$\sum \left(\frac{\rm Model}{\rm \Lambda CDM}-1\right)$', fontsize=15)
                    ax.set_xlabel(param_name_i, fontsize=15)
                    ax.set_ylabel(param_name_j, fontsize=15)
    plt.tight_layout() 
    return fig, axs
