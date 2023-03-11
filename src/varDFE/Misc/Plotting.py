import matplotlib
matplotlib.use('Agg') # so it doesn't pop up graphics on hoffman (before importing pyplot)
import matplotlib.pyplot as plt
import dadi
# ggplot2-like plots
import pandas as pd
import numpy as np
import plotnine
from plotnine import ggplot, aes, geom_col, theme, theme_bw, element_blank, facet_wrap, geom_tile, labs, geom_line, geom_point, scale_fill_gradientn
from dadi import Spectrum
# supress plotnine future warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def plot_dadi_1d(outprefix, model, fs):
    fig=plt.figure(1)
    outputFigure=outprefix + '.png'
    dadi.Plotting.plot_1d_comp_multinom(model, fs)
    plt.savefig(outputFigure)

    return outputFigure


# format the 1d fs data to pandas dataframe
def fs2pddf(fs, popid):
    # check that things should be folded
    # if not fs.folded:
    #     raise TypeError('Spectrum need to be folded')
    outdf = pd.DataFrame.from_dict({'count':fs.data, 'mask':fs.mask, 'site_freq':range(fs.sample_sizes[0]+1)})
    # remove masked lines
    outdf = outdf.query('mask == False')
    # get percent plots
    outdf = outdf.assign(percent=outdf['count']/outdf['count'].sum())
    outdf = outdf.assign(pop=popid)
    return outdf

def ggplot_dadi_1d(outprefix, model, fs, yvar, returnplot = False):
    if yvar not in ('count','percent'):
        raise IOError('ggplot_dadi_1d input yvar only takes count or percent')
    datadf=fs2pddf(fs=fs,popid='data')
    modeldf=fs2pddf(fs=model,popid='model')
    plotdf=pd.concat(objs=(datadf,modeldf))
    pp=(ggplot(plotdf,aes(x='site_freq',y=yvar,fill='pop'))+
            geom_col(position='dodge')+
            theme_bw()+
            theme(legend_position='top',legend_title=element_blank()))
    if returnplot:
        return pp
    else:
        pp.save(filename=outprefix+'.pdf')
        return None

def ggplot_ref_spectra_1d(outprefix, dictspectra):
    """
    dictspectra: Output from dict_spectra(). spectra object indexed with gamma values.
    """
    gammalist=[]
    for gamma in dictspectra.keys():
        gammalist.append(gamma)
    gammalist.sort()
    # subset gamma list
    if 0 not in gammalist:
        raise ValueError('Input dictspectra does not have neutral spectrum')
    subgamma=gammalist[:2]+gammalist[gammalist.index(0)-2:gammalist.index(0)+3]+gammalist[-2:]
    subgamma_names = [("%0.3e" % val) for val in subgamma]

    spectra_dfl = []
    for ii, gamma in enumerate(subgamma):
        # construct a spectrum object for not neuspec
        if gamma == 0: pass
        else:
            fs=Spectrum(dictspectra[gamma])
        datadf=fs2pddf(fs=fs, popid=subgamma_names[ii])
        spectra_dfl.append(datadf)
    plotdf=pd.concat(objs=spectra_dfl)
    # assign list levels
    poplevels = pd.Categorical(plotdf['pop'],categories=subgamma_names)
    plotdf = plotdf.assign(pop=poplevels)
    pp=(ggplot(plotdf,aes(x='site_freq',y='percent',fill='pop'))+
            geom_col()+
            facet_wrap('~ pop') +
            theme_bw()+
            theme(legend_position='top',legend_title=element_blank()))
    pp.save(filename=outprefix+'.pdf',height=5,width=6,format='pdf')
    return pp

def ggplot_dfe_pdf(outprefix,pdf,params):
    xxn = np.logspace(-6,-1, num = 100) # only negative gammas
    xxb = np.concatenate([-np.logspace(-6,-1, num = 100),np.logspace(-6,-1, num = 100)]) # both negative and positive gammas
    if pdf.__name__ == 'lourenco_eq_pdf':
        xx = xxb
    else:
        xx = xxn
    # get yy
    yy = pdf(xx,params)
    plotdf = pd.DataFrame({'xx' : xx, 'yy' : yy})
    pp = (ggplot(plotdf,aes(x='xx',y='yy'))+
        geom_line()+
        theme_bw())
    pp.save(filename=outprefix+'.pdf',height=4,width=4,format='pdf')
    return None

def ggplot_gridsearch(outprefix, ll_griddf, ll_max):
    """
    ll_griddf: pd.DataFrame object with columns `[var0,var1,ll_model]`
    """
    # copy a plotdf
    plotdf = ll_griddf.__deepcopy__()
    labls = plotdf.columns.values.tolist()
    # get maximum values
    bestdf = pd.DataFrame(data = [ll_max], columns=['var0','var1','ll_model'])
    plotdf.rename(columns = {plotdf.columns[0]: 'var0',plotdf.columns[1]: 'var1'}, inplace = True)
    # better plotting
    pp=(ggplot(plotdf,aes(x='var0',y='var1',fill='ll_model'))+
        geom_tile()+
        geom_point(bestdf, shape = "o", size = 5, color = "blue", fill = "blue") +
        scale_fill_gradientn(colors = ["#606060","#FFFF00FF","#FF8000FF","#FF0000FF"],
            breaks = [-30,-100,-1000,-5000],limits = [-5000,-30],trans = 'pseudo_log') +
        labs(x=labls[0],y=labls[1])+
        theme_bw()+
        theme(legend_position='top',
            legend_title=element_blank(),
            aspect_ratio=1))
    pp.save(filename=outprefix+'.pdf',height=6,width=6,format='pdf')
    return pp



