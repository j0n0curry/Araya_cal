import pandas as pd
import streamlit as st
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import math
import seaborn as sns 
pd.set_option('display.max_columns', None)
sns.set_theme()


st.set_page_config(layout="wide")


version = 0.1


    
#Set up main page of application / Header info / data collection / file selection / remove files / Reset

#Main landing page greating / info


st.title('Araya comparison analysis tool ' +str(version))

st.write("Upload parsed csv files for comparison - use ePCR Viewer to download 'all data csv' for each Araya before starting here - do this for each read set for your 'test Araya'")
st.write("Link to ePCR viewer [Here](https://share.streamlit.io/j0n0curry/new-calibration-test/main/ePCR_new_cal_OX_viewv1.py)")
st.write('click on images to increase size')
####start loading data and set up for persistent dataframe via cache ans session state - TODO

@st.cache
def load_data(data):
    df = pd.read_csv(data)
    df = normalise_values(df)
    df['UID'] = df['Run_ID'].astype(str) + df['Well']
    return(df)

#####start of stateful - found streamlit blog - TODO - assign dropdown values and changes in data frame to select variables
##### dataframe will persistent through call session state and assigning the cvs to this - TODO - parser with caching for files#
##### after upload - likely pd.concat - reassign comp to new variable - cache here or allow mutation of orginal - horrible and messy 
##### rerun of small merge function to preseve and assign new df and then allow selection of columns or arrays etc....
def persist_dataframe():
    # drop column from dataframe
    delete_col = st.session_state["delete_col"]
    if delete_col in st.session_state["updated_df1"]:
        st.session_state["updated_df1"] = st.session_state["updated_df1"].drop(
            columns=[delete_col]
        )
    else:
        st.sidebar.warning("Column previously deleted. Select another column.")
    with col2:
        st.write("Updated dataframe")
        st.dataframe(st.session_state["updated_df1"])
        st.write(st.session_state["updated_df1"].columns.tolist())
  
######normalise values for z scores - absolute deviations from the mean - takes in all processes but could be used to call after session
######state is called to calculate abs deviations for a called group using result as a filter
def normalise_values(df): 
    df['norm_zscore'] = (df.ROX_RFU - df.ROX_RFU.mean())/df.ROX_RFU.std(ddof=0)
    df['cfo_zscore'] = (df.VIC_RFU - df.VIC_RFU.mean())/df.VIC_RFU.std(ddof=0)
    df['fam_zscore'] = (df.FAM_RFU - df.FAM_RFU.mean())/df.FAM_RFU.std(ddof=0)
    df['nFAM_zscore'] = (df.norm_N_Cov - df.norm_N_Cov.mean())/df.norm_N_Cov.std(ddof=0)
    df['nVIC_zscore'] = (df.norm_RNaseP - df.norm_RNaseP.mean())/df.norm_RNaseP.std(ddof=0)
    return(df)


######function to create percentage chage but avoids 0 0 div problems 
def pct_change(first, second):
        diff = second - first
        change = 0
        try:
            if diff > 0:
                change = (diff / first) * 100
            elif diff < 0:
                diff = first - second
                change = -((diff / first) * 100)
        except ZeroDivisionError:
            return float('inf')
        return change


#assign confidence elipse - calculate covariance matrix - use covar to calculate Pearson later. 

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
    
    


###### concordance set generate concordance results by merging two called sets of data from each araya and assessing the results
###### provides plot bar of concordance - table of concordance results - plot of True and false alignment by group using ll data
###### by Araya overlay - easy to spot areas where sample calles are discordant - TODO - xport table -with wells? Not sure important 
###### might be?


#@st.cache(suppress_st_warning=True)
def concord_set(df1,df2):
    ####take two araya data and merge in to new set - do no mutate original post cache and session state. 
    cd_set = df1.merge(df2[['FAM_RFU', 'VIC_RFU','ROX_RFU','norm_N_Cov','norm_RNaseP', 'Result']], how = 'left', left_on = df1['UID'], right_on = df2['UID'])
    st.dataframe(cd_set.head(2))
    cd_set['concord'] = cd_set['Result_x'] == cd_set['Result_y']
    st.write(cd_set.concord.describe())
    col1,col2 = st.columns(2)
    
    #st.write(cd_set.concord.unique())
    #create eval between two states
    pos_concord = cd_set.concord.value_counts()[True]
    neg_concord = cd_set.concord.value_counts()[False]
    
    #st.write(pos_concord)
    #generate concordance eval result - compare Test to Reference
    percent_concord = round(np.abs(pct_change(pos_concord, neg_concord)),2)
    
    col1, col2, col3 = st.columns(3)

    #plot bar chat or concordance with values
    plt.figure(figsize = [10,8])
    plt.bar('Concordant', pos_concord, alpha=0.5, label=str(pos_concord))
    plt.bar('Discordant', neg_concord, alpha=0.5, label=str(neg_concord))
        
    plt.legend(loc='upper right')

    plt.title('Total number of analysed wells by result ' + str(len(df1)) + ' with a percentage of concordance between Arayas ' + str(percent_concord) + '%')
    plt.legend()
    with col1:
        st.pyplot(plt)

    #plot overlay of two Arays with overlay 
    fig, ax_nstd = plt.subplots(figsize=(10, 8))

    #data = Araya1[Araya1['norm_N_Cov'] <= 20]

    colors = {False:'red', True:'green'}

    x = cd_set['norm_RNaseP_x']

    y = cd_set['norm_N_Cov_x']

    x2 = cd_set['norm_RNaseP_y']
    y2 = cd_set['norm_N_Cov_y']

    x1 = cd_set['norm_RNaseP_x']
          
    y1 = cd_set['norm_N_Cov_x']

    #spearman = stats.spermanr

    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)

    #z_factor_score = ('Z factor - signal to noise score is ' + str(round(z_fac(control['norm_N_Cov'], NTC['norm_N_Cov'], 3), 3)) + ' Z Factor of less than 0.5 is poor / between 0.5 and 0.75 is adequate / score of greater than 0.75 is ideal: for Z Factor 3 SD of 1 - 3* [pos sd - neg sd]/[pos mu - neg mu]')

    ax_nstd.scatter(x1, y1, s=10, c = cd_set.concord.map(colors), label = 'True')

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick', linewidth=3.0)
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--', linewidth=3.0)
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':', linewidth=3.0)

    ax_nstd.scatter(x2, y2, s=10, c= cd_set.concord.map(colors))

    #confidence_ellipse(x2, y2, ax_nstd, n_std=1,
     #                  label=r'$1\sigma$', edgecolor='red', linewidth=3.0)
    #confidence_ellipse(x2, y2, ax_nstd, n_std=2,
     #                  label=r'$2\sigma$', edgecolor='grey', linestyle='--', linewidth=3.0)
    #confidence_ellipse(x2, y2, ax_nstd, n_std=3,
     #                  label=r'$3\sigma$', edgecolor='green', linestyle=':', linewidth=3.0)

    ax_nstd.set_xlabel('normalised VIC')
    ax_nstd.set_ylabel('normalised FAM')
    ax_nstd.set_title('A2 - Positive Paitent Samples - FAM / VIC RFU ')# + str(z_factor_score))

    ax_nstd.legend()
    #plt.savefig('A2_all~data.png')


    with col2:
        st.pyplot(plt)



    fig, ax_nstd = plt.subplots(figsize=(10, 8))

    data = df1[df1['norm_N_Cov'] <= 20]

    x = df1['norm_RNaseP']

    y = df1['norm_N_Cov']

    x2 = df2['norm_RNaseP']
    y2 = df2['norm_N_Cov']

    x1 = df1['norm_RNaseP']
          
    y1 = df1['norm_N_Cov']

    #spearman = stats.spermanr

    ax_nstd.axvline(c='black', lw=2)
    ax_nstd.axhline(c='black', lw=2)



    #z_factor_score = ('Z factor - signal to noise score is ' + str(round(z_fac(control['norm_N_Cov'], NTC['norm_N_Cov'], 3), 3)) + ' Z Factor of less than 0.5 is poor / between 0.5 and 0.75 is adequate / score of greater than 0.75 is ideal: for Z Factor 3 SD of 1 - 3* [pos sd - neg sd]/[pos mu - neg mu]')

    ax_nstd.scatter(x1, y1, s=10, label = 'Reference')

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label=r'$1\sigma$ Reference', edgecolor='firebrick', linewidth=3.0)
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label=r'$2\sigma$ Reference', edgecolor='fuchsia', linestyle='--', linewidth=3.0)
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                       label=r'$3\sigma$ Reference', edgecolor='blue', linestyle=':', linewidth=3.0)

    ax_nstd.scatter(x2, y2, s=10, label = 'Test')

    confidence_ellipse(x2, y2, ax_nstd, n_std=1,
                       label=r'$1\sigma$ Test', edgecolor='red', linewidth=3.0)
    confidence_ellipse(x2, y2, ax_nstd, n_std=2,
                       label=r'$2\sigma$ Test', edgecolor='grey', linestyle='--', linewidth=3.0)
    confidence_ellipse(x2, y2, ax_nstd, n_std=3,
                       label=r'$3\sigma$ Test', edgecolor='green', linestyle=':', linewidth=3.0)

    ax_nstd.set_xlabel('normalised VIC')
    ax_nstd.set_ylabel('normalised FAM')
    ax_nstd.set_title('Reference vs Test - Data Overlay - FAM / VIC RFU ')# + str(z_factor_score))
    ax_nstd.legend(loc='upper right')
    ax_nstd.legend()



    with col3:
        st.pyplot(plt)
        
    new = cd_set[cd_set['concord'] == False]
    st.write('Result groups discordant')
    st.dataframe(new.Result_x.value_counts())
     
    return(cd_set)
    
    






uploaded_file1 = st.sidebar.file_uploader("Uploaded Reference Araya", type=['csv'], accept_multiple_files=False, key = 'key')
uploaded_file2 = st.sidebar.file_uploader("Uploaded Comparator Araya", type=['csv'], accept_multiple_files=False, key = 'key')



df1 = load_data(uploaded_file1)

st.dataframe(df1.head())
    
df2 = load_data(uploaded_file2)
st.dataframe(df2.head())
if uploaded_file1 and uploaded_file2 is not None:
    concord_set(df1,df2)

# callback to session_state
# initialize session state variable
if "updated_df1" not in st.session_state:
    st.session_state.updated_df1 = df1
    
if "updated_df2" not in st.session_state:
    st.session_state.updated_df2 = df2







def conf_plot(data, data2, name, n, k, m, d, *args, **kwargs):
    
    
    
    fig, ax_nstd = plt.subplots(figsize=(10, 8))
    data = data
    data2 = data2
    n = str(n)
    k = str(k)
    x = data[n]
    y = data2[k]
    name = str(name)
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
    cov = np.cov(x, y)
    p = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
  


    


    col1, col2, col3 = st.columns(3)

    median_diff = pct_change(x.median(),y.median())
    print(median_diff)

    ax_nstd.scatter(x, y, s=8)

    confidence_ellipse(x, y, ax_nstd, n_std=1,
                       label=r'$1\sigma$', edgecolor='firebrick', linewidth=3.0)
    confidence_ellipse(x, y, ax_nstd, n_std=2,
                       label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--', linewidth=3.0)
    confidence_ellipse(x, y, ax_nstd, n_std=3,
                       label=r'$3\sigma$', edgecolor='blue', linestyle=':', linewidth=3.0)

    ax_nstd.set_xlabel(str(n) + str(m))
    ax_nstd.set_ylabel(str(k) + str(d))
    ax_nstd.set_title(str(name) + ' Araya comparison - all sample plot  ' + str(n) + '/' + str(k) + ' Pearson Correlation ' + str(round(p,2)))# + 'Median % difference to comparator ' + str(round(median_diff),1)))
    ax_nstd.legend()
    
    with col1:
        st.pyplot(plt)
    

    bins = 100
    
    plt.figure(figsize = [10,8])
    plt.hist(x, bins, alpha=0.5, label=str(m))
    plt.hist(y, bins, alpha=0.5, label=str(d))
    
    plt.legend(loc='upper right')
    plt.xlabel(str(n) + ' Araya')

    plt.title(str(name) + ' Signal distribution overlay ' + str(n) + '/' + str(k))
    plt.legend()
    
    with col2:
        st.pyplot(plt)
    
    m1 = np.asarray(x)
    m2 = np.asarray(y)

    f, ax = plt.subplots(1, figsize = (10,8))
    sm.graphics.mean_diff_plot(m1, m2,sd_limit=3, ax = ax)
    plt.title(str(name) + ' Difference Plot ' + str(n) + '/' + str(k) + ' Positive adjust down / negative adjust up - Difference of the median value ' + str(round(median_diff,2)) + '%')
    with col3:
        st.pyplot(plt)
    return(plt.title)
   # fig.savefig('efig.pdf')



conf_plot(df1, df2, 'VIC', 'VIC_RFU', 'VIC_RFU', 'Reference', 'Test')
    

conf_plot(df1, df2, 'FAM', 'FAM_RFU', 'FAM_RFU', 'Reference', 'Test')
    

    
conf_plot(df1, df2, 'ROX', 'ROX_RFU', 'ROX_RFU', 'Reference', 'Test')


conf_plot(df1, df2, 'nFAM', 'norm_N_Cov', 'norm_N_Cov', 'Reference', 'Test')
    

conf_plot(df1, df2, 'nVIC', 'norm_RNaseP', 'norm_RNaseP', 'Reference', 'Test')
    
    
#conf_plot(df1, df2, 'ROX_standard deviations', 'norm_zscore', 'norm_zscore', 'Reference', 'Test')
#conf_plot(df1, df2, 'FAM standard deviations', 'fam_zscore', 'fam_zscore', 'Reference', 'Test')
#conf_plot(df1, df2, 'VIC standard deviations', 'cfo_zscore', 'cfo_zscore', 'Reference', 'Test')
