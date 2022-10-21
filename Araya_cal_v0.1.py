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


version = 0.3


    
#Set up main page of application / Header info / data collection / file selection / remove files / Reset

#Main landing page greating / info


st.title('Araya comparison analysis tool ' +str(version))

st.write('Please ensure you have read SOP - Araya Characterization / N# - version XXXXX prior to comencing the report') 

st.write("Upload parsed csv files for comparison - use ePCR Viewer to download 'all data csv' for each Araya before starting here - do this for each read set for your 'test Araya'")
st.write("Link to ePCR viewer [Here](https://share.streamlit.io/j0n0curry/new-calibration-test/main/ePCR_new_cal_OX_viewv1.py)")
st.write('Uploaded files - Reference is the Araya being used to compare to while test is the Araya being adjusted - no warranties supplied or implied for this application')
st.write('click on images to increase size')

 
####start loading data and set up for persistent dataframe via cache ans session state - TODO

        
def assign_tape_dye(df):
    Arrays = {0:'FAM_RFU', 1:'FAM_RFU', 2:'FAM_RFU',
              3:'VIC_RFU', 4:'VIC_RFU', 5: 'VIC_RFU', 
              6 :'ROX_RFU', 7 : 'ROX_RFU', 8 :'ROX_RFU'}
    sorted_tapes = df.Run_ID.unique()
    ordered_tapes = {k: v for v, k in enumerate(sorted_tapes)}

    df['dye_type'] = df['Run_ID'].map(ordered_tapes)
    df['dye_type'] = df['dye_type'].map(Arrays)
    return(df)



def normalise_values(df):
    ROX_median = df[df['dye_type'] == 'ROX_RFU']['ROX_RFU'].median()
    df['nVIC'] = df[df['dye_type'] == 'VIC_RFU']['VIC_RFU'] / ROX_median
    df['nFAM'] = df[df['dye_type'] == 'FAM_RFU']['FAM_RFU'] / ROX_median
    return(df)



@st.cache
def load_data(data):
    df = pd.read_csv(data)
    df = assign_tape_dye(df)
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
        #st.write(change)
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



st.sidebar.header('Reference Araya')
Araya_Ref = st.sidebar.text_input('Please enter the Serial Number of the Reference Araya here')

uploaded_file1 = st.sidebar.file_uploader("Uploaded Reference Araya", type=['csv'], accept_multiple_files=False, key = 'key')


st.sidebar.header('Test Araya')
Araya_Test = st.sidebar.text_input('Please enter the Serial Number of the Test Araya here')

uploaded_file2 = st.sidebar.file_uploader("Uploaded Comparator Araya", type=['csv'], accept_multiple_files=False, key = 'key')





if uploaded_file1 and uploaded_file2 is not None:
    df1 = load_data(uploaded_file1)
    df2 = load_data(uploaded_file2)
   
else:
    st.warning('Please upload Araya files')
    st.stop() 

# callback to session_state
# initialize session state variable
if "updated_df1" not in st.session_state:
    st.session_state.updated_df1 = df1
    
if "updated_df2" not in st.session_state:
    st.session_state.updated_df2 = df2


table = []





def conf_plot(data, data2, name, n, k, m, d, *args, **kwargs):
    

    fig, ax_nstd = plt.subplots(figsize=(10, 8))
    
    n = str(n)
    k = str(k)
    data = data
    data2 = data2
    x = data[n]
    y = data2[k]
    name = str(name)
    ax_nstd.axvline(c='grey', lw=1)
    ax_nstd.axhline(c='grey', lw=1)
    cov = np.cov(x, y)
    corr_m = np.corrcoef(x,y)
    corr_xy = corr_m[0,1]
    r_squared = corr_xy**2
    p = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    corr_m = np.corrcoef(x,y)
    corr_xy = corr_m[0,1]
    r_squared = corr_xy**2
    

    col1, col2, col3 = st.columns(3)

    median_diff = pct_change(x.median(),y.median())
    
    #st.write(x.median())
    #st.write(y.median())
    
    frame = pd.DataFrame([[name, np.round(x.median(),0), y.median(), median_diff, p, r_squared]],
                       columns=['Dye Channel', 'Reference Median', 'Test Median', 'Percentage Median Difference', 'Pearson Correlation', 'R_squared'])
    table.append(frame)

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


def pass_fail(row):

    if row['Percentage Median Difference'] < 2.5 and row['Percentage Median Difference'] >-2.5:
        return('Pass')
    
    else:
        return('Fail')
        

df1 = normalise_values(df1)

df2 = normalise_values(df2)
    

conf_plot(df1[df1['dye_type'] == 'FAM_RFU'],df2[df2['dye_type'] == 'FAM_RFU'], 'FAM', 'FAM_RFU', 'FAM_RFU', 'Reference', 'Test')
    

conf_plot(df1[df1['dye_type'] == 'VIC_RFU'],df2[df2['dye_type'] == 'VIC_RFU'], 'VIC', 'VIC_RFU', 'VIC_RFU', 'Reference', 'Test')
    

    
conf_plot(df1[df1['dye_type'] == 'ROX_RFU'],df2[df2['dye_type'] == 'ROX_RFU'], 'ROX', 'ROX_RFU', 'ROX_RFU', 'Reference', 'Test')
    

conf_plot(df1[df1['dye_type'] == 'FAM_RFU'], df2[df2['dye_type'] == 'FAM_RFU'], 'normalised FAM', 'nFAM', 'nFAM', 'Reference', 'Test')
conf_plot(df1[df1['dye_type'] == 'VIC_RFU'], df2[df2['dye_type'] == 'VIC_RFU'], 'normalised VIC', 'nVIC', 'nVIC', 'Reference', 'Test')




final = pd.concat(table)

final.set_index('Dye Channel', inplace = True)


final['Pass Fail']= final.apply(lambda row: pass_fail(row), axis = 1)

machine_info = pd.DataFrame([[Araya_Ref, Araya_Test]], columns = ['Reference Araya Serial Number', 'Test Araya Serial Number'])


st.table(machine_info)

st.table(final)



if 'Fail' in final['Pass Fail'].unique():
    st.header('Verification Failed')
else:
    st.header('Verification Passed')
