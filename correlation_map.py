import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
# Correlation in properties
# =============================================================================
def correlation_map(data,axes,title,filename):
    
    corre = data.corr(method='pearson')
   
    fig,ax = plt.subplots()
    im = ax.imshow(corre, cmap='seismic')
    ax.set_xticks(np.arange(len(axes)))
    ax.set_yticks(np.arange(len(axes)))

    ax.set_xticklabels(axes,fontsize=10)
    ax.set_yticklabels(axes,fontsize=10)

    plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
    ax.set_title(title)
    fig.tight_layout()
    cbar = fig.colorbar(im)
    cbar.set_label('Pearson Correlation', rotation=270,
                   labelpad=10)
    plt.savefig(filename,dpi=300)
    plt.show()
    return;
# =============================================================================
# correlation map for properties   
# =============================================================================
prop_data = pd.read_csv('properties.csv')
title = 'Property correlation map'
axes = ['CRT','CT','RS','RT']
filename='correlation_prop.png'
correlation_map(prop_data,axes,title,filename)
## =============================================================================
## correlation map for preparation treatments
## =============================================================================
prep_data = pd.read_csv('preparation.csv')
# replace the cooling discription to actual cooling rate number
# reference cooling rate in Lai_1991
prep_data.replace('Furnace cool',0.03, inplace=True) 
prep_data.replace('Air cool',0.88, inplace=True)
prep_data.replace('Water quench',40.00, inplace=True)
prep_data.replace('Oil quench',16.67, inplace=True)

title = 'Preparation correlation map'
axes = ['NTE','NTI','TTE','TTI','TR','ATE','ATI']
filename='correlation_prepa.png'
correlation_map(prep_data,axes,title,filename)
# =============================================================================
# correlation map of compositions
# =============================================================================
comp_data = pd.read_csv('composition.csv')
title = 'Composition correlation map'
axes = ['C','Si','Mn','P','S','Cr','Mo','W','Ni','Cu','V','Nb','N','Al','B','Co','Ta','O','Re','Fe']
filename='correlation_composition.png'
correlation_map(comp_data,axes,title,filename)
# =============================================================================
# correlation compositions with properties 
# =============================================================================
comp_prop_data = pd.concat([comp_data, prop_data],axis=1)
title = 'Composition and properties correlation map'
axes = ['C','Si','Mn','P','S','Cr','Mo','W','Ni','Cu','V','Nb','N','Al','B','Co','Ta','O','Re','Fe','CRT','CT','RS','RT']
filename=title
correlation_map(comp_prop_data,axes,title,filename)
# =============================================================================
# correlation treatment with properties
# =============================================================================
prep_prop_data = pd.concat([prep_data,prop_data],axis=1)
title = 'Preparation and properties correlation map'
axes = ['NTE','NTI','TTE','TTI','TR','ATE','ATI','CRT','CT','RS','RT']
filename = title
correlation_map(prep_prop_data,axes,title,filename)

# =============================================================================
# ANOVA analysis 
# =============================================================================

















