

import matplotlib.pyplot as plt
import pandas as pd
import os 
import seaborn as sns
import numpy as np

shape_bias_src =  '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/shape_bias'

save_dir = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/plot_shape_bias_feb23'



# accuracy_graphs_sept22

# Sample file
# Category,Shape_Choices,Texture_Choices,Other_Choices,Shape_Bias,
# Total, 255,354,591,0.4187192118226601,
# airplane, 1, 23, 51, 0.041666666666666664, 
# bicycle, 23, 21, 31, 0.5227272727272727, 
# boat, 3, 21, 51, 0.125, 
# car, 26, 18, 31, 0.5909090909090909, 
# chair, 8, 21, 46, 0.27586206896551724, 
# dog, 7, 25, 43, 0.21875, 
# keyboard, 12, 28, 35, 0.3, 
# oven, 17, 23, 35, 0.425, 
# bear, 7, 24, 44, 0.22580645161290322, 
# bird, 7, 22, 46, 0.2413793103448276, 
# bottle, 41, 18, 16, 0.6949152542372882, 
# cat, 11, 25, 39, 0.3055555555555556, 
# clock, 50, 7, 18, 0.8771929824561403, 
# elephant, 4, 30, 41, 0.11764705882352941, 
# knife, 3, 23, 49, 0.11538461538461539, 
# truck, 35, 25, 15, 0.5833333333333334, 



# Shape bias and proportions change
def recalc_shape_bias(df):
    # example use: after grouping
    
    # won't change
    df['Correct_Choices'] = df['Shape_Choices'] + df['Texture_Choices']
    df['n_Choices'] = df['Correct_Choices'] + df['Other_Choices']
    # Change
    df['Prop_Correct'] = (df['Correct_Choices'] / df['n_Choices']) * 100
    # df['Shape_Bias'] = (df['Shape_Bias'] * 100).round(2)
    # Redo for certainty!
    df['Shape_Bias'] = ((df['Shape_Choices'] /df['Correct_Choices'] ) * 100).round(2)
    return df


# Added Categories
animal = ['dog', 'cat', 'bea', 'bir', 'ele']
inanimate = ['air', 'bic', 'boa', 'car', 'cha', 'key', 'ove', 'bot',  'clo', 'kni', 'tru']
inanimate_small = ['clo', 'bot', 'kni',]
inanimate_large = ['air', 'bic', 'boa', 'car', 'cha',  'key', 'ove', 'tru']

counter = 0
n_dfs = 0
fix_nets = []

tot_all_models = []
all_model_cats = []

# create one df with all models and group on basis of model_pth
for file_OR_dir in os.listdir(shape_bias_src):
    
    # Get path to file / directory 
    file_OR_dir_path = os.path.join(shape_bias_src, file_OR_dir)
    
    # Only keep files NOT subsirectories
    if os.path.isfile(file_OR_dir_path):
        
        # change variable name for clarity
        file_path = file_OR_dir_path
        
        # extract network name 
        suffix = '_16_class_IN.txt'
        len_suffix = len(suffix)
        prefix = 'shape_bias_'
        len_prefix = len(prefix)

        
        network_name = file_OR_dir[len_prefix:-len_suffix]
        print(network_name)
        
        if 'epoch' in network_name:
            
            # Load df (remove any leading spaces)
            df=pd.read_csv(file_path, sep=',', skipinitialspace=True)
            
            # Remove empty column
            df.drop(columns=['Unnamed: 5'], inplace=True)
            
            # Add columns to df
            # Add column to specify which network this is
            
            # Temporarily includes epoch too!
            df['Network']=network_name
            
            df['Category_short'] = df['Category'].str[:3]
            # df['Epoch'] = df['Network'].str.split("epoch").str[-1]
            df['Epoch'] = df['Network'].str.split("epoch").str[-1].astype(int)
            
            # Remove Epoch from netwrok
            # Truncated names!
            # df['Network']=df['Network'].str.rsplit("epoch").str[0].str.rstrip(to_strip='_')
            df['Network']=df['Network'].str[:-7].str.rstrip(to_strip='_')
            

            df = recalc_shape_bias(df)
            
            # Keep only rows with category total
            df_tot = df.loc[df['Category'] == 'Total']
            
            # keep row per category - removing total!
            df_cats = df.loc[df['Category'] != 'Total']



            # Check that the df has the expected number of columns (before adding to list to append!)
            # (as sometimes formatting went wrong)
            expected_columns= ['Category', 'Shape_Choices', 'Texture_Choices', 'Other_Choices',
                                'Shape_Bias', 'Network', 'Category_short',
                                'Correct_Choices' , 'n_Choices', 'Prop_Correct',
                                'Epoch']
            n_expected_columns = len(expected_columns)

            if len(df.columns) == n_expected_columns:
                tot_all_models.append(df_tot)
                all_model_cats.append(df_cats)
                
            elif len(df.columns) > n_expected_columns:
                counter +=1
                fix_nets.append(network_name)
                
            n_dfs += 1 
            
        # df2 = df.loc[:, ~df.columns.isin(['Fee'])]
        
        else:
            print(f'Cannot use this net- do not know what epoch was tested! {network_name}')
            # presumably both CMS nets were the final eoch - I think this was 120 - CHECK!!
            # CMC_AN
            # CMC_RN50v2
            
# # Tidying Code:
# Found df with leading space
# df_to_tidy = frame[frame[' Shape_Choices'].notnull()]
# print(df_to_tidy)
# print(df_to_tidy.columns)
# pd.set_option('display.max_colwidth', None)
# print(df_to_tidy['Network'].unique)


# Add column for epoch


print(n_dfs)
print(counter)
print(f'networks with incorrect formatting:\n{fix_nets}')
print(len(tot_all_models))

    
totals_df = pd.concat(tot_all_models, axis=0, ignore_index=True)
cats_df = pd.concat(all_model_cats, axis=0, ignore_index=True)
# totals_df.drop(columns=['Unnamed: 5'], inplace=True)
print(totals_df.columns)
print(totals_df)

print(cats_df)
print(cats_df['Epoch'])




# Add Animate, Inanimate distinction column for every category
cats_df['Animate'] = np.where(cats_df['Category_short'].isin(animal), 'animate', 'inanimate')
cats_df.loc[cats_df['Category_short'].isin(inanimate_small), 'Animate'] = 'inanimate_small'
cats_df.loc[cats_df['Category_short'].isin(inanimate_large), 'Animate'] = 'inanimate_large'
print(cats_df['Animate'])





# # Select an epoch
# totals_df_epoch60 = totals_df[totals_df['Epoch']==60]

print(len(totals_df))

# At threshold of 25 156 --> 145
# Set Threshold for proportion that need to be correct overall?
# 2/16 correct categoried => chance == (1/16 * 2 ), minimum threshold should be ? 12.5 !
# add threshold into filename!
# prop_correct_threshdold = 50
prop_correct_threshdold = 25
# prop_correct_threshdold = 0
print(f'len: {len(totals_df)}')
totals_df = totals_df[totals_df['Prop_Correct']> prop_correct_threshdold]
print(f'len: {len(totals_df)}')
# # Network
print(len(totals_df))



# List to choose values for subset
print('Choose values for subset:')
print(cats_df['Network'].unique())

# SELECT SUBSET OF NEWORKS ##########
# 1. All Networks
# net_subset = 'all_nets'
# all nets:
# subset_of_nets_cat = list(cats_df['Network'].unique())
# print(subset_of_nets_cat)
# subset_of_nets_tot = list(totals_df['Network'].unique())
# print(subset_of_nets_tot)
# print(set(subset_of_nets_cat) - set(subset_of_nets_tot)) #sup_RN50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30
# print(set(subset_of_nets_tot) - set(subset_of_nets_cat) )#empty



# # 2. 
# net_subset = 'blur_4x40_0x30_vs_0'
# net_subset = 'blur_4x40_0x30_vs_0_lr'
# subset_of_nets = [
#                 #   'from_sup_RN50_gauss_0_for_60_epoch', 
#                   'sup_RN50_gauss_0_for_60',
#                   'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2',
#                   'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30', #is this one correct?
                  
#                   'sup_RN50_gauss_0_for_60_epoch_lr_15',
#                   'sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15'
#                   ]

# 
# subset_of_nets = [
#                 # 'from_sup_RN50_gauss_0_for_60_epoch', 
#                 'sup_RN50_gauss_0_for_60',
#                 # 'sup_RN50_from_gauss_0_for_60_epoch',
#                 # 'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_60',
#                 'sup_RN50_from_gauss_4_for_10_epoch_to_gauss_0_for_50',
#                 'sup_RN50_from_gauss_4_for_20_epoch_to_gauss_0_for_40',
#                 'sup_RN50_from_gauss_4_for_15_epoch_to_gauss_0_for_45',
#                 'sup_RN50_from_gauss_4_for_20_epoch_to_gauss_0_for_40',
#                 # 'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2',
#                 # 'sup_RN50_from_gauss_4_for_40_epoch_to_gauss_0_for_15',
#                 'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30',
#                 'sup_RN50_from_gauss_4_for_35_epoch_to_gauss_0_for_25',
#                 'sup_RN50_from_gauss_4_for_50_epoch_to_gauss_0_for_10',
#                 'sup_RN50_from_gauss_4_for_40_epoch_to_gauss_0_for_20',
#                   ]



# # 2. Duration of Blur, fixed amount (4)
# net_subset = 'blur_4_to_0_modify_length'
# rename_chosen_models_exp = {
#                             "sup_RN50_gauss_0_for_60": "0:60",
#                             "sup_RN50_from_gauss_4_for_10_epoch_to_gauss_0_for_50": "10:50",
#                             "sup_RN50_from_gauss_4_for_20_epoch_to_gauss_0_for_40": "20:40",
#                             "sup_RN50_from_gauss_4_for_15_epoch_to_gauss_0_for_45": "15:45",
#                             "sup_RN50_from_gauss_4_for_20_epoch_to_gauss_0_for_40":"20:40",
#                             "sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30":"30:30",
#                             "sup_RN50_from_gauss_4_for_35_epoch_to_gauss_0_for_25":"35:25",
#                             "sup_RN50_from_gauss_4_for_40_epoch_to_gauss_0_for_20":"40:20",
#                             "sup_RN50_from_gauss_4_for_50_epoch_to_gauss_0_for_10":"50:10",
#                             }
# subset_of_nets = ["0:60","10:50","20:40","15:45","20:40","30:30","35:25","40:20","50:10"]



# 3. Amount of blur, fixed duration (4)
net_subset = 'blur_modify_amt_first_30_epochs'
rename_chosen_models_exp={
                            'sup_RN50_gauss_0_for_60':'0',
                            'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'4',
                            'sup_RN50_from_gauss_2_for_30_epoch_to_gauss_0_for_30':'2',
                            'sup_RN50_from_gauss_6_for_30_epoch_to_gauss_0_for_30':"6",
                          }
subset_of_nets = ['0','2','4','6']



# # 4. Blurry shape bias images to determine theoretical upper limit
# # TESTED ON BLURRY SET
# net_subset = 'blurryNets_tested-equivalentlyBlurryShapeBiasImages'

# # QUESTION posed from graphs - are the high res nets actually learning the shape better !!!!!
# rename_chosen_models_exp={
#                             'gauss_4_sup_RN50_gauss_0_for_60':'Tr-0(60E), Te-4',
#                             'sup_RN50_gauss_0_for_60':'Tr-0(60E), Te-0',
#                             'gauss_1_sup_RN50_gauss_1_for_60_epoch':'Tr-1(60E), Te-1',
#                             'gauss_2_sup_RN50_gauss_2_for_60':'Tr-2(60E), Te-2',
#                             # 'gauss_4_sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15':'Tr-4(30E)0(30E),Te-4(lr)', #NOT TURNING UP!
#                             'gauss_4_sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15':'Tr-4(30E)-->0(30E), Te-4 (lr)', #NOT TURNING UP!
#                             # 'gauss_4_sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15':'TEST', #NOT TURNING UP!
#                             'gauss_4_sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'Tr-4(30E)-->0(30E), Te-4',
#                             # 'gauss_2_sup_RN50_gauss_2_for_60_epoch':'Tr-2(60E), Te-2',
#                             'gauss_4_sup_RN50_gauss_4_for_60_epoch_lr_15':'Tr-4(60E), Te-4 (lr)', #Why is this not appearing /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/plot_shape_bias_feb23/blurryNets_tested-equivalentlyBlurryShapeBiasImages_threshCorrect-25/nets-blurryNets_tested-equivalentlyBlurryShapeBiasImages_epochs-All_threshCorrect-25.png!!!!!
#                             'gauss_4_sup_RN50_gauss_4_for_60':'Tr-4(60E), Te-4',
#                             'gauss_6_sup_RN50_gauss_6_for_60':'Tr-6(60E), Te-6',
                            
#                           }
# subset_of_nets = ['Tr-0(60E), Te-0','Tr-1(60E), Te-1','Tr-0(60E), Te-4','Tr-2(60E), Te-2','Tr-4(60E), Te-4 (lr)','Tr-4(60E), Te-4',  'Tr-4(30E)-->0(30E), Te-4', 'Tr-4(30E)-->0(30E), Te-4 (lr)','Tr-6(60E), Te-6']



# # # 5. Size of conv Kernels in Layer 1
# # Note that all netws are conv 7 by default
# # Note: have not trained a ;15-blurryPY' net...
net_subset = 'kernelSize-conv1'

rename_chosen_models_exp={

                            'sup_RN50_conv1_15_gauss_0_for_60_epoch':'21-HighRes',
                            'sup_RN50_gauss_0_for_60':'7-HighRes',
                            'sup_RN50_conv1_21_gauss_0_for_60_epoch':'21-highRes',
                            'sup_RN50_conv1_21_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'21-Blur-HighRes',
                            'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'7-Blur-HighRes',
                            'sup_RN50_conv1_21_from_gauss_0_for_30_epoch_to_gauss_4_for_30':'21-reverseBlurry',
                          }
full_subset_of_nets = ['7-highRes','15-highRes','21-highRes', '7-blurryPY', '21-blurryPY', '21-reverseBlurry']
# To include reverse blur trajectory.. (21) TO DO: add reverse (7)
# subset_of_nets = ['7-highRes','15-highRes','21-highRes', '7-blurryPY', '21-blurryPY']
subset_of_nets_highRes = ['7-highRes','15-highRes','21-highRes']
subset_of_nets_7_21 = ['7-HighRes','21-HighRes', '7-Blur-HighRes', '21-Blur-HighRes']

subset_of_nets = subset_of_nets_7_21

# # 6. Extended Training to 90 epochs
# net_subset = 'extendedTraining-epoch-90'
# rename_chosen_models_exp={
#                             # 'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'PTBlurry',
#                             # 'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_60':'PTBlurry',
#                             'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'Blur-HighRes',
#                             'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_60':'Blur-HighRes',
#                             'sup_RN50_gauss_0_for_60':'HighRes',
#                             'sup_RN50_gauss_0_for_90_epoch':'HighRes',
#                           }
# # subset_of_nets = ['PTBlurry','HighRes']
# subset_of_nets = ['Blur-HighRes', 'HighRes']

# Additional option- VOGELSANG CONDITION extended 
# 'sup_RN50_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch_to_gauss_0_for_30'



# # 8. effects of Learning Rate
# net_subset = 'learningRate-modifyFreqofChange'
# rename_chosen_models_exp={
#                             'sup_RN50_gauss_0_for_60_epoch_lr_15':'HighRes_LR-15E',
#                             'sup_RN50_gauss_0_for_60':'HighRes_LR-30E',
#                             'sup_RN50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15':'PTBlurry_LR-15E',
#                             'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'PTBlurry_LR-30E',
#                           }
# subset_of_nets = ['HighRes_LR-15E','HighRes_LR-30E', 'PTBlurry_LR-15E', 'PTBlurry_LR-30E']



# # 9. Core Nets
# # 3. Amount of blur, fixed duration (4)
# net_subset = 'coreNets'
# # HIGH RES vs LOW REST TEST SET - CAUTIONNEED ACCURACY TOO!!!!!!!!!!!!!!!!!!!!
# # THIS IS THE ONE TO OPTIMISE ANIMACY EXPERIMENT FOR! 
# rename_chosen_models_exp={
#                             'sup_RN50_gauss_0_for_60':'HighRes',
#                             'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'Blur-HighRes',
#                             # WORK ON THIS SECTION
#                             # TEST ALL WITH HIGH AND LOW RES SH IMAGES BUT NEED TO COMPARE ACCURACY!!!!!
#                             # 'sup_RN50_from_gauss_0_for_30_epoch_to_gauss_4_for_30':'HighRes-Blur',
#                             # 'sup_RN50_from_gauss_6_for_30_epoch_to_gauss_0_for_30':"6",
#                             # 'gauss_4_sup_RN50_gauss_4_for_60_epoch_lr_15':'Tr-4(60E), Te-4 (lr)', #Why is this not appearing /home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/plot_shape_bias_feb23/blurryNets_tested-equivalentlyBlurryShapeBiasImages_threshCorrect-25/nets-blurryNets_tested-equivalentlyBlurryShapeBiasImages_epochs-All_threshCorrect-25.png!!!!!
#                           }
# subset_of_nets = ['Blur-HighRes', 'HighRes']


# 10+ other Experiments Networksexperimente!
# REPEAT TRAINING
#  'sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2'
# Has a repeat baseline been run?

# GRADED TRAJECTORIES
#  'sup_RN50_from_gauss_6_for_15_epoch_to_gauss_4_for_15_epoch_to_gauss_2_for_15_epoch_to_gauss_0_for_15'
# 'sup_RN50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20'

# miscellaneous networks
#  'sup_RN50_from_gauss_0_for_60_epoch' - error chcek this agains other 45 - re-run to verify!





# THeoretical notes - 
# Boosting the learning rate really seemed to make  difference!!!!!!!!!!!!!!!
# TO DO:
# Pretraining blurry gave BIG boost for shapr bias on blurry stimuli - THINK ABOUT THIS!

#  'gauss_4_sup_RN50_from_gauss_4_for_30_epoch_to_gauss_0_for_30'



# subselects relavant rows from df based on the code above

totals_df.replace(regex=rename_chosen_models_exp, inplace=True)
cats_df.replace(regex=rename_chosen_models_exp, inplace=True)


cats_df = cats_df.loc[cats_df['Network'].isin(subset_of_nets)]
totals_df = totals_df.loc[totals_df['Network'].isin(subset_of_nets)]


# END OF SELECT SUBSET OF NEWORKS #############




out_dir = f'{net_subset}_threshCorrect-{prop_correct_threshdold}'
save_out_subset =  os.path.join(save_dir,out_dir)
if not os.path.exists(save_out_subset):
    os.makedirs(save_out_subset)
    
save_out_subset_animacy = os.path.join(save_out_subset,'animacyAnalysis')
if not os.path.exists(save_out_subset_animacy):
    os.makedirs(save_out_subset_animacy) 

cats_df.sort_values(by=['Epoch'], inplace=True)
totals_df.sort_values(by=['Epoch'], inplace=True)


# ALL EPOCHS
# FIGURE - Plot shape bias of all netwroks over time (at each epoch)
fig, ax = plt.subplots()

fig.set_figheight(4)
fig.set_figwidth(6)



# Pair colours to make it easier to visualise pretrained blurry and high res nets 
# Starts out of phase by 1!
if net_subset == 'kernelSize-conv1':
    cmap = sns.color_palette("Paired", n_colors=len(subset_of_nets))
    for i, (key, grp) in enumerate(totals_df.groupby(['Network'])):
        ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias', label=key, color=cmap[i-1] )

# Paired colours for ease of comparison
# In phase!
elif ((net_subset == 'learningRate-modifyFreqofChange') or (net_subset == 'coreNets') or (net_subset == 'extendedTraining-epoch-90')):
    cmap = sns.color_palette("Paired", n_colors=len(subset_of_nets))
    for i, (key, grp) in enumerate(totals_df.groupby(['Network'])):
        ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias', label=key, color=cmap[i] )

else:  
    for key, grp in totals_df.groupby(['Network']):
        
        ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias', label=key)

    
    
# box = ax.get_position()

# ax.set_position([box.x0, box.y0 + box.height * 0.1,
#                  box.width, box.height * 0.9])





plt.ylabel("Shape Bias (%)")
if net_subset =='blur_modify_amt_first_30_epochs':
    
    plt.title("Effect of pretraining (Epochs 0-30) with varying amounts of blur\non Shape Bias over time")
    # ax.legend(title="Blur ($\sigma$)", loc='center left', bbox_to_anchor=(1, 0.50))
    # ax.legend(title="Blur ($\sigma$)", loc="lower left", ncol=4,  )
    plt.ylim(15, 35)  
    plt.xlim(30, 60)  
    ax.legend(title="Blur ($\sigma$)",loc='upper center', bbox_to_anchor=(0.5, -0.9),
          fancybox=True, shadow=True, ncol=4)

    # plt.legend(loc="lower left", ncol=len(df.columns))
    
if net_subset =='blur_4_to_0_modify_length':
    plt.title("Effect of varying duration of blurry pretraining ($\sigma$=4)\nShape Bias over time")
    ax.legend(title="Duration PT:T\n(Epochs)", loc='center left', bbox_to_anchor=(1, 0.50))
    plt.ylim(15, 35)  
    plt.xlim(5, 60)  
    
# INTERESTING GRAPH!
if net_subset == 'blurryNets_tested-equivalentlyBlurryShapeBiasImages':
    plt.title("Shape bias when tested on images with equivalent blur to training set \nShape Bias over time")
    ax.legend(title="Train-Train", loc='center left', bbox_to_anchor=(1, 0.50))
    plt.ylim(10, 70)  

if net_subset == 'kernelSize-conv1':
    plt.title(f"Effect of Modifying Kernel Size in Conv1 on\nShape Bias over time")
    plt.ylim(15, 35) 
    
if net_subset == 'extendedTraining-epoch-90':
    plt.title(f"Shape Bias over an extended time period")
    plt.ylim(10, 40)  
    plt.xlim(30, 95)  
    
if net_subset == 'learningRate-modifyFreqofChange':
    # TO TO : Put horizontal lines in a different axis!
    # Add LR 
    plt.title(f"Effect of modifying the frequency of reducing the Learning Rate\non Shape bias over Time")
    
    plt.axvline(x = 15, color = 'pink', label = 'LR-changeSome-15')
    plt.axvline(x = 30, color = 'r', label = 'LR-changeAll-30')
    plt.axvline(x = 45, color = 'pink', label = 'lLR-changeSome-45')
    plt.axvline(x = 60, color = 'r', label = 'LR-changeAll-60')
    # plt.axvline(x =60, color = 'white', label = '') 
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

if net_subset == 'coreNets':
     plt.title(f"Effect of pretraining on blurry images on Shape bias over time")
     plt.ylim(10, 40)  
        
plt.savefig(f'{save_out_subset}/nets-{net_subset}_epochs-All_threshCorrect-{prop_correct_threshdold}.png', bbox_inches="tight")
 
 
print(f'totals_df: {totals_df}')
# CHOSEN EPOCHS:
if net_subset == 'extendedTraining-epoch-90':
    firstEpoch=35
    midEpoch=60
    lastEpoch=90
    chosen_epochs_df =totals_df.loc[totals_df['Epoch'].isin([firstEpoch, midEpoch, lastEpoch])]
    


elif net_subset !='blur_4_to_0_modify_length':
    firstEpoch=35
    lastEpoch=60
    
    # CAUTION: some int, some str - should just convert epoch column to int...
    # IF PROBLEM - PROVEVLLY HERE.
    if (net_subset =='learningRate-modifyFreqofChange') or (net_subset =='kernelSize-conv1')  or (net_subset =='coreNets') or (net_subset == 'blur_modify_amt_first_30_epochs'):
        chosen_epochs_df =totals_df.loc[totals_df['Epoch'].isin([firstEpoch, lastEpoch])]
    else:
        chosen_epochs_df =totals_df.loc[totals_df['Epoch'].isin([f'{firstEpoch}', f'{lastEpoch}'])]
    print(f'chosen_epochs_df: {chosen_epochs_df}')

# FIGURE - Plot shape bias of all netwroks over time (at each epoch)
fig, ax = plt.subplots()

if net_subset !='extendedTraining-epoch-90':
    fig.set_figheight(4)
    fig.set_figwidth(2)
    plt.xticks([firstEpoch, lastEpoch])


# Set colour palette
if (net_subset =='learningRate-modifyFreqofChange') or (net_subset =='kernelSize-conv1'):
    cmap = sns.color_palette("Paired", n_colors=len(subset_of_nets))
# else:
#     cmap = sns.color_palette("Paired", n_colors=len(subset_of_nets)) #flare
    
if net_subset =='kernelSize-conv1':
    for i, (key, grp) in enumerate(chosen_epochs_df.groupby(['Network'])):
        if len(subset_of_nets) == 5:
            ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias', label=key, color=cmap[i-1] )
        else:
            ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias', label=key, color=cmap[i])
            print('here')
else: 
    for i, (key, grp) in enumerate(chosen_epochs_df.groupby(['Network'])):
            ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias' )


if net_subset == 'blurryNets_tested-equivalentlyBlurryShapeBiasImages':
    plt.title("Shape bias when tested on images with equivalent blur to training set \nShape Bias over time")
    ax.legend(title="Train-Train", loc='center left', bbox_to_anchor=(1, 0.50))

plt.ylim(10, 70)  

if net_subset =='blur_modify_amt_first_30_epochs':
    plt.title("Effect of pretraining (Epochs 0-30) with varying amounts of blur\non Shape Bias over time")
    ax.legend(title="Blur ($\sigma$)", loc='center left', bbox_to_anchor=(1, 0.50))
    plt.ylim(10, 40)  

if net_subset == 'learningRate-modifyFreqofChange':
    plt.title(f"Effect of modifying the frequency of\nreducing the Learning Rate\nEpoch:{firstEpoch}-{lastEpoch}")
    plt.ylim(10, 40)  
    ax.legend(title="Network", loc='center left', bbox_to_anchor=(1, 0.50))

if net_subset =='kernelSize-conv1':
    plt.ylim(10, 40)  
    ax.legend(title="Kernel Size-Network", loc='center left', bbox_to_anchor=(1, 0.50))
    print('again')

if net_subset =='coreNets':
    plt.ylim(10, 40)  
    ax.legend(title="Network", loc='center left', bbox_to_anchor=(1, 0.50))
    plt.title(f"Effect of pretraining on blurry images on Shape bias over time\nEpoch:{firstEpoch}-{lastEpoch}")
    #  if net_subset == 'coreNets':
    #  plt.title(f"Effect of pretraining on blurry images on Shape bias over time")

plt.ylabel("Shape Bias (%)")  

if net_subset =='extendedTraining-epoch-90':
    plt.ylim(10, 40)  
    plt.xticks([firstEpoch, midEpoch, lastEpoch])  
    plt.savefig(f'{save_out_subset}/nets-{net_subset}_compareEpochs-{firstEpoch}-{midEpoch}-{lastEpoch}_threshCorrect-{prop_correct_threshdold}.png', bbox_inches="tight")

# elif net_subset !='extendedTraining-epoch-90':
else:
    plt.savefig(f'{save_out_subset}/nets-{net_subset}_compareEpochs-{firstEpoch}-{lastEpoch}_threshCorrect-{prop_correct_threshdold}.png', bbox_inches="tight")




# Konkle Categories
animate_labels =['Animate', 'Inanimate Small', 'Inanimate Large', 'Inanimate'] 
animate_code_labels =['animate', 'inanimate_small', 'inanimate_large', 'inanimate'] 


for i, animate_category in enumerate([animal, inanimate_small, inanimate_large, inanimate]):
    
    animate_category_label = animate_labels[i]
    animate_category_code_label = animate_code_labels[i]
    print(f'Animate category is: {animate_category_label}')
    df_animate =cats_df.loc[cats_df['Category_short'].isin(animate_category)]


    

    grouped_animate = df_animate.groupby(['Network', 'Epoch']).sum().reset_index()
    
    grouped_animate = recalc_shape_bias(grouped_animate)
    grouped_animate = grouped_animate[grouped_animate['Prop_Correct']> prop_correct_threshdold]


    # FIGURE 
    fig, ax = plt.subplots()

    cmap = sns.color_palette("Paired", n_colors=len(subset_of_nets))
    for i, (key, grp) in enumerate(grouped_animate.groupby(['Network'])):
        ax = grp.plot(ax=ax, x='Epoch',kind='line', marker='o', y='Shape_Bias', label=key, color=cmap[i])
        

    ax.legend(title="Network",
            fancybox=True,  ncol=1) 
    
    plt.ylabel("Shape Bias (%)")
    plt.xlabel("Epoch")
    
    # CAUTION - UPDATE THIS TITL EPER NET!
    plt.title(f'Effect of blurry pretraining on\nShape bias of {animate_category_label} categories over time\n')
    
    
    # Lr lines
    # plt.axvline(x = 30, color = 'r', label = 'lr changes all')
    # plt.axvline(x = 60, color = 'r', label = 'lr changes all')
    # plt.axvline(x = 15, color = 'pink', label = 'lr changes some')
    # plt.axvline(x = 45, color = 'pink', label = 'lr changes some')
    
    plt.ylim(0, 60)  
    plt.savefig(f'{save_out_subset_animacy}/nets-{net_subset}_shapeBias-{animate_category_code_label}_byNetwork_overTime_threshCorrect-{prop_correct_threshdold}.png', bbox_inches="tight")


# Plot Bar chart at chosen Epoch 60
# chosen_epoch=60
if ((net_subset == 'blur_modify_amt_first_30_epochs') or (net_subset == 'coreNets')):
    epochs_to_plot = [35, 60]
elif net_subset == 'extendedTraining-epoch-90':
    epochs_to_plot = [65, 90] #NB SHOULD BE MORE BUT NEED TO FIX FIRST!
    
else:
    epochs_to_plot = [60]
    

    
for j, chosen_epoch in enumerate(epochs_to_plot):
    # Recall that totals_df was already thresholded above!
    totals_df_chosen_epoch = totals_df[totals_df['Epoch']==chosen_epoch]
    # totals_df_chosen_epoch0_to_plot = totals_df_chosen_epoch.loc[:, totals_df_chosen_epoch.columns.isin(['Network', 'Shape_Bias'])]

    totals_df_chosen_epoch.sort_values(by=['Network'], inplace=True)
    print('totals_df_chosen_epoch')
    print(totals_df_chosen_epoch)
    print(len(totals_df_chosen_epoch))

    fig, ax = plt.subplots()
    

    if net_subset == 'learningRate-modifyFreqofChange':
        # Paired colour Palette
        sns.barplot(data=totals_df_chosen_epoch, y="Shape_Bias", x="Network", palette=sns.color_palette("Paired"))
    
    # TO DO: figure how to get the paired colour palatte out of phase for 'kernelSize-conv1':
    # Paired colour Palette
    # net_subset == 'kernelSize-conv1':
    else: 
        # sns.barplot(data=totals_df_chosen_epoch, y="Shape_Bias", x="Network",  palette=sns.color_palette("Paired")) #flare
        if len(subset_of_nets) ==3:
            ax =sns.barplot(data=totals_df_chosen_epoch, y="Shape_Bias", x="Network",  palette=sns.color_palette("viridis_r"), order=subset_of_nets_highRes)
            plt.xlabel("Size of Kernel in First Convolutional Layer")
            ax.set_xticklabels(['7', '15', '21'])
        else:
            sns.barplot(data=totals_df_chosen_epoch, y="Shape_Bias", x="Network",  palette=sns.color_palette("rocket_r"))
            

        
    if net_subset =='blur_modify_amt_first_30_epochs':
        plt.title(f"Effect of pretraining (Epochs 0-30) with varying amounts of blur\nShape Bias at Epoch {chosen_epoch}")
        plt.xlabel("Pretrained Blur ($\sigma$)")
        plt.ylim(15, 35)  
        
    if net_subset =='blur_4_to_0_modify_length':
        plt.title(f"Effect of varying duration of blurry pretraining ($\sigma$=4)\nShape Bias at Epoch {chosen_epoch}")
        plt.xlabel("Duration of Blurry Pretraining:High Resolution Training in Epochs")
        plt.ylim(20, 35)   #20?
        
    if net_subset == 'blurryNets_tested-equivalentlyBlurryShapeBiasImages':
        plt.title(f"Shape bias at Epoch {chosen_epoch}\nTest set equivalent blur to training set")
        plt.ylim(20, 70) 
        plt.xticks( rotation=90)  #fontsize = 6,
            
    if net_subset == 'kernelSize-conv1':
        plt.title(f"Effect of Modifying Kernel Size on\nShape bias at Epoch {chosen_epoch}")
        plt.ylim(15, 35) 
    
    if net_subset == 'extendedTraining-epoch-90':
        plt.title(f"Shape bias at Epoch {chosen_epoch}")
    
    if net_subset == 'learningRate-modifyFreqofChange':
        plt.ylim(15, 35) 
        plt.title(f"Effect of modifying the frequency of reducing the Learning Rate\non Shape biasEpoch {chosen_epoch}")

    plt.ylabel("Shape Bias (%)")
    
    
    
    # if blur_4_to_0_modify_length
    plt.savefig(f'{save_out_subset}/nets-{net_subset}_shapeBias_epoch-{chosen_epoch}_threshCorrect-{prop_correct_threshdold}_plot-bar.png', bbox_inches="tight")



    # ANIMACY ANALYSIS - RUN ON coreNets ONLY!
    if net_subset =='coreNets':
        # Group by Konkle categories 
        print('cats_df')
        print(cats_df['Epoch'])
        

        cats_df_chosen_epoch = cats_df[cats_df['Epoch']==chosen_epoch]
        
        # CAUTION - this isn't choosing a GIVEN EPOCH!!!!!!!!
        
        # Plot each category seperately 
        # Order Categories grouping by animacy
        order_of_cat=['bear', 'bird', 'cat','dog', 'elephant','airplane', 'bicycle', 'boat', 'car', 'chair',  'keyboard', 'oven', 'truck','clock', 'bottle', 'knife']


        # Change aspect ratio!
        ax = sns.catplot(data=cats_df_chosen_epoch, kind="bar",y="Shape_Bias", x="Category", hue="Network", hue_order=subset_of_nets, aspect = 2.2, order=order_of_cat, errorbar=None, palette=sns.color_palette("Paired"))
        plt.ylabel("Shape Bias (%)")
        # plt.xlabel("Category")
        plt.title(f'Shape bias per category by network: Epoch {chosen_epoch}')
        fig.set_figwidth(20)
        plt.savefig(f'{save_out_subset_animacy}/nets-{net_subset}_shapeBias-byNetbyCategory-_epoch-{chosen_epoch}_threshCorrect-{prop_correct_threshdold}_plot-bar.png', bbox_inches="tight")


        
        # net_subset = 'coreNets'
        
        grouped_by_animate_categories= cats_df.groupby(['Network', 'Epoch', 'Animate']).sum().reset_index()
        print('grouped_by_animate_categories')
        print(grouped_by_animate_categories)
        # reset columns
        grouped_by_animate_categories = recalc_shape_bias(grouped_by_animate_categories)
        print(grouped_by_animate_categories)
        grouped_by_animate_categories = grouped_by_animate_categories[grouped_by_animate_categories['Prop_Correct']> prop_correct_threshdold]
        grouped_by_animate_categories = grouped_by_animate_categories[grouped_by_animate_categories['Epoch']==chosen_epoch]
        print(grouped_by_animate_categories)


        animate_order_labels =['Animate', 'Inanimate Large','Inanimate Small'] 
        animate_order =['animate', 'inanimate_large','inanimate_small'] 

        ax = sns.catplot( data=grouped_by_animate_categories, kind="bar",y="Shape_Bias", x="Network", hue="Animate",  palette=sns.color_palette("flare"))
        plt.title(f'Shape bias per category of animacy by network: Epoch {chosen_epoch}')
        plt.ylabel("Shape Bias (%)")
        plt.xlabel("Network")
        plt.savefig(f'{save_out_subset_animacy}/nets-{net_subset}_shapeBias-byAnimacybyNet_epoch-{chosen_epoch}_threshCorrect-{prop_correct_threshdold}_plot-bar.png', bbox_inches="tight")

        ax = sns.catplot(data=grouped_by_animate_categories, kind="bar",y="Shape_Bias", x="Animate", hue="Network", hue_order=subset_of_nets, order=animate_order, palette=sns.color_palette("Paired"))
        plt.ylabel("Shape Bias (%)")
        plt.xlabel("Animacy")
        



        ax.set_xticklabels(animate_order_labels)
        plt.title(f'Shape bias per network by category of animacy: Epoch {chosen_epoch}')
        plt.savefig(f'{save_out_subset_animacy}/nets-{net_subset}_shapeBias-byNetbyAnimacy_epoch-{chosen_epoch}_threshCorrect-{prop_correct_threshdold}_plot-bar.png', bbox_inches="tight")



        # Potential Graph?:
        # stacked - shape choices, texture choices vs other choices



