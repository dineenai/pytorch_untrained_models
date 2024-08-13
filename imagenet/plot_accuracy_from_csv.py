import matplotlib.pyplot as plt
import pandas as pd
import os 


csv_src = "/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/model_accuracy_csv"
save_dir = '/home/ainedineen/blurry_vision/pytorch_untrained_models/imagenet/accuracy_graphs_sept22'
# iterate over all files in directory: model_accuracy_csv


all_models = []
# create one df with all models and group on basis of model_pth
for filename in os.listdir(csv_src):
    f = os.path.join(csv_src, filename)
    model_name = os.path.splitext(filename)[0]
    # Removes .csv suffix
    print(model_name)
    # https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    df = pd.read_csv(f, index_col=None, header=0)
    all_models.append(df)

frame = pd.concat(all_models, axis=0, ignore_index=True)
frame.drop(columns=['Unnamed: 0'], inplace=True)
# print(frame.columns)
print(frame.head)
# all paths are unique - dif no of epochs!
print(frame['model_pth'].nunique())


# Plot Top 1 and Top 5 accuracy
for accuracy in [1, 5]:

    top1_df = frame[["model_pth", "epoch", f"top{accuracy}acc"]]
    print(top1_df)
    print(top1_df.columns)

    loss_df = frame[["model_pth", "epoch", "loss"]]
    loss_df['model_name'] = loss_df['model_pth'].str.split('/').str[3]
    print(loss_df['loss'])
    # only 15 unique models as epoch nolonger relavant
    # print()
    print(loss_df['model_name'].nunique())

    # Select rows which do not have NaN value in column 'loss'
    loss_notNull_df = loss_df[~loss_df['loss'].isnull()]
    print(loss_notNull_df)

    #  7 models remaining (why do 8 have no loss?)
    print(loss_notNull_df['model_name'].nunique())



    top1_df.model_pth = top1_df.model_pth.astype(str)
    print(top1_df.dtypes)

    top1_df['model_name'] = top1_df['model_pth'].str.split('/').str[3]
    # pd.options.display.max_colwidth = 400
    # print(top1_df['model_pth'].str.split('/').str[3])
    print(top1_df.head)

    print(top1_df)
    top1_df_bymodel = top1_df[["model_name", "epoch", f"top{accuracy}acc"]]
    print(top1_df_bymodel.head)

    import math
    def round_up_to_nearest_10(num):
        return math.ceil(num / 10) * 10
    def round_down_to_nearest_10(num):
        return math.floor(num / 10) * 10

    max_top1 = top1_df_bymodel[f'top{accuracy}acc'].max()
    max_top1 = round_up_to_nearest_10(max_top1)

    min_top1 = top1_df_bymodel[f'top{accuracy}acc'].min()
    min_top1 = round_down_to_nearest_10(min_top1)
    # print('max')
    print(max_top1, min_top1)



    # works but too squished: fix - prints each graph seperatly!
    pd.pivot_table(top1_df_bymodel.reset_index(),
                index='epoch', columns='model_name', values=f'top{accuracy}acc'
                ).plot(subplots=True, figsize=(10,30), ylim=[min_top1, max_top1], grid=True, title="TITLE")
                #   set subplots to false for all on same graph!
    # plt.grid(True)
    plt.savefig(f'{save_dir}/allNets_seperate_top1.png', bbox_inches="tight")



    print('Model Name:')
    print(top1_df_bymodel['model_name'].unique())
    print(len(top1_df_bymodel['model_name'].unique()))
    # 15 models hereh



    # net_subset = 'all_nets'
    # net_subset = 'blur_4_to_0_modify_length'
    

    # Vraiants of gauss 4
    # Add identifier to 
    # missing 4 for 30 to compare train 2 to!
    # chosen_models = ['supervised_resnet50_from_gauss_6_for_30_epoch_to_gauss_0_for_30',
    #                  'supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2',
    #                  'supervised_resnet50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15']


    # allModels = ['supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_0_for_40',
    #                 'supervised_resnet50_gauss_4_for_30_epoch_to_gauss_0_for_30_lr_15',
    #                 # 'supervised_resnet50_gauss_0_for_90_epoch',
    #                 'supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30',
    #                 'supervised_resnet50_gauss_0_for_60_epoch',
    #                 'supervised_resnet50_gauss_1_for_60_epoch',
    #                 'supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30',
    #                 'supervised_resnet50_from_gauss_2_for_15_epoch_to_gauss_0_for_45',
    #                 # 'supervised_resnet50_from_gauss_0_for_30_epoch_to_gauss_4_for_30_epoch_to_gauss_0_for_30', #90 epochs
    #                 # 'supervised_resnet50_from_gauss_6_for_15_epoch_to_gauss_4_for_15_epoch_to_gauss_2_for_15_epoch_to_gauss_0_for_15',
    #                 'supervised_resnet50_from_gauss_6_for_30_epoch_to_gauss_0_for_30',
    #                 'supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30_TRAIN2',
    #                 'supervised_resnet50_from_gauss_4_for_15_epoch_to_gauss_0_for_45',
    #                 'supervised_resnet50_gauss_0_for_60_epoch_lr_15',
    #                 'supervised_resnet50_from_gauss_4_for_10_epoch_to_gauss_0_for_50',
    #                 'supervised_resnet50_from_gauss_4_for_20_epoch_to_gauss_2_for_20_epoch_to_gauss_0_for_20']



    # net_subset = 'blur_modify_amt_first_30_epochs'
    # net_subset = 'modify_lr'
    net_subset = 'sampled_1e'

    
    rename_chosen_models_exp ={       
                                'supervised_resnet50_gauss_0_for_60_epoch':'0',
                                'supervised_resnet50_from_gauss_2_for_30_epoch_to_gauss_0_for_30':'2',
                                'supervised_resnet50_from_gauss_4_for_30_epoch_to_gauss_0_for_30':'4',
                                'supervised_resnet50_from_gauss_6_for_30_epoch_to_gauss_0_for_30':"6",
                                }
    
    # chosen_models = ['0',
    #                  '2',
    #                  '4',
    #                  '6']
    
    # chosen_models = ['supervised_resnet50_gauss_0_for_5_epochs_changelr_1e']
    chosen_models = ['supervised_resnet50_gauss_0_1e']
    
    
    save_dir_exp = os.path.join(save_dir, f'{net_subset}')
    if not os.path.exists(save_dir_exp):
        os.makedirs(save_dir_exp)
    
    
    top1_df_bymodel.replace(regex=rename_chosen_models_exp, inplace=True)

    loss_notNull_df.replace(regex=rename_chosen_models_exp, inplace=True)


    # lr_models = 

    # df[df['A'].isin([3, 6])]
    top1_df_by_chosen_models = top1_df_bymodel[top1_df_bymodel['model_name'].isin(chosen_models)]

    print('HERE:')
    print(top1_df_by_chosen_models)
    fig, ax = plt.subplots()

    fig.set_figheight(4)
    fig.set_figwidth(6)
    # All Models:
    # for key, grp in top1_df_bymodel.groupby(['model_name']):
    # Subset ONLY:
    for key, grp in top1_df_by_chosen_models.groupby(['model_name']):
        ax = grp.plot(ax=ax, kind='line', x='epoch', y=f'top{accuracy}acc', label=key,  marker='o', )
   
    # Modifie 4 Oct 23
    
    if net_subset == 'blur_modify_amt_first_30_epochs':
        plt.xlim([30, 60])
        if accuracy ==5:
            plt.ylim([85, 95])
        if accuracy ==1:
            plt.ylim([65, 75])
        plt.title(f"Effect of pretraining (Epochs 0-30) with varying amounts of blur\non Top {accuracy} Accuracy")

    # elif net_subset == 'modify_lr':
        # plt.xlim([0, 5])
    else:
        
        if accuracy ==5:
            plt.ylim([0, 100])
        if accuracy ==1:
            plt.ylim([0, 100])


    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    
    plt.ylabel(f"Top {accuracy} Accuracy (%)")
    plt.xlabel("Epoch")
    # Put a legend below current axis
    # ax.legend(title="Blur ($\sigma$)", loc='upper center', bbox_to_anchor=(0.5, -0.05),
    #           fancybox=True, shadow=True, ncol=1)
    ax.legend(title="Blur ($\sigma$)", loc='center left', bbox_to_anchor=(1, 0.50))
    



    plt.savefig(f'{save_dir_exp}/{net_subset}_top-{accuracy}accuracy.png', bbox_inches="tight")


    # Epoch 35 and 60 only
    if net_subset == 'blur_modify_amt_first_30_epochs':
        firstEpoch=35
        lastEpoch=60
    elif net_subset == 'modify_lr':
        firstEpoch=1
        lastEpoch=5
    else:
        firstEpoch=1
        lastEpoch=15


    top1_df_by_chosen_models_chosen_epochs =top1_df_by_chosen_models.loc[top1_df_by_chosen_models['epoch'].isin([firstEpoch, lastEpoch])]
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(2)
    plt.xticks([firstEpoch, lastEpoch])
    for key, grp in top1_df_by_chosen_models_chosen_epochs.groupby(['model_name']):
        # ax = grp.plot(ax=ax, x='Epoch',kind='line',marker='o',  y='Shape_Bias', label=key, color=cmap[i] )
        ax = grp.plot(ax=ax, kind='line', x='epoch', y=f'top{accuracy}acc', label=key,  marker='o', )
    




        
    if net_subset == 'blur_modify_amt_first_30_epochs':
        if accuracy ==5:
            plt.ylim([85, 95])
            plt.yticks([85,90, 95])

        if accuracy ==1:
            plt.ylim([65, 75])
            plt.yticks([65,70, 75])

        plt.title(f"Effect of pretraining (Epochs 0-30) with varying amounts of blur\non Top {accuracy} Accuracy\nEpochs {firstEpoch}-{lastEpoch}")

    else:
        print('else using default values...')
    


    plt.ylabel(f"Top {accuracy} Accuracy (%)")  
    plt.xlabel("Epoch")  

    ax.legend(title="Blur ($\sigma$)", loc='center left', bbox_to_anchor=(1, 0.50))
    
    plt.savefig(f'{save_dir_exp}/{net_subset}_top-{accuracy}accuracy_epochs-{firstEpoch}-{lastEpoch}.png', bbox_inches="tight")

        
    # COMMENTED OUT FROM HERE
    # LOSS

    max_loss = loss_notNull_df['loss'].max()
    max_loss = round_up_to_nearest_10(max_loss)

    min_loss = loss_notNull_df['loss'].min()
    min_loss = round_down_to_nearest_10(min_loss)
    # print('max')
    print(min_loss, max_loss)
    
    print(f'Loss df:{loss_notNull_df}')

    loss_df_by_chosen_models = loss_notNull_df[loss_notNull_df['model_name'].isin(chosen_models)]

    fig, ax = plt.subplots()
    
    print(f'Models that currently have data on loss: {loss_notNull_df}')
    print(f'Chosen Models were: {chosen_models}')



    # Loss for chosen models:loss_df_by_chosen_models
    # loss for all models:  loss_notNull_df
    for key, grp in loss_df_by_chosen_models.groupby(['model_name']):
        ax = grp.plot(ax=ax, kind='line', x='epoch', y='loss', label=key, marker='o', )

    if net_subset == 'blur_modify_amt_first_30_epochs':
        ax.set_ylim([min_loss, 3])
    else:
        ax.set_ylim([min_loss, 10])



    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    plt.savefig(f'{save_dir_exp}/loss_{net_subset}.png', bbox_inches="tight")




