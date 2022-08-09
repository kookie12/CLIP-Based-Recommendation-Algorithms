from sklearn import preprocessing
import numpy as np
import pandas as pd
import os

def custom_transform_vec(vec):
    max_item = max(vec)
    curr_range = max_item - min(vec)
    new_vec = []
    for i in range(len(vec)):
        dist = round((max_item - vec[i]) * ((1/curr_range)**2), 4)
        new_vec.append(vec[i] - dist)
        
    min_new_vec = min(new_vec)
    new_vec_upscaled = [x-min_new_vec for x in new_vec]
    
    return new_vec_upscaled

def custom_transform_df(df):
    new_dict = {}
    for col in df.columns:
        if (col == 'styles' or col == 'colors' or col == 'moods'):
            new_dict[col] = df[col]
        else:
            curr_list = df[col]
            curr_max = max(curr_list)
            curr_range = curr_max - min(curr_list)
            new_list = []
            for i in range(len(curr_list)):
                dist = round((curr_max - curr_list[i]) * ((1/curr_range)**2), 4)
                new_list.append(curr_list[i] - dist)
            min_new_list = min(new_list)
            new_list_upscaled = [x-min_new_list for x in new_list]
                
            new_dict[col] = new_list_upscaled
    
    return pd.DataFrame(new_dict)

def extend_score_vectors():
    style_score_df = pd.read_csv('../results/scores/non-transformed/style_score_bg_KYJ.csv')
    color_score_df = pd.read_csv('../results/scores/non-transformed/color_score_bg_KYJ.csv')
    mood_score_df = pd.read_csv('../results/scores/non-transformed/mood_score_bg_KYJ.csv')
    df_style_sc_extended = custom_transform_df(style_score_df)
    df_color_sc_extended = custom_transform_df(color_score_df)
    df_mood_sc_extended = custom_transform_df(mood_score_df)

# df_style_sc_extended.to_csv('../results/scores/transformed/extended3_style_score_bg_KYJ.csv', index=False)
# df_color_sc_extended.to_csv('../results/scores/transformed/extended3_color_score_bg_KYJ.csv', index=False)
# df_mood_sc_extended.to_csv('../results/scores/transformed/extended3_mood_score_bg_KYJ.csv', index=False)
