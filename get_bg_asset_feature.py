# Modified by Sehyun Kim, 2022-07-29(July 29th, 2022), @RebuilderAI, Seoul, South Korea

import os
import clip
import torch
from eval_keywords import get_kwords_scores_customized_CLIP
import pandas as pd

clip_version = "ViT-B/16"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_version, device=device, jit=False)

img_moods = ['calm', 'peaceful', 'cozy', 'relaxing',
             'active', 'lively', 'dynamic',
             'green', 'eco-friendly', 'nature',
             'cheerful', 'joyful', 'fun',
             'gloomy', 'dark', 'somber',
             'eccentric', 'mysterious', 'fantastic',
             'romantic', 'lovely', 'beautiful']

img_colors = ['White', 'Yellow', 'Blue', 'Red', 'Green', 'Black', 'Brown', 'Beige', 
              'Ivory', 'Silver', 'Purple', 'Navy', 'Gray', 'Orange', 'Pink', 'Khaki']

img_styles = ['black modern', 'white modern', 'black luxury', 'white luxury', 'nature with green plant', 'street hiphop', 
              'dark brown wood', 'light brown wood', 'neon sign', 'antique', 'traditional', 'industrial', 'cozy', 
              'science-fiction', 'magical', 'pink romantic', 'artistic', 'office', 'bathroom']

img_places = ['bed room', 'cafe', 'library', 'modern dining room', 'office', 'park', 'yard with table']


if __name__ == '__main__':
    bg_images_dir = 'Rendering-Recomm-API/acon'
    keyword_dict = {'file name': [], 'color': [], 'mood': [], 'place':[]}
    place_score_dict = {}
    color_score_dict = {}
    mood_score_dict = {}

    for (root, dirs, files) in os.walk(bg_images_dir):
        if len(files) > 0: 
            for file_name in files:
                abs_file_path = os.path.abspath(os.path.join(root, file_name))
                # relative file path for static folder
                rel_file_path = os.path.join(root, file_name)
                static_rel_fPath = rel_file_path.lstrip('Rendering-Recomm-API/')
                
                places, colors, place_scores, color_scores, mood_scores = \
                get_kwords_scores_customized_CLIP(rel_file_path, img_moods, img_places, img_colors)
                # static_rel_path = os.path.join('acon', file_name)
                keyword_dict['file name'].append(static_rel_fPath)
                keyword_dict['place'].append(', '.join(places))
                keyword_dict['color'].append(', '.join(colors))
                # modified for static directory in main.py
                place_score_dict[static_rel_fPath] = place_scores
                # color_score_dict[static_rel_fPath] = color_scores
                # mood_score_dict[static_rel_fPath] = mood_scores

    place_score_dict['place'] = img_places
    # color_score_dict['color'] = img_colors
    # mood_score_dict['mood'] = img_moods
    
    ##### Save keyword features of background assets, into csv file #####
    place_sc_df = pd.DataFrame(place_score_dict)
    # color_sc_df = pd.DataFrame(color_score_dict)
    # mood_sc_df = pd.DataFrame(mood_score_dict)
    
    place_sc_df.set_index('place', inplace=True)
    # color_sc_df.set_index('color', inplace=True)
    # mood_sc_df.set_index('mood', inplace=True)
    
    # mood_sc_df.to_csv('Rendering-Recomm-API/results/acon/mood_score.csv')
    # color_sc_df.to_csv('Rendering-Recomm-API/results/acon/color_score.csv')
    place_sc_df.to_csv('Rendering-Recomm-API/results/acon/place_score.csv')

    # color_sc_df = pd.DataFrame(color_score_dict)
    # color_sc_df.set_index('colors', inplace=True)
    # color_sc_df.to_csv('../results/scores/color_score_bg_KYJ.csv')

    # mood_sc_df = pd.DataFrame(mood_score_dict)
    # mood_sc_df.set_index('moods', inplace=True)
    # mood_sc_df.to_csv('../results/scores/mood_score_bg_KYJ.csv')