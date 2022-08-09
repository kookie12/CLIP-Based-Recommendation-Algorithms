# Modified by Sehyun Kim, GwanHyeong Koo, 2022-07-29(July 29th, 2022), @RebuilderAI, Seoul, South Korea
# 이 파일은 이미지 데이터에 대한 정보를 csv 파일로 저장하는 파일입니다.

import os
import clip
import torch
# from eval_keywords import get_kwords_scores_CLIP
from ImageCaptioning_def import get_img_feats, get_text_feats, get_nn_text, prompt_llm
from profanity_filter import ProfanityFilter
import pandas as pd
import numpy as np
import pickle
import cv2

clip_version = "ViT-B/16"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_version, device=device, jit=False)

# test_bg_img_dir = "../seankim/images_instagram/data"

# Load scene categories from Places365 and compute their CLIP features.
# place_categories = np.loadtxt('./data_params_feats/categories_places365.txt', dtype=str)
place_categories = np.loadtxt('./categories_places365.txt', dtype=str)
place_texts = []
for place in place_categories[:, 0]:
    place = place.split('/')[2:]
    if len(place) > 1:
        place = place[1] + ' ' + place[0]
    else:
        place = place[0]
    place = place.replace('_', ' ')
    place_texts.append(place)
place_feats = get_text_feats(model, [f'Photo of a {p}.' for p in place_texts])

# obj_text_fName = "./data_params_feats/object_texts_" + str(clip_feat_dim) + ".pkl"
# obj_feat_fName = "./data_params_feats/object_feats_" + str(clip_feat_dim) + ".pkl"
obj_text_fName = "./object_texts_" + str(clip_feat_dim) + ".pkl"
obj_feat_fName = "./object_feats_" + str(clip_feat_dim) + ".pkl"

if not os.path.exists(obj_text_fName):
    # Load object categories from Tencent ML Images
    # with open('./data_params_feats/dictionary_and_semantic_hierarchy.txt') as fid:
    with open('./dictionary_and_semantic_hierarchy.txt') as fid:
        object_categories = fid.readlines()
    object_texts = []
    pf = ProfanityFilter()
    # len(object_categories) = 11166
    for object_text in object_categories[1:]:
        object_text = object_text.strip()
        object_text = object_text.split('\t')[3]
        safe_list = ''
        for variant in object_text.split(','):
            text = variant.strip()
            if pf.is_clean(text):
                safe_list += f'{text}, '
        safe_list = safe_list[:-2]
        if len(safe_list) > 0:
                object_texts.append(safe_list)
    
    # Remove redundant categories
    object_texts = [o for o in list(set(object_texts)) if o not in place_texts]
    object_feats = get_text_feats(model, [f'Photo of a {o}.' for o in object_texts])

    with open(obj_text_fName, 'wb') as txt_fd:
        pickle.dump(object_texts, txt_fd)
    with open(obj_feat_fName, 'wb') as feat_fd:
        pickle.dump(object_feats, feat_fd)

else:
    with open(obj_text_fName, "rb") as txt_fd:
        object_texts = pickle.load(txt_fd)
    with open(obj_feat_fName, "rb") as feat_fd:
        object_feats = pickle.load(feat_fd)
        
def get_kwords_CLIP(img_path):
    
    # Load image
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2.COLOR_BGR2RGB
    
    img_feats = get_img_feats(model, preprocess, img)
    
    # img_bright_feats = get_text_feats(model, [f'Photo brightness: {t}' for t in img_bright])
    # sorted_img_brights, img_bright_scores = get_nn_text(img_bright, img_bright_feats, img_feats)
    
    # img_temp_feats = get_text_feats(model, [f'Photo temperature: {t}' for t in img_temps])
    # sorted_img_temps, img_temp_scores = get_nn_text(img_temps, img_temp_feats, img_feats)
    
    # img_moods_feats = get_text_feats(model, [f'Photo mood: {t}' for t in img_moods])
    # sorted_img_moods, img_mood_scores = get_nn_text(img_moods, img_moods_feats, img_feats)
 
    # img_styles = ['modern', 'luxury', 'nature', 'street-hiphop', 'wood', 'neon', 'antiuqe', 'traditional',
    #           'industrial', 'cozy', 'science-fiction', 'simple', 'abstract', 'office', 'bathroom']
    
    img_styles = ['black modern', 'white modern', 'black luxury', 'white luxury', 'nature with green plant', 'street hiphop', 
              'dark brown wood', 'light brown wood', 'neon sign', 'antique', 'traditional', 'industrial', 'cozy', 
              'science-fiction', 'magical', 'pink romantic', 'artistic', 'office', 'bathroom']

    img_colors = ['White', 'Yellow', 'Blue', 'Red', 'Green', 'Black', 'Brown', 'Beige', 'Ivory', 'Silver', 'Purple', 'Navy', 'Gray', 'Orange', 'Pink', 'Khaki']
    obj_topk = 50
 
    ### Zero-shot VLM: classify image color
    img_colors_feats = get_text_feats(model, [f'Background color is {t}' for t in img_colors])
    sorted_img_colors, img_color_scores = get_nn_text(img_colors, img_colors_feats, img_feats)

    # Zero-shot VLM: classify places.
    place_topk = 3
    # The place where this photo was taken is 
    place_feats = get_text_feats(model, [f'Photo was taken at {p}' for p in place_texts])
    sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats)

    # Zero-shot VLM: classify objects.
    sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats)
    object_list = ''
    for i in range(obj_topk):
        object_list += f'{sorted_obj_texts[i]}, '
    object_list = object_list[:-2]
    
    img_styles_feats = get_text_feats(model, [f'Photo style is {t}' for t in img_styles])
    sorted_img_styles, img_style_scores = get_nn_text(img_styles, img_styles_feats, img_feats)
    
    return sorted_img_styles[:5], sorted_img_colors[:5], \
    sorted_places[:3], sorted_obj_texts[:3]


def make_imgdir_csv():

    res_dict = {'category': [], 'file name': [], 'style': [], 'color': [], 'style score vec': [], 'color score vec': []}
    # res_dict['place'] = []
    # res_dict['object'] = []

    instagram_img_dir = './background_asset_copy/KYJ'

    for (root, dirs, files) in os.walk(instagram_img_dir):
        print("root : ", root)
        print("dirs : ", dirs)
        print("files : ", files)
        if len(files) > 0: 
            for index, file_name in enumerate(files):
                # file_path = os.path.abspath(os.path.join(root, file_name))
                file_path = os.path.join(os.path.join(root, file_name))
                rel_file_path = os.path.join(root, file_name)
                styles, colors, style_scores, color_scores = get_kwords_CLIP(rel_file_path)
                # file_name = file_path.replace('./background_asset_copy/KYJ/', '')
                raw_category = file_path.replace('./background_asset_copy/KYJ/', '')
                category = raw_category.split('_')[1]
                print("### category : ", category)
                res_dict['category'].append(category)
                res_dict['file name'].append(raw_category) # file_name -> raw_category @kookie12
                res_dict['style'].append(', '.join(styles))
                res_dict['color'].append(', '.join(colors))
                res_dict['style score vec'].append(style_scores)
                res_dict['color score vec'].append(color_scores)
                # res_dict['place'].append(', '.join(places))
                # res_dict['object'].append(', '.join(objs))

    df = pd.DataFrame(res_dict)
    df.set_index("file name", inplace=True)
    df = df.replace('\n','', regex=True)

    df.to_csv('./results/bg_asset_keywords_KYJ_0805_new3.csv')
    
if __name__=='__main__':    
    make_imgdir_csv()