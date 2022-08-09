# Modified by Sehyun Kim, Gwan Hyeong Koo, 2022-08-06(Aug 6th, 2022), @RebuilderAI, Seoul, South Korea

import torch
import os
import cv2
import numpy as np
from profanity_filter import ProfanityFilter
from ImageCaptioning_def import get_img_feats, get_nn_text_customized, get_text_feats, prompt_llm
from torch import nn
import pandas as pd
import pickle
import clip
print("here in eval3.py")

# clip_feat_dim depends on clip_version. In this case, clip_feat_dim is set to 512
clip_version = "ViT-B/16"
# Available CLIP model versions: ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"] {type:"string"}
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Must set jit = False for training
model, preprocess = clip.load(clip_version, device=device, jit=False)

model.cuda().eval()

# Load scene categories from Places365 and compute their CLIP features.
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

obj_text_fName = "./object_texts_" + str(clip_feat_dim) + ".pkl"
obj_feat_fName = "./object_feats_" + str(clip_feat_dim) + ".pkl"

if not os.path.exists(obj_text_fName):
    # Load object categories from Tencent ML Images.
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

img_moods_original = ['calm', 'monotonous', 'gloomy', 'cozy', 'hopeful', 
                'promising', 'horrible', 'scary', 'mysterious', 
                'peaceful', 'romantic', 'solitary', 'touching', 'depressing', 
                'fantastic', 'lively']

img_moods = ['calm', 'peaceful', 'cozy', 'relaxing',
             'active', 'lively', 'dynamic',
             'green', 'eco-friendly', 'nature',
             'cheerful', 'joyful', 'fun',
             'gloomy', 'dark', 'somber',
             'eccentric', 'mysterious', 'fantastic',
             'romantic', 'lovely', 'beautiful']

img_styles = ['black modern', 'white modern', 'black luxury', 'white luxury', 'nature with green plant', 'street hiphop', 
              'dark brown wood', 'light brown wood', 'neon sign', 'antique', 'traditional', 'industrial', 'cozy', 
              'science-fiction', 'magical', 'pink romantic', 'artistic', 'office', 'bathroom']

img_colors = ['White', 'Yellow', 'Blue', 'Red', 'Green', 
            'Black', 'Brown', 'Beige', 'Ivory', 'Silver', 
            'Purple', 'Navy', 'Gray', 'Orange', 'Pink', 'Khaki']

obj_topk = 10
num_captions = 3

# def seans_img2text_CLIP(prod_img_path, bg_style_sc_path, bg_color_sc_path):
#     # Load image 
#     prod_image = cv2.imread(prod_img_path)
#     prod_image = cv2.cvtColor(prod_image, cv2.COLOR_BGR2RGB) # , cv2.COLOR_BGR2RGB
#     prod_img_feats = get_img_feats(model, preprocess, prod_image)
    
#     # define cos similarity
#     cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6) 
    
#     # Get image style, color, and mood
#     # bg_style_sc_path may be '../results/extended_style_score_bg_KYJ.csv'
#     bg_style_score_df = pd.read_csv(bg_style_sc_path)
#     color_score_df = pd.read_csv(bg_color_sc_path)
    
#     ### Zero-shot VLM: classify image style
#     img_styles_feats = get_text_feats(model, [f'Style of the image is {t}.' for t in img_styles])
#     sorted_img_styles, prod_img_style_scores = get_nn_text_customized(img_styles, img_styles_feats, prod_img_feats)

#     res_keyword = {'file name':[], 'style similarity' : [], 
#                     'color similarity' : []}
    
#     cp_prod_style_sc = prod_style_sc.copy()
#     cp_prod_color_sc = prod_color_sc.copy()
#     cp_bg_style_sc = bg_style_sc.copy()
#     cp_bg_color_sc = bg_color_sc.copy()

    
#     for i in range(len(df_background)):
#         bg_styles = df_background['style'][i].split(', ')
#         bg_colors = df_background['color'][i].split(', ')
        
#         curr_file_name = df_background['file name'][i]
        
#         style_sim_sc = cos_similarity(torch.tensor(cp_prod_style_sc), torch.tensor(cp_bg_style_sc[curr_file_name]))
#         color_sim_sc = cos_similarity(torch.tensor(cp_prod_color_sc), torch.tensor(cp_bg_color_sc[curr_file_name]))
        
#         res_keywords['count style'].append(cnt_style)
#         res_keywords['count color'].append(cnt_color)
#         res_keywords['style similarity'].append(style_sim_sc)
#         res_keywords['color similarity'].append(color_sim_sc)
    
#     res_keywords.update({'file name':df_background['file name']})
        
#     return res_keywords


def img2text_CLIP(prod_img_path):
    # Load image 
    prod_image = cv2.imread(prod_img_path)
    prod_image = cv2.cvtColor(prod_image, cv2.COLOR_BGR2RGB) # , cv2.COLOR_BGR2RGB
    prod_img_feats = get_img_feats(model, preprocess, prod_image)
    
    print("start imt2text_CLIP 1")
    
    # define cos similarity
    # dimenstion = 0이어야 scalar 가 나온다!
    cos_similarity = nn.CosineSimilarity(dim=0, eps=1e-6) 
    
    # Get image style, color, and mood
    bg_style_score_df = pd.read_csv('./results/extended_style_score_bg_KYJ.csv')
    color_score_df = pd.read_csv('./results/extended_color_score_bg_KYJ.csv')
    
    print("start imt2text_CLIP 2")
    
    ### Zero-shot VLM: classify image style
    img_styles_feats = get_text_feats(model, [f'Style of the image is {t}.' for t in img_styles])
    sorted_img_styles, prod_img_style_scores = get_nn_text_customized(img_styles, img_styles_feats, prod_img_feats)
    styles_list, styles_top5_images = [], []
    
    print("start imt2text_CLIP 3")
    
    # Get top 1 style using custom histogram equalization
    for index, col in enumerate(bg_style_score_df.columns):
        if col != 'styles':
            print("col : ", col)
            print("1 : ", torch.Tensor(prod_img_style_scores))
            print("1.5 : ", bg_style_score_df[col])
            print("2 : ", torch.Tensor(bg_style_score_df[col]))
            styles_list.append((cos_similarity(torch.tensor(prod_img_style_scores), torch.tensor(bg_style_score_df[col])), col))
    
    print("start imt2text_CLIP 4")
    styles_list.sort(reverse=True)
    
    # styles_top5_images = styles_list[:6][1] 
    for i in range(0, 6):
        styles_top5_images.append(styles_list[i][1]) # image name이 들어있다 @kookie12
        
    print("start imt2text_CLIP 4.5")
    styles_top5 = [] # 이건 위에서 선택된 5개의 이미지의 style을 뽑아내는 list @kookie12
    temp_list = []
    
    print("start imt2text_CLIP 5")
    # 선택된 5개 사진의 style이 어떤 것인지 가져옵니다 @kookie12
    for image in styles_top5_images:
        selected_df = bg_style_score_df[['styles', image]]
    
        for idx, row in selected_df.iterrows():
            # (image, colors) tuple
            temp_list.append((row[1], row[0]))
        
        temp_list.sort(key=lambda x: x[0], reverse=True)
        for style in temp_list[:5]:    
            styles_top5.append((image, style))
        temp_list = []
    
    print("styles_top5_images: ", styles_top5_images) # 나중에 지울 것.. @kookie12
    print("styles_top5: ", styles_top5) # 나중에 지울 것.. @kookie12
    
    # img_style = sorted_img_styles[0]

    ### Zero-shot VLM: classify image color
    # img_colors_feats = get_text_feats(model, [f'Color of the image background is {t}.' for t in img_colors])
    img_colors_feats = get_text_feats(model, [f'Color of the image is {t}.' for t in img_colors])
    sorted_img_colors, img_color_scores = get_nn_text_customized(img_colors, img_colors_feats, prod_img_feats)
    img_color = sorted_img_colors[0]

    # Zero-shot VLM: classify objects.
    # obj_topk = 10
    sorted_obj_texts, obj_scores = get_nn_text_customized(object_texts, object_feats, prod_img_feats)
    object_list = ''
    for i in range(obj_topk):
        object_list += f'{sorted_obj_texts[i]}, '
    object_list = object_list[:-2]

    # GPT-3 is All You Need - please save him..
    prompt_show = f'''
    This object is {object_list} and the object color is {img_color}. 
    Please recommend a background that goes well with selling this item. 
    What kind of studio, best studio light, and props would fit? 
    Explain why you think the background is the best for this item in detail.
    '''
    
    # Using GPT-3, generate image captions
    caption_texts_show = [prompt_llm(prompt_show, temperature=0.9) for _ in range(num_captions)]
    # caption_texts_style_color = [prompt_llm(prompt_style_color, temperature=0.9) for _ in range(num_captions)]

    # Zero-shot VLM: rank captions
    caption_feats = get_text_feats(model, caption_texts_show)
    sorted_captions_show, caption_scores = get_nn_text_customized(caption_texts_show, caption_feats, prod_img_feats)
    
    # It only returns a single caption(how many sentences should you generate depends on your taste)
    return sorted_captions_show, styles_top5_images, styles_top5 #, sorted_captions_style_color

if __name__ == '__main__':
    sorted_captions_show, styles_top5_images, styles_top5 = img2text_CLIP("./test_images/book.jpg")
    print("sorted_captions_show: ", sorted_captions_show)
    print("styles_top5_images: ", styles_top5_images)
    print("styes_top5: ", styles_top5)
