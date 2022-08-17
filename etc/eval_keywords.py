# Modified by Sehyun Kim, 2022-07-29(July 29th, 2022), @RebuilderAI, Seoul, South Korea

import torch
import os
import cv2
from profanity_filter import ProfanityFilter
from ImageCaptioning_def import get_img_feats, get_text_feats, get_nn_text_customized
import pickle
import clip
import numpy as np

# clip_feat_dim depends on clip_version. In this case, clip_feat_dim is set to 512
clip_version = "ViT-B/16"
# Available CLIP model versions: ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"] {type:"string"}
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Must set jit = False for training
model, preprocess = clip.load(clip_version, device=device, jit=False)
model.cuda().eval()

places365_txt_path = '/home/ubuntu/RebuilderAI/Rendering-Recommendation-API/categories_places365.txt'
obj_texts_path = '/home/ubuntu/RebuilderAI/Rendering-Recommendation-API/object_texts_512.pkl'
obj_feats_path = '/home/ubuntu/RebuilderAI/Rendering-Recommendation-API/object_feats_512.pkl'
semantic_hier_txt_path = '/home/ubuntu/RebuilderAI/Rendering-Recommendation-API/categories_semantic_hierarchy.txt'


# Load scene categories from Places365 and compute their CLIP features.
place_categories = np.loadtxt(places365_txt_path, dtype=str)
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

if not os.path.exists(obj_texts_path):
    # Load object categories from Tencent ML Images
    
    with open(semantic_hier_txt_path) as fid:
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

    with open(obj_texts_path, 'wb') as txt_fd:
        pickle.dump(object_texts, txt_fd)
    with open(obj_feats_path, 'wb') as feat_fd:
        pickle.dump(object_feats, feat_fd)

else:
    with open(obj_texts_path, "rb") as txt_fd:
        object_texts = pickle.load(txt_fd)
    with open(obj_feats_path, "rb") as feat_fd:
        object_feats = pickle.load(feat_fd)

obj_topk = 50
num_captions = 5

def get_kwords_scores_customized_CLIP(img_path, img_moods, img_places, img_colors):
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # cv2.COLOR_BGR2RGB
    img_feats = get_img_feats(model, preprocess, img)

    img_colors_feats = get_text_feats(model, [f'Background color: {t}' for t in img_colors])
    sorted_img_colors, img_color_scores = get_nn_text_customized(img_colors, img_colors_feats, img_feats)

    img_moods_feats = get_text_feats(model, [f'Photo mood: {t}' for t in img_moods])
    sorted_img_moods, img_mood_scores = get_nn_text_customized(img_moods, img_moods_feats, img_feats)

    sorted_obj_texts, obj_scores = get_nn_text_customized(object_texts, object_feats, img_feats)
    object_list = ''
    
    for i in range(obj_topk):
        object_list += f'{sorted_obj_texts[i]}, '
    object_list = object_list[:-2]
    
    # img_styles_feats = get_text_feats(model, [f'Photo style: {t}' for t in img_styles])
    # sorted_img_styles, img_style_scores = get_nn_text_customized(img_styles, img_styles_feats, img_feats)
    img_places_feats = get_text_feats(model, [f'Photo is taken at: {t}' for t in img_places])
    sorted_img_places, img_places_scores = get_nn_text_customized(img_places, img_places_feats, img_feats)
    
    return sorted_img_places[:5], sorted_img_colors[:5], \
        img_places_scores, img_color_scores, img_mood_scores
