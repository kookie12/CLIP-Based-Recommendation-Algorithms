from importlib_metadata import Pair
import torch
import os
import cv2
import numpy as np
from profanity_filter import ProfanityFilter
from ImageCaptioning_def import get_img_feats, get_text_feats, get_nn_text, prompt_llm
import pickle
import time
import clip
from transformers import BertTokenizer, BertModel
import pandas as pd
from csv import writer
import csv

# clip_feat_dim depends on clip_version. In this case, clip_feat_dim is set to 512
clip_version = "ViT-B/16"
# Available CLIP model versions: ["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14"] {type:"string"}
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768, 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# Must set jit = False for training
model, preprocess = clip.load(clip_version, device=device, jit=False)

model.cuda().eval()
print("start")
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

# Zero-shot VLM: Classify image mood
img_moods = ['calm', 'monotonous',  'festive', 'gloomy', 'dreary', 'grotesque', 'cozy', 'hopeful', 
                'hopeless', 'promising', 'horrible', 'scary', 'frightening', 'humorous', 'mysterious', 
                'peaceful', 'romantic', 'solitary', 'urgent', 'tense', 'tragic', 'comic', 'desperate', 
                'dynamic', 'moving', 'touching', 'encouraging', 'heartening', 'depressing', 'discouraging', 
                'disheartening', 'fantastic', 'awesome', 'spectacular', 'stressful', 'lively', 'brisk', 'dull', 
                'boring', 'wearisome', 'tiresome', 'inspiring', 'relaxing', 'nostalgic', 'disgusting', 
                'delightful', 'joyful', 'pleasant', 'merry', 'idle', 'solemn', 'grave', 'annoying', 'irritating', 
                'threatening', 'gorgeous', 'prophetic', 'suspenseful', 'thrilling', 'pastoral', 'pitiful', 
                'magnificent', 'natural']

img_colors = ['White', 'Yellow', 'Blue', 'Red', 'Green', 'Black', 'Brown', 'Beige', 'Azure', 'Ivory', 'Teal', 'Silver', 'Purple', 'Navy blue', 'Pea green', 'Gray', 'Orange', 'Maroon', 'Charcoal', 'Aquamarine', 'Coral', 'Fuchsia', 'Wheat', 'Lime', 'Crimson', 'Khaki', 'Hot pink', 'Magenta', 'Olden', 'Plum', 'Olive', 'Cyan']
img_lights = ['well-lit', 'bright', 'natural light', 'evenly distributed', 'concentrated', 'dark', 'gloomy', 'cozy', 'romantic', 'grow', 'strong']
# obj_topk = 10
num_captions = 5
output_dict = {}
output_dict_en = {}

abs_path = os.path.dirname(os.path.abspath(__file__))
print("abs_path : ", abs_path)

def img2text_CLIP(img_path):
    # img_path = '.' + img_path
    # print("img_path = ", img_path)
    # Load image 
    image = cv2.imread(img_path)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # , cv2.COLOR_BGR2RGB
    img_feats = get_img_feats(model, preprocess, image)
    
    ### Zero-shot VLM: classify image mood
    img_moods_feats = get_text_feats(model, [f'Mood of the image is {t}.' for t in img_moods])
    sorted_img_moods, img_mood_scores = get_nn_text(img_moods, img_moods_feats, img_feats)
    img_mood = sorted_img_moods[0]

    ### Zero-shot VLM: classify image color
    img_colors_feats = get_text_feats(model, [f'Color of the image background is {t}.' for t in img_colors])
    sorted_img_colors, img_color_scores = get_nn_text(img_colors, img_colors_feats, img_feats)
    img_color = sorted_img_colors[0]
    
    ### Zero-shot VLM: classify image lightening
    img_lights_feats = get_text_feats(model, [f'Light atmosphere of the image is {t}.' for t in img_lights])
    sorted_img_lights, img_mood_scores = get_nn_text(img_lights, img_lights_feats, img_feats)
    img_light = sorted_img_lights[0]

    # Zero-shot VLM: classify places.
    # place_topk = 3
    place_feats = get_text_feats(model, [f'Photo of a {p}.' for p in place_texts ])
    sorted_places, places_scores = get_nn_text(place_texts, place_feats, img_feats)

    # Zero-shot VLM: classify objects.
    obj_topk = 2
    sorted_obj_texts, obj_scores = get_nn_text(object_texts, object_feats, img_feats)

    # for i in range(obj_topk):
    #     print(f'{i}st : {sorted_obj_texts[i]}')
    #     object_list += f'{sorted_obj_texts[0][i]}, '
    # object_list = object_list[:-2]
    object_list_string = sorted_obj_texts[0]
    object_list = object_list_string.split(', ')
    # print("######## objects = ", object_list)
    
    if len(object_list) >= 2:
        objects = f'{object_list[0]}, {object_list[1]}'
        
    else:
        objects = f'{object_list[0]}'
    
    # print("######## objects = ", objects)

    # Zero-shot LM: generate captions.
    # prompt = f'''
    #     I think there might be a {object_list}.
    #     Please recommend a background that goes well with selling this item. What kind of studio, lighting atmosphere, and props would fit?'''
    label = f'''The mood of this studio image is {img_mood} and studio lightening is {img_light},
    this image includes a {objects}, and the background color is {img_color}.'''
    print("label : ", label)
    return label
        
        
def get_label_list(file_name, file_path, img_name):
    csv_file_path = os.path.join(os.getcwd(), file_name)
    df = pd.read_csv(csv_file_path)
    file_paths = df[file_path].to_list()
    img_names = df[img_name].to_list()
    return file_paths, img_names

def embedding():        
    # bert pretrained model
    print("here")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    labelFile = 'image_labeling_0722.csv'
    file_paths, img_names = get_label_list(labelFile, 'file_path', 'file_name')
    # print("file_paths : ", file_paths)
    # print("file_name : ", img_names)
    object_dict = {}
    for file_path, img_name in zip(file_paths, img_names):
        # translate label to embedding feats using Bert
        
        img_path = os.path.join(file_path, img_name)
        # img_path = "background_asset/KYJ/1_modern_18/modernA1.jpg"
        # img_path = "static/20220722020815/images/00000.png"
        # img_path = "modernA1.jpg"
        # print("img_path 1 : ", img_path)
        # img_path = os.path.join(abs_path, img_path)
        print("img_path : ", img_path)
        label = img2text_CLIP(img_path)
        encoded_input = tokenizer(label, return_tensors='pt')    
        output = model(**encoded_input).pooler_output
        
        output_dict[f'{img_path}'] = output
        output_dict_en[f'{img_path}'] = label
        
    # with open('answer.csv', 'w') as f_object:
        # f_object = open('answer.csv', 'w', newline='')
        # writer_object = writer(f_object)
        pairs = output_dict_en.items()
        keys = output_dict_en.keys()
        values = output_dict_en.values()
        object_dict = {'file_path':keys, 'caption':values}
        df = pd.DataFrame(object_dict)
        df.to_csv('answer.csv')
        # object_dict[key] = value
        # for pair in pairs:
        #     pair_first = pair[0]
        #     pair_second = pair[1]
        #     writer_object.writerow([pair_first, pair_second])
            
        # writer_object.writerow(output_dict_en)
        # writer_object = csv.DictWriter(f_object)
        # writer_object.writerows(output_dict_en.items())
        # f_object.close()
            
    with open('./bg_text_feat_3d.pkl', 'wb') as fid1:
        pickle.dump(output_dict, fid1)

# if '__name__'=='main':
embedding()
    


