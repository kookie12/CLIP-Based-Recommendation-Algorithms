import os
import clip
import torch
from eval import img2text_CLIP
import pandas as pd
from torch import nn
import time

clip_version = "ViT-B/16" 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(clip_version, device=device, jit=False)
cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

product_img_path = '../test_images/cup.jpg'
abs_path = os.path.dirname(os.path.abspath(__file__))

# img2text_CLIP() 함수에 경로를 넣어줄 때 반드시 절대경로를 넣어줘야 인식된다.. 왜그럴까?
print("abs_path: ", abs_path)
print("real_path: ", os.path.realpath(__file__))
print("here path: ", os.getcwd())
# img_dir = os.path.join(abs_path, 'test_images/cup.jpg')
img_dir = './test_images/cup.jpg'

begin_time = time.time()
sorted_captions_show, sorted_captions_style_color = img2text_CLIP(img_dir)
print("sorted_captions_show : ", sorted_captions_show[0])
print("len: ", len(sorted_captions_show))
print(" ##### ")
print("sorted_captions_style_color : ", sorted_captions_style_color[0].lower())
print(" ############## ")
for item in sorted_captions_style_color:
    print("item : ", item.lower())

df_bg = pd.read_csv('./results/bg_asset_keywords_KYJ_new.csv')

img_styles = ['modern', 'luxury', 'nature', 'street', 'wood', 'neon', 'antiuqe', 'traditional',
              'industrial', 'cozy', 'science-fiction', 'simple', 'abstract', 'office', 'bathroom']

img_colors = ['white', 'yellow', 'blue', 'red', 'green', 'black', 'brown', 'beige', 'ivory', 'silver', 'purple', 'navy', 'gray', 'orange', 'pink', 'khaki']

recommend_styles_colors = sorted_captions_style_color[0].lower()

styles = []
colors = []

for style in img_styles:
    if style in recommend_styles_colors:
        print("style : ", style)
        styles.append(style)

for color in img_colors:
    if color in recommend_styles_colors:
        print("color : ", color)
        colors.append(color)

print("we want : ", styles, " and ", colors)

def naive_match(prod_styles, prod_colors, df_background):
    #res_keywords = {'file_name' : [], 'count color' : []}
    res_keywords = []
    res_keywords_temp = []
    # 추천된 3개의 style을 포함하는 폴더에만 접근합니다
    #for style in prod_styles:
    print("here in naive_match")
    # [7_antique_18, ]
    file_name_list = [] # only using for my thought..
    
    for i in range(len(df_background)):
        bg_name = df_background['file name'][i]
        bg_styles = df_background['style'][i].split(', ')
        bg_colors = df_background['color'][i].split(', ')
        cnt_style, cnt_color = 0, 0
        
        # file_name_list가 비어있지 않고, 새로운 폴더명으로 이동한 경우
        if file_name_list and bg_name.split('/')[0] not in file_name_list[-1]:
            print("here in file name initial")
            print("#### file_name_list : ", file_name_list)
            print("bg_name.split('/')[0] : ", bg_name.split('/')[0])
            print("file_name_list : ", file_name_list[-1])
            res_keywords_temp.sort(key=lambda x: x[2], reverse=True)
            res_keywords.extend(res_keywords_temp[:2])
            res_keywords_temp = []
            
        # file_name_list에 bg_name이 없으면 새로 넣어주자
        if bg_name.split('/')[0] not in file_name_list:
            file_name_list.append(bg_name.split('/')[0])
        
        for style in prod_styles:
            # print("bg_name : ", bg_name, " style : ", style)
            if style in bg_name:
                # print("here in style!!!! : ", bg_name)
                bg_styles = df_background['style'][i].split(', ')
                bg_colors = df_background['color'][i].split(', ')
                
                for j in range(3):
                    cnt_style += (bg_styles[j].lower() == prod_styles[j])
                    cnt_color += (bg_colors[j].lower() == prod_colors[j])
                    
                res_keywords_temp.append([bg_name, cnt_style, cnt_color])
         
    if res_keywords_temp:
        res_keywords_temp.sort(key=lambda x: x[2], reverse=True)
        res_keywords.extend(res_keywords_temp[:2])
        res_keywords_temp = []
         
    print("file_name_list : ", file_name_list)
    return res_keywords
                

def naive_match_origin(prod_styles, prod_colors, df_background):
    res_keywords = {'file name':[], 'count style' : [], 'count color' : []}
    
    for i in range(len(df_background)):
        bg_styles = df_background['style'][i].split(', ')
        bg_colors = df_background['color'][i].split(', ')
        cnt_style, cnt_color = 0, 0
        
        for j in range(3):
            cnt_style += (bg_styles[j].lower() == prod_styles[j])
            cnt_color += (bg_colors[j].lower() == prod_colors[j])
            
        res_keywords['file name'].append(df_background['file name'][i])
        res_keywords['count style'].append(cnt_style)
        res_keywords['count color'].append(cnt_color)
        
    return res_keywords

def kwords_score_match(styles, colors, style_sc, color_sc, df_background):
    res_keywords = {'file name':[], 'count style' : [], 'count color' : [], 'style similarity' : [], 'color similarity' : []}
    
    for i in range(len(df_background)):
        bg_styles = df_background['style'][i].split(', ')
        bg_colors = df_background['color'][i].split(', ')
        cnt_style, cnt_color = 0, 0
        
        for j in range(5):
            cnt_style += (bg_styles[j] == styles[j])
            cnt_color += (bg_colors[j].lower() == colors[j])
        
        df_background['style score vec'].replace('\n','', regex=True)
        style_sim_sc = cos_similarity(torch.tensor(style_sc), torch.tensor(df_background['style score vec'][i]))
        color_sim_sc = cos_similarity(torch.tensor(color_sc), torch.tensor(df_background['color score vec'][i]))
        
        res_keywords['file name'].append(df_background['file name'][i])
        res_keywords['count style'].append(cnt_style)
        res_keywords['count color'].append(cnt_color)
        res_keywords['style similiarity'].append(style_sim_sc)
        res_keywords['color similiarity'].append(color_sim_sc)
        
    return res_keywords
    

res_keywords = naive_match(styles, colors, df_bg) 
print('result : ', res_keywords)
#####
# keywords_scores_dict = kwords_score_match(prod_styles, prod_colors, prod_style_scores, prod_color_scores, df_bg)
# df_kwords_and_scores = pd.DataFrame(keywords_scores_dict)
# df_kwords_and_scores.set_index("file name", inplace=True)

# df_kwords_and_scores.to_csv('../results/kwords_and_scores.csv', index=False)

