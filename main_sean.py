# Modified by Sehyun Kim, Gwan Hyeong Koo, 2022-08-04(August 4th, 2022), @RebuilderAI, Seoul, South Korea

from flask import Flask, request, render_template
import os
import datetime
import time
from transformers import BertTokenizer, BertModel
from eval import img2text_CLIP
import pickle
from torch import nn
from googletrans import Translator
import csv
import pandas as pd

abs_path = os.path.dirname(os.path.abspath(__file__))
# In the current directory 'templates' directory has html templates(index.html, etc.)
tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

# tokenizer and bert_model embeds caption texts into vectors(text feature vectors)
# Cosine similarity can be calculated from a pair of text feature vectors
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
translator = Translator()
cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)
nCaption = 5

imgPath = os.path.join(abs_path, 'static')

with open("bg_text_feat.pkl", "rb") as fd:
    bg_text_feat = pickle.load(fd)

with open("bg_text_feat_3d.pkl", "rb") as fd:
    bg_text_feat_3d = pickle.load(fd)

# Returns homepage
@app.route('/', methods=['GET'])
def home(project_name='', img_paths=''):
    return render_template("home.html", project_name=project_name, img_paths=img_paths)

# If user uploads an image, this function is called
@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Create 'static' folder in the current directory if it does not exist
        stream_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        save_dir = os.path.join(abs_path, 'static', stream_id)
        # Create 'images' folder in the 'static' folder if it does not exist
        img_save_dir = os.path.join(save_dir, 'images')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
            os.mkdir(img_save_dir)
        
        requests = request.files.getlist('images')
        filePathList = []
        # Relative path of the uploaded images is needed to display(through render_template() ) them in the browser
        relFilePathList = []
        
        # Save uploaded image in the 'images' folder
        for idx, req in enumerate(requests):
            relFilePath = os.path.join(stream_id, 'images')
            relFilePath = os.path.join(relFilePath, f"{idx:05d}.png")
            filePath = os.path.join(img_save_dir, f"{idx:05d}.png")
            req.save(filePath)
            relFilePathList.append(relFilePath)
            filePathList.append(filePath)

        # Begin step 1) generating a caption from uploaded image step 2) recommending background asset
        begin_time = time.time()
        # img2text_CLIP takes an image file(path) and returns a caption(text) that describes input image the best
        print("filePathList[0] : ", filePathList[0]) # home/ubuntu/...
        caption_orig_list, recommend_style_colors = img2text_CLIP(filePathList[0])
        
        recommend_style_color = recommend_style_colors[0].lower()
        
        styles = []
        colors = []
        img_styles = ['modern', 'luxury', 'nature', 'street', 'wood', 'neon', 'antiuqe', 'traditional',
              'industrial', 'cozy', 'science-fiction', 'simple', 'abstract', 'office', 'bathroom']

        img_colors = ['white', 'yellow', 'blue', 'red', 'green', 'black', 'brown', 'beige', 'ivory', 'silver', 'purple', 'navy', 'gray', 'orange', 'pink', 'khaki']
        df_background = pd.read_csv('./results/bg_asset_keywords_KYJ_new.csv')
        
        for style in img_styles:
            if style in recommend_style_color:
                print("style : ", style)
                styles.append(style)

        for color in img_colors:
            if color in recommend_style_color:
                print("color : ", color)
                colors.append(color)

        print("we want : ", styles, " and ", colors)

        #### def naive_match()
        prod_styles = styles
        prod_colors = colors
        
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
        
        print("result : ", res_keywords)
        rel_img_path_list = []
        for i in range(len(res_keywords)):
            rel_img_path_list.append(os.path.join('background_asset/KYJ', res_keywords[i][0]))
        
        print("rel_img_path_list : ", rel_img_path_list)
        
        caption_orig_best = caption_orig_list[0]
        caption_trans_best = translator.translate(caption_orig_best, src='en', dest='ko').text
        end_time = time.time()
        exec_time = end_time - begin_time
        
        return render_template('result.html', num_caption=nCaption, filePath=relFilePathList[0], caption_eng=caption_orig_best,
                               caption_ko=caption_trans_best, recommended_imgs=rel_img_path_list, time=round(exec_time, 2))

    except Exception as e:
        print(e)
        return render_template('fail.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9082, debug=False)