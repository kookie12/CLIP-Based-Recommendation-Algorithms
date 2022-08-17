# Modified by Sehyun Kim, Gwan Hyeong Koo, 2022-07-20(July 20th, 2022), @RebuilderAI, Seoul, South Korea

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
        caption_orig_list, caption_style_color = img2text_CLIP(filePathList[0])
        rec_img_fName_list = []
        rec_img_fName_list_3d = []
        
        bg_img_list = []
        for i in range(nCaption):
            encoded_input = tokenizer(caption_orig_list[i], return_tensors='pt')
            caption_feat = bert_model(**encoded_input).pooler_output

            # For instagram image -> cosine similarity with background image
            # sim_score_list = []
            # for bg_img_fName, candidate in bg_text_feat.items():
            #     sim_score_list.append((cos_similarity(caption_feat, candidate), bg_img_fName))
            # rec_img_fName = max(sim_score_list)[1]
            # rec_img_fName_list.append(rec_img_fName)
            
            # For 3d background asset -> cosine similarity with background image
            sim_score_list_3d = []

            for bg_img_fName, candidate in bg_text_feat_3d.items():
                if bg_img_fName in rec_img_fName_list_3d:
                    # print("hey hey hey!!! candidate : ", candidate)
                    continue
                sim_score_list_3d.append((cos_similarity(caption_feat, candidate), bg_img_fName))

            
            sort_list = sorted(sim_score_list_3d, key=lambda x: x[0], reverse=True)
            sort_list_2 = sorted(sim_score_list_3d, key=lambda x: x[0], reverse=False)
            
            print()
            print("### max top-5 sim_score_list_3d : ", sort_list[0], " ", sort_list[1], " ", sort_list[2], " ", sort_list[3], " ", sort_list[4])
            print("### min top-5 sim_score_list_3d : ", sort_list_2[0], " ", sort_list_2[1], " ", sort_list_2[2], " ", sort_list_2[3], " ", sort_list_2[4])
            print("### min sim_score_list_3d : ", min(sim_score_list_3d))
            print("### maax sim_score_list_3d : ", max(sim_score_list_3d))
            rec_img_fName_3d = max(sim_score_list_3d)[1]
            rec_img_fName_list_3d.append(rec_img_fName_3d)
            
        # sorted_score_list = sorted(sim_score_list)
        # rec_img_fName = sorted_score_list[0][1]
        
        print("########### check here ###########")
        # print("rec_img_fName_list_3d : ", rec_img_fName_list_3d)
        rel_img_path_list = []
        for i in range(nCaption):            
            ### For instagram image
            # rec_img_fName = '000101' imgPath = 'static/instagram_img/'
            # rec_img_fPath = os.path.join(imgPath, rec_img_fName_list_3d[i])
            # # if img file extension is jpg
            # if os.path.isfile(rec_img_fPath + '.jpg'):
            #     fName_with_ext = rec_img_fName_list[i] + '.jpg'
            # # if img file extension is png
            # elif os.path.isfile(rec_img_fPath + '.png'):
            #     fName_with_ext = rec_img_fName_list[i] + '.png'   
            # rel_img_path_list.append(os.path.join('instagram_img', fName_with_ext))
            
            ### For 3d background asset
            rel_img_path_list.append(rec_img_fName_list_3d[i])
        
        print("rel_img_path_list : ", rel_img_path_list)
        
        # find 3d background caption 
        csv_file_path = os.path.join(os.getcwd(), 'static/3d_asset_bg_caption_0727.csv')
  
        # read csv file thorugh pandas
        caption_list = []
        df = pd.read_csv(csv_file_path) 
        for item in rel_img_path_list:
            for i, image in enumerate(df['file_path'].values):
                if item == image:
                    caption_list.append(df['caption'].values[i])
                    break;
        
        caption_orig_best = caption_orig_list[0]
        caption_trans_best = translator.translate(caption_orig_best, src='en', dest='ko').text
        end_time = time.time()
        exec_time = end_time - begin_time
        
        return render_template('result.html', num_caption=nCaption, filePath=relFilePathList[0], caption_eng=caption_orig_best,
                               caption_ko=caption_trans_best, recommended_imgs=rel_img_path_list, recommended_captions=caption_list, time=round(exec_time, 2))

    except Exception as e:
        print(e)
        return render_template('fail.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9082, debug=False)