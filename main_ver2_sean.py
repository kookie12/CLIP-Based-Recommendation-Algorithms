# Modified by Sehyun Kim, Gwan Hyeong Koo, 2022-07-20(July 20th, 2022), @RebuilderAI, Seoul, South Korea

from readline import set_completion_display_matches_hook
from flask import Flask, request, render_template
import os
import datetime
import time
from transformers import BertTokenizer, BertModel
from eval2 import img2text_CLIP, seans_img2text_CLIP
from torch import nn
from googletrans import Translator
import pandas as pd

abs_path = os.path.dirname(os.path.abspath(__file__))

# In the current directory 'templates' directory has html templates(index.html, etc.)
tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

# Tokenizer and bert_model embeds caption texts into vectors(text feature vectors)
# Input of cosine similarity: a pair of text feature vectors
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
translator = Translator()
cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

# num_captions: how many captions for each uploaded image?
# num_show_images: how many recommended background asset images to show?
# bg_info_fName: csv file(dataframe) which has background asset feature information(in text)
num_captions = 5
num_show_images = 2
bg_info_fName = './results/bg_asset_keywords_KYJ_0805_new.csv'

@app.route('/', methods=['GET'])
def home(project_name='', img_paths=''):
    return render_template("home.html", project_name=project_name, img_paths=img_paths)

# If user uploads an image, the following function(upload) is called
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
        
        # Relative(static direcotry) path of uploaded images is required for displaying in the html
        uploaded_img_rel_paths = []
        uploaded_img_abs_paths = []
        
        # Save the uploaded image in 'images' folder
        for idx, req in enumerate(requests):
            rel_fPath = os.path.join(stream_id, 'images')
            rel_fPath = os.path.join(rel_fPath, f"{idx:05d}.png")
            abs_fPath = os.path.join(img_save_dir, f"{idx:05d}.png")
            req.save(abs_fPath)
            uploaded_img_rel_paths.append(rel_fPath)
            uploaded_img_abs_paths.append(abs_fPath)

        # STEP 1: Generate captions from the uploaded image
        # STEP 2: Recommend background asset for the uploaded image
        begin_time = time.time()
        
        # img2text_CLIP function takes an image file(path) as input
        # and returns captions(in string) that best describes the input image
        sorted_captions, style_rec_bg_imgs, rec_styles = img2text_CLIP(uploaded_img_abs_paths[0])
        
        caption_orig_best = sorted_captions[0]
        caption_trans_best = translator.translate(caption_orig_best, src='en', dest='ko').text
        end_time = time.time()
        exec_time = end_time - begin_time
        
        return render_template('result.html', num_caption=num_captions, filePath=uploaded_img_rel_paths[0], caption_eng=caption_orig_best,
                               caption_ko=caption_trans_best, recommended_imgs=style_rec_bg_imgs, recommended_captions=rec_styles, time=round(exec_time, 2))

    except Exception as e:
        print(e)
        return render_template('fail.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9082, debug=False)