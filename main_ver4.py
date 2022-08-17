# Modified by Sehyun Kim, Gwan Hyeong Koo, 2022-08-06(Aug 6th, 2022), @RebuilderAI, Seoul, South Korea

from eval4 import img2text_CLIP
from flask import Flask, request, render_template
import os    
import datetime
import time
from transformers import BertTokenizer, BertModel
from googletrans import Translator

abs_path = os.path.dirname(os.path.abspath(__file__))
# In the current directory 'templates' directory has html templates(index.html, etc.)
tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

# tokenizer and bert_model embeds caption texts into vectors(text feature vectors)
# Cosine similarity can be calculated from a pair of text feature vectors
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
translator = Translator()
n_caption = 6

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
        fPath_list = []
        # Relative path of the uploaded images is needed to display(through render_template() ) them in the browser
        rel_fPath_list = []
        
        # Save uploaded image in the 'images' folder
        for idx, req in enumerate(requests):
            relFilePath = os.path.join(stream_id, 'images')
            relFilePath = os.path.join(relFilePath, f"{idx:05d}.png")
            filePath = os.path.join(img_save_dir, f"{idx:05d}.png")
            req.save(filePath)
            rel_fPath_list.append(relFilePath)
            fPath_list.append(filePath)

        # Begin step 1) generating a caption from uploaded image step 2) recommending background asset
        begin_time = time.time()
        # img2text_CLIP takes an image file(path) and returns a caption(text) that describes input image the best
        sorted_captions_show, cos_top5_images, moods_top5, colors_top5 = img2text_CLIP(fPath_list[0])

        rel_img_path_list = []
        rel_img_path_list = cos_top5_images
        
        caption_list = []
        for temp in cos_top5_images:
            #temp = temp.replace('acon', '')
            temp = temp.split('/')[1]
            caption_list.append(temp)
        
        # image의 category를 보내주기 위해서 parsing을 합니다!
        caption_orig_best = sorted_captions_show[0]
        caption_trans_best = translator.translate(caption_orig_best, src='en', dest='ko').text
        end_time = time.time()
        exec_time = end_time - begin_time
        
        return render_template('result.html', num_caption=n_caption, filePath=rel_fPath_list[0], 
                               caption_eng=caption_orig_best,
                               caption_ko=caption_trans_best, recommended_imgs=rel_img_path_list, 
                               recommended_captions=caption_list, time=round(exec_time, 2))

    except Exception as e:
        print(e)
        return render_template('fail.html')
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9082, debug=False)