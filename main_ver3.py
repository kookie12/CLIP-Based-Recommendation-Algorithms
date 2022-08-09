# Modified by Sehyun Kim, Gwan Hyeong Koo, 2022-08-06(Aug 6th, 2022), @RebuilderAI, Seoul, South Korea

from flask import Flask, request, render_template
import os
import datetime
import time
from transformers import BertTokenizer, BertModel
from eval3 import img2text_CLIP
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
nCaption = 6

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
        
        caption_orig_list, styles_top5_images, styles_top5 = img2text_CLIP(filePathList[0])
        print("main // styles_top5_images : ", styles_top5_images)
        print("main // styles_top5 : ", styles_top5)
        # recommend_style_color = recommend_style_colors[0].lower()
        # print("recommend_style_color : ", recommend_style_color)
        # styles = []
        # colors = []
        # img_styles = ['modern', 'luxury', 'nature', 'street', 'wood', 'neon', 'antiuqe', 'traditional',
        #       'industrial', 'cozy', 'science-fiction', 'simple', 'abstract', 'office', 'bathroom']
        
        ################################################################################
        # img_styles = ['black modern', 'white modern', 'black luxury', 'white luxury', 'nature with green plant', 'street hiphop', 
        #       'dark brown wood', 'light brown wood', 'neon sign', 'antique', 'traditional', 'industrial', 'cozy', 
        #       'science-fiction', 'magical', 'pink romantic', 'artistic', 'office', 'bathroom']

        # img_colors = ['white', 'yellow', 'blue', 'red', 'green', 'black', 'brown', 'beige', 'ivory', 'silver', 'purple', 'navy', 'gray', 'orange', 'pink', 'khaki']
        # df_background = pd.read_csv('./results/bg_asset_keywords_KYJ_0805_new3.csv')
        
        # for style in img_styles:
        #     if style in recommend_style_color:
        #         print("style : ", style)
        #         styles.append(style)

        # for color in img_colors:
        #     if color in recommend_style_color:
        #         print("color : ", color)
        #         colors.append(color)

        # print("we want : ", styles, " and ", colors)

        # #### def naive_match()
        # prod_styles = styles
        # prod_colors = colors
        
        # res_keywords = []
        # res_keywords_temp = []
        # # 추천된 3개의 style을 포함하는 폴더에만 접근합니다
        # #for style in prod_styles:
        # print("here in naive_match")
        # [7_antique_18, ]
        # file_name_list = [] # only using for my thought..
        
        # for i in range(len(df_background)):
        #     # category = df_background.iloc[i]['category'] # iloc은 행을 가져오는...
        #     category = df_background['category'][i] # pandas는 column 먼저!! 
        #     bg_name = df_background['file name'][i] # 이제 이건 안쓰일듯 하다...
        #     bg_styles = df_background['style'][i].split(', ')
        #     bg_colors = df_background['color'][i].split(', ')
        #     cnt_style, cnt_color = 0, 0
            
        #     # file_name_list가 비어있지 않고, 새로운 폴더명으로 이동한 경우
        #     if file_name_list and category not in file_name_list[-1]: # bg_name.split('/')[0] -> category @kookie12
        #         print("here in file name initial")
        #         print("#### file_name_list : ", file_name_list)
        #         print("#### bg name : ", bg_name)
        #         print("category : ", category)
        #         print("file_name_list[-1] : ", file_name_list[-1])
        #         res_keywords_temp.sort(key=lambda x: x[2], reverse=True)
        #         res_keywords.extend(res_keywords_temp[:2])
        #         res_keywords_temp = []
                
        #     # file_name_list에 bg_name이 없으면 새로 넣어주자
        #     if bg_name.split('/')[0] not in file_name_list:
        #         file_name_list.append(bg_name.split('/')[0]) # @kookie12
            
        #     for style in prod_styles:
        #         # print("bg_name : ", bg_name, " style : ", style)
        #         if style in bg_name:
        #             # print("here in style!!!! : ", bg_name)
        #             bg_styles = df_background['style'][i].split(', ')
        #             bg_colors = df_background['color'][i].split(', ')
                    
        #             for j in range(3):
        #                 cnt_style += (bg_styles[j].lower() == prod_styles[j])
        #                 cnt_color += (bg_colors[j].lower() == prod_colors[j])
                        
        #             res_keywords_temp.append([bg_name, cnt_style, cnt_color]) # bg_name -> @
            
        # if res_keywords_temp:
        #     res_keywords_temp.sort(key=lambda x: x[2], reverse=True)
        #     res_keywords.extend(res_keywords_temp[:2])
        #     res_keywords_temp = []
            
        # print("file_name_list : ", file_name_list)
        
        
        #########################################################
        # rec_img_fName_list = []
        # rec_img_fName_list_3d = []
        # bg_img_list = []
        # for i in range(nCaption):
        #     encoded_input = tokenizer(caption_orig_list[i], return_tensors='pt')
        #     caption_feat = bert_model(**encoded_input).pooler_output

        #     # For instagram image -> cosine similarity with background image
        #     # sim_score_list = []
        #     # for bg_img_fName, candidate in bg_text_feat.items():
        #     #     sim_score_list.append((cos_similarity(caption_feat, candidate), bg_img_fName))
        #     # rec_img_fName = max(sim_score_list)[1]
        #     # rec_img_fName_list.append(rec_img_fName)
            
        #     # For 3d background asset -> cosine similarity with background image
        #     sim_score_list_3d = []

        #     for bg_img_fName, candidate in bg_text_feat_3d.items():
        #         if bg_img_fName in rec_img_fName_list_3d:
        #             # print("hey hey hey!!! candidate : ", candidate)
        #             continue
        #         sim_score_list_3d.append((cos_similarity(caption_feat, candidate), bg_img_fName))

            
        #     sort_list = sorted(sim_score_list_3d, key=lambda x: x[0], reverse=True)
        #     sort_list_2 = sorted(sim_score_list_3d, key=lambda x: x[0], reverse=False)
            
        #     print()
        #     print("### max top-5 sim_score_list_3d : ", sort_list[0], " ", sort_list[1], " ", sort_list[2], " ", sort_list[3], " ", sort_list[4])
        #     print("### min top-5 sim_score_list_3d : ", sort_list_2[0], " ", sort_list_2[1], " ", sort_list_2[2], " ", sort_list_2[3], " ", sort_list_2[4])
        #     print("### min sim_score_list_3d : ", min(sim_score_list_3d))
        #     print("### maax sim_score_list_3d : ", max(sim_score_list_3d))
        #     rec_img_fName_3d = max(sim_score_list_3d)[1]
        #     rec_img_fName_list_3d.append(rec_img_fName_3d)
            
        # # sorted_score_list = sorted(sim_score_list)
        # # rec_img_fName = sorted_score_list[0][1]
        
        # print("########### check here ###########")
        # # print("rec_img_fName_list_3d : ", rec_img_fName_list_3d)
        # rel_img_path_list = []
        # for i in range(nCaption):            
        #     ### For instagram image
        #     # rec_img_fName = '000101' imgPath = 'static/instagram_img/'
        #     # rec_img_fPath = os.path.join(imgPath, rec_img_fName_list_3d[i])
        #     # # if img file extension is jpg
        #     # if os.path.isfile(rec_img_fPath + '.jpg'):
        #     #     fName_with_ext = rec_img_fName_list[i] + '.jpg'
        #     # # if img file extension is png
        #     # elif os.path.isfile(rec_img_fPath + '.png'):
        #     #     fName_with_ext = rec_img_fName_list[i] + '.png'   
        #     # rel_img_path_list.append(os.path.join('instagram_img', fName_with_ext))
            
        #     ### For 3d background asset
        #     rel_img_path_list.append(rec_img_fName_list_3d[i])
        
        # print("rel_img_path_list : ", rel_img_path_list)
        
        # # find 3d background caption 
        # csv_file_path = os.path.join(os.getcwd(), 'static/3d_asset_bg_caption_0727.csv')
  
        # # read csv file thorugh pandas
        # caption_list = []
        # df = pd.read_csv(csv_file_path) 
        # for item in rel_img_path_list:
        #     for i, image in enumerate(df['file_path'].values):
        #         if item == image:
        #             caption_list.append(df['caption'].values[i])
        #             break;
        
        res_keywords = styles_top5_images
        print("result : ", res_keywords)
        rel_img_path_list = []
        # for i in range(len(res_keywords)):
        #     # 이미지 path를 보낼 때 static을 포함하면 안된다. 자동으로 static을 붙여서 연결한다!!
        #     rel_img_path_list.append(os.path.join('background_asset/KYJ', res_keywords[i][0]))
        rel_img_path_list = styles_top5_images
        
        print("rel_img_path_list : ", rel_img_path_list)
        
        caption_list = []
        
        ## trash code sorry..
        for temp in styles_top5_images:
            temp = temp.replace('background_asset/KYJ/', '')
            print("temp : ", temp)
            caption_list.append(temp)
        
        
        # image의 category를 보내주기 위해서 parsing을 합니다!
        # for item in res_keywords:
        #     for style in img_styles:
        #         if style in item[0]:
        #             caption_list.append(style)
                    
        print("caption_list : ", caption_list)
        
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