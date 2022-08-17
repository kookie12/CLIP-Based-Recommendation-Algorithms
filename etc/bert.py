from transformers import BertTokenizer, BertModel
from torch import nn
from gensim.models import Word2Vec, KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    cos_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    img_moods = ['calm', 'monotonous',  'festive', 'gloomy', 'cozy', 'hopeful', 
                    'hopeless', 'horrible', 'scary', 'humorous', 'mysterious', 
                    'peaceful', 'romantic', 'solitary', 'urgent', 'tense', 'tragic', 'comic', 'desperate', 
                    'dynamic', 'moving', 'touching', 'encouraging', 'heartening', 'depressing',  
                    'fantastic', 'awesome', 'spectacular', 'stressful', 'lively', 'brisk',
                    'boring', 'relaxing', 'nostalgic', 'disgusting', 
                    'joyful', 'pleasant', 'grave', 'annoying',
                    'gorgeous', 'suspenseful', 'thrilling', 
                    'magnificent', 'natural', 'cheerful', 'playful', 'fun']

    img_colors = ['White', 'Yellow', 'Blue', 'Red', 'Green', 'Black', 'Brown', 'Beige', 'Ivory', 'Silver', 'Purple', 'Navy blue', 'Gray', 'Orange', 'Charcoal', 'Aquamarine', 'Coral','Khaki']

    # to find the cos similarity score
    caption_1 = ['this is a test', 'this is a test2', 'white', 'happy', 'people', 'cloud', 'car', 'angry', 'cheerful', 'a']
    caption_2 = ['this is a test', 'this is a test3', 'black', 'gloomy', 'sad', 'sun', 'annoy', 'happy', 'cheer', 'man']
    nCaption = len(caption_1)

    for i in range(nCaption):
        encoded_input_1 = tokenizer(caption_1[i], return_tensors='pt')
        caption_feat_1 = bert_model(**encoded_input_1).pooler_output

        encoded_input_2 = tokenizer(caption_2[i], return_tensors='pt')
        caption_feat_2 = bert_model(**encoded_input_2).pooler_output

        # For instagram image -> cosine similarity with background image
        # sim_score_list = []
        # for bg_img_fName, candidate in bg_text_feat.items():
        #     sim_score_list.append((cos_similarity(caption_feat, candidate), bg_img_fName))
        # rec_img_fName = max(sim_score_list)[1]
        # rec_img_fName_list.append(rec_img_fName)
        
        # For 3d background asset -> cosine similarity with background image
        sim_score_list = []
        
        sim_score_list.append((cos_similarity(caption_feat_1, caption_feat_2)))
        print("sim_score_list : ", sim_score_list)
        
        
        
def word2vec():

    # model = Word2Vec.load('word2vec_okt.model')
    # # model = Word2Vec.load_word2vec_format('path-to-vectors.txt', binary=False)
    # csv_file_path = os.path.join(os.getcwd(), 'static/3d_asset_bg_caption_0727.csv')
    model_path = os.path.join(os.getcwd(), 'GoogleNews-vectors-negative300.bin')
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print(word2vec_model.vectors.shape)
    print(word2vec_model.similarity('this', 'is'))
    print(word2vec_model.similarity('white', 'white'))
    print(word2vec_model.similarity('white', 'black'))
    print(word2vec_model.similarity('white', 'yellow'))
    print(word2vec_model.similarity('white', 'beige'))
    print(word2vec_model.similarity('cozy', 'relax'))
    print(word2vec_model.similarity('gloomy', 'depressing'))
    print(word2vec_model.similarity('cheerful', 'fun'))
    print(word2vec_model.similarity('happy', 'sad'))
    print(word2vec_model.similarity('happy', 'horrible'))
    print(word2vec_model.similarity('moving', 'touching'))
    print(word2vec_model.most_similar('cozy'))
    print(word2vec_model.most_similar('relax'))
    # word2vec_model.most_similar('man')
    
def tsne_plot():
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    img_moods = ['calm', 'monotonous',  'festive', 'gloomy', 'cozy', 'hopeful', 
                    'hopeless', 'horrible', 'scary', 'humorous', 'mysterious', 
                    'peaceful', 'romantic', 'solitary', 'urgent', 'tense', 'tragic', 'comic', 'desperate', 
                    'dynamic', 'moving', 'touching', 'encouraging', 'heartening', 'depressing',  
                    'fantastic', 'awesome', 'spectacular', 'stressful', 'lively', 'brisk',
                    'boring', 'relaxing', 'nostalgic', 'disgusting', 
                    'joyful', 'pleasant', 'grave', 'annoying',
                    'gorgeous', 'suspenseful', 'thrilling', 
                    'magnificent', 'natural', 'cheerful', 'playful', 'fun']

    img_colors = ['White', 'Yellow', 'Blue', 'Red', 'Green', 'Black', 'Brown', 'Beige', 'Ivory', 'Silver', 'Purple', 'Navy blue', 'Gray', 'Orange', 'Charcoal', 'Aquamarine', 'Coral','Khaki']
    
    model_path = os.path.join(os.getcwd(), 'GoogleNews-vectors-negative300.bin')
    word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    
    for word in img_moods:
        
        tokens.append(word2vec_model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(5, 5)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
def plot_2d_graph(vocabs, xs, ys):
    fig = plt.figure()
    fig.setsize_inches(40, 20)
    plt.figure(figsize=(1, 1, 1))
    plt.scatter(xs, ys, marker = 'o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))
        
def word2vec2():
    sentences = [
                ['this', 'is', 'a',   'good',      'product'],
                ['it',   'is', 'a',   'excellent', 'product'],
                ['it',   'is', 'a',   'bad',       'product'],
                ['that', 'is', 'the', 'worst',     'product']
            ]

    # 문장을 이용하여 단어와 벡터를 생성한다.
    model = Word2Vec(sentences, size=300, window=3, min_count=1, workers=1)

    # 단어벡터를 구한다.
    word_vectors = model.wv

    vocabs            = word_vectors.vocab.keys()
    word_vectors_list = [word_vectors[v] for v in vocabs]

    # 단어간 유사도를 확인하다
    print(word_vectors.similarity(w1='it', w2='this'))

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    xys = pca.fit_transform(word_vectors_list)
    xs = xys[:,0]
    ys = xys[:,1]

    plot_2d_graph(vocabs, xs, ys)
    
    
if __name__ == "__main__":
    # bert()
    # word2vec()
    tsne_plot()