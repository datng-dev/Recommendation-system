import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fastai
from fastai.vision import *
from fastai.metrics import accuracy, top_k_accuracy
from annoy import AnnoyIndex
import zipfile
import time
from PIL import Image
import os, os.path
import pickle
from flask import Flask, request
import json
from load_features import SaveFeatures
from PIL import Image
import requests
from io import BytesIO
import shutil
import gc

app = Flask(__name__)

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "load_features"
        return super().find_class(module, name)

class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
        self.features = None
    def hook_fn(self, module, input, output): 
        out = output.detach().cpu().numpy()
        if isinstance(self.features, type(None)):
            self.features = out
        else:
            self.features = np.row_stack((self.features, out))
    def remove(self): 
        self.hook.remove()


#print(imgs)



@app.route('/')
def hello():
    return 'Hello World!'

@app.route('/recommend', methods=['POST'])
def recommend():
    
    imgs = []
    path = "img/"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(f)
    print(imgs)
    data_df2 = pd.DataFrame(imgs, columns=['image_path'])
    data_df2['category'] = 0

    train_image_list2 = ImageList.from_df(df=data_df2, path=path, cols='image_path').split_by_rand_pct(0.2, seed=101).label_from_df(cols='category')
    data2 = train_image_list2.transform(get_transforms(), size=224).databunch(bs=128).normalize(imagenet_stats)

    #file_to_read = open("stored_object2-18.pickle", "rb")
    #saved_features = pickle.load(file_to_read)

    with open('stored_object2-50.pickle', 'rb') as f:
        unpickler = MyCustomUnpickler(f)
        saved_features = unpickler.load()

    print('load pickle successfully')
    #file_to_read.close()

    # prepare the data for generating recommendations (exlcude test data)
    # get the embeddings from trained model
    img_path = [str(x) for x in (list(data2.train_ds.items) +list(data2.valid_ds.items))]
    #label = [data2.classes[x] for x in (list(data2.train_ds.y.items) +list(data2.valid_ds.y.items))]
    #label_id = [x for x in (list(data2.train_ds.y.items) +list(data2.valid_ds.y.items))]
    data_df_ouput = pd.DataFrame({'img_path': img_path})
    #data_df_ouput = pd.DataFrame({'img_path': img_path})
    data_df_ouput['embeddings'] = np.array(saved_features.features).tolist()
    print(data_df_ouput)

    def get_similar_images_annoy(annoy_tree, img_index, number_of_items=12):
        start = time.time()
        img_id, img_label  = data_df_ouput.iloc[img_index, [0, 1]]
        similar_img_ids = annoy_tree.get_nns_by_item(img_index, number_of_items+1)
        end = time.time()
        print(f'{(end - start) * 1000} ms')
        # ignore first item as it is always target image
        return img_id, img_label, data_df_ouput.iloc[similar_img_ids[1:]] 


    # for images similar to centroid 
    def get_similar_images_annoy_centroid(annoy_tree, vector_value, number_of_items=12):
        start = time.time()
        similar_img_ids = annoy_tree.get_nns_by_vector(vector_value, number_of_items+1)
        end = time.time()
        print(f'{(end - start) * 1000} ms')
        # ignore first item as it is always target image
        return data_df_ouput.iloc[similar_img_ids[1:]]


    # more tree = better approximation
    ntree = 100
    #"angular", "euclidean", "manhattan", "hamming", or "dot"
    metric_choice = 'angular'

    annoy_tree = AnnoyIndex(len(data_df_ouput['embeddings'][0]), metric=metric_choice)

    # # takes a while to build the tree
    for i, vector in enumerate(data_df_ouput['embeddings']):
        annoy_tree.add_item(i, vector)
    _  = annoy_tree.build(ntree)

    def centroid_embedding(outfit_embedding_list):
        number_of_outfits = outfit_embedding_list.shape[0]
        length_of_embedding = outfit_embedding_list.shape[1]
        centroid = []
        for i in range(length_of_embedding):
            centroid.append(np.sum(outfit_embedding_list[:, i])/number_of_outfits)
        return centroid

    #outfit_img_ids = [1, 2,  5, 6,  7, 8, 9, 10, 11, 12, 27]
    event = json.loads(request.data)

    outfit_img_path = []
    for url in event:
        filename = url.split('/')[-1]
        outfit_img_path.append(filename)
    
    outfit_embedding_list = []
    
    for img_path in outfit_img_path:
        outfit_embedding_list.append(data_df_ouput.loc[data_df_ouput['img_path'] == "img/"+img_path, 'embeddings'].values[0])
    outfit_embedding_list = np.array(outfit_embedding_list)
    outfit_centroid_embedding = centroid_embedding(outfit_embedding_list)
    similar_images_df = get_similar_images_annoy_centroid(annoy_tree, outfit_centroid_embedding, 10)


    similar_images=[]
    for index in similar_images_df.index.values:
        similar_images.append(data_df_ouput.iloc[index,0])
    
    similar_images_path=[]
    for image in similar_images:
        image = image.split('/')[-1]
        image_path = "https://i.imgur.com/"+image
        similar_images_path.append(image_path)
    return str(similar_images_path)

@app.route('/download')
def download():
    if os.path.exists("img"):
        shutil.rmtree('img', ignore_errors=True)

    os.mkdir("img")
    
    imgs = []
    response_API = requests.get("https://oswrs-api-app.herokuapp.com/api/recommendation/send_all")
    data_img = response_API.text
    imgs = json.loads(data_img)
    print(imgs)
    
    for url in imgs:
        img_data = requests.get(url).content
        filename = url.split('/')[-1]
        with open("img" + "/" +filename, 'wb') as handler:
            handler.write(img_data)
    return "200"
@app.route('/show')
def show():
    if os.path.exists("stored_object2-50.pickle"):
        os.remove("stored_object2-50.pickle")
    imgs = []
    path = "img/"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(f)
    print(imgs)
    data_df2 = pd.DataFrame(imgs, columns=['image_path'])
    data_df2['category'] = 0
    

    train_image_list2 = ImageList.from_df(df=data_df2, path=path, cols='image_path').split_by_rand_pct(0.2, seed=101).label_from_df(cols='category')
    data2 = train_image_list2.transform(get_transforms(), size=224).databunch(bs=128).normalize(imagenet_stats)
    
    def load_learner(data, pretrained_model, model_metrics, model_path):
        learner = cnn_learner(data, pretrained_model, metrics=model_metrics)
        learner.model = torch.nn.DataParallel(learner.model)
        learner = learner.load(os.getcwd()+model_path,strict=False,remove_module=True )
        return learner
    
    pretrained_model = models.resnet50 # simple model that can be trained on free tier
    model_metrics = [accuracy, partial(top_k_accuracy, k=1), partial(top_k_accuracy, k=5)]   
    model_path = r"/resnet50-fashion"
    learner = load_learner(data2, pretrained_model, model_metrics, model_path)
    
    saved_features = SaveFeatures(learner.model.module[1][4])
    # _= learner.get_preds(data2.train_ds)
    # _= learner.get_preds(DatasetType.Valid)

    del data_df2
    del train_image_list2
    del data2
    del model_metrics
    del pretrained_model
    del learner
    del saved_features
    gc.collect()
    # del saved_features
    # del _


    print("200")
    return "200"
@app.route('/extract')
def extract():
    if os.path.exists("img"):
        shutil.rmtree('img', ignore_errors=True)

    os.mkdir("img")
    
    imgs = []
    response_API = requests.get("https://oswrs-api-app.herokuapp.com/api/recommendation/send_all")
    data_img = response_API.text
    imgs = json.loads(data_img)
    print(imgs)
    
    for url in imgs:
        img_data = requests.get(url).content
        filename = url.split('/')[-1]
        with open("img" + "/" +filename, 'wb') as handler:
            handler.write(img_data)
#FFFF
    if os.path.exists("stored_object2-50.pickle"):
        os.remove("stored_object2-50.pickle")
    imgs = []
    path = "img/"
    valid_images = [".jpg",".gif",".png",".tga"]
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        imgs.append(f)
    print(imgs)
    data_df2 = pd.DataFrame(imgs, columns=['image_path'])
    data_df2['category'] = 0
    

    train_image_list2 = ImageList.from_df(df=data_df2, path=path, cols='image_path').split_by_rand_pct(0.2, seed=101).label_from_df(cols='category')
    data2 = train_image_list2.transform(get_transforms(), size=224).databunch(bs=128).normalize(imagenet_stats)
    
    def load_learner(data, pretrained_model, model_metrics, model_path):
        learner = cnn_learner(data, pretrained_model, metrics=model_metrics)
        learner.model = torch.nn.DataParallel(learner.model)
        learner = learner.load(os.getcwd()+model_path,strict=False,remove_module=True )
        return learner
    
    pretrained_model = models.resnet50 # simple model that can be trained on free tier
    model_metrics = [accuracy, partial(top_k_accuracy, k=1), partial(top_k_accuracy, k=5)]
       
    model_path = r"/resnet50-fashion"
    
    learner = load_learner(data2, pretrained_model, model_metrics, model_path)
    #print(learner.model.module)
    saved_features = SaveFeatures(learner.model.module[1][4])
    _= learner.get_preds(data2.train_ds)
    _= learner.get_preds(DatasetType.Valid)
   
    img_path = [str(x) for x in (list(data2.train_ds.items) +list(data2.valid_ds.items))]
    
    #label = [data2.classes[x] for x in (list(data2.train_ds.y.items) +list(data2.valid_ds.y.items))]
    #label_id = [x for x in (list(data2.train_ds.y.items) +list(data2.valid_ds.y.items))]
    data_df_ouput = pd.DataFrame({'img_path': img_path})    
    #data_df_ouput = pd.DataFrame({'img_path': img_path})
    data_df_ouput['embeddings'] = np.array(saved_features.features).tolist()
    print(data_df_ouput)

    file_to_store = open("stored_object2-50.pickle", "wb")
    pickle.dump(saved_features, file_to_store)
    file_to_store.close()

    del saved_features
    del data_df_ouput
    del learner
    del data2
    del _
    del img_path
    del file_to_store
    del data_df2
    del train_image_list2
    del model_metrics
    del pretrained_model
    

    gc.collect()

    print("200")
    
    return "500"

if __name__ == '__main__':
    from load_features import SaveFeatures
    app.run(debug=True)
    