# GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# importar os pacotes necessários
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import glob
import os
import pandas as pd


import SimpleITK as sitk
import time
from tqdm import tqdm
import traceback

# width = new image width (needs to be bigger than current)
def augmentation(exam_id, input_path, output_path):
    try:
        print(exam_id + ':')
        
        image = sitk.ReadImage(input_path)
        npyImage = sitk.GetArrayFromImage(image)

        imgAug = ImageDataGenerator(horizontal_flip=True,
                                    vertical_flip=True,
                                    rotation_range=90,
                                    fill_mode='nearest',
                                    #novos
                                    zoom_range=[0.3,1.0],
                                   )

        imgGen = imgAug.flow(image, save_to_dir=output_path,
                             save_format="nii.gz", save_prefix=exam_id+'_')
        
        # zerar padding
        # imgMin = np.amin(imgAug)
        # npyImage_aux = imgAug
        # npyImage_aux = imgAug - imgMin

        itkImage = sitk.GetImageFromArray(npyImage_aux)
        sitk.WriteImage(itkImage, output_path)

        del image

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def exec_augmentation(src_dir, dst_dir, ext, width, reverse = False, desc = None):
    try:
        os.stat(dst_dir)
    except:
        os.mkdir(dst_dir)    

    input_pathAll = glob.glob(src_dir + '/*' + ext)
    input_pathAll.sort(reverse=reverse) 

    exam_ids = []
    input_paths = []
    output_paths = []

    for input_path in input_pathAll:
        exam_id = os.path.basename(input_path.replace(ext, ''))
        output_path = dst_dir + '/' + exam_id + '_zeroPedding' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc=desc)):
        augmentation(exam_id, input_paths[i], output_paths[i])
            
def main():
    ext = '.nii.gz'
    width = 640 # new width
    dataset = 'dataset1'
    main_dir = f'/home/anatielsantos/mestrado/datasets/dissertacao/{dataset}/image/ZeroPedding/'
    
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/DataAugmentation'.format(main_dir)

    exec_augmentation(src_dir, dst_dir, ext, width, reverse = False, desc = 'Data Augmentation')

if __name__=="__main__":    
    start = time.time()
    main()
    stop = time.time()
    print("Elapsed time: "+str(stop - start)+" seconds")






















# def calcula_fator_multiplicador_classe(path_raiz,fator=False):
    
#     path_classes = sorted(glob.glob(path_raiz+"*"))

#     peso_classes = []
#     for classe in path_classes:
#         imagens_da_classe = len(sorted(glob.glob(classe+'/*')))
#         peso_classes.append(imagens_da_classe)
#     maior = np.max(peso_classes)

#     multiplicador_por_classe = []
#     for classe in path_classes:
#         imagens_da_classe = len(sorted(glob.glob(classe+'/*')))
        
#         if not fator:
#             fator_classe = int(maior // imagens_da_classe)
#         else:
#             fator_classe = int(fator // imagens_da_classe)
#         nome_da_classe = os.path.basename(classe)
#         multiplicador_por_classe.append([nome_da_classe,fator_classe])
    
#     pesos = pd.DataFrame(multiplicador_por_classe).T 

#     return pesos

# def salva_imagens_dataaugmentation(path_pasta,formato,numero_de_imagens):
    
#     imagens_da_classe = sorted(glob.glob(path_pasta+'*'))
    
#     for img in imagens_da_classe:

#         # definir caminhos da imagem original e diretório do output
#         IMAGE_PATH = img
#         OUTPUT_PATH = path_pasta

#         # nome da imagem para ser o prefixo
#         nome_da_img = os.path.basename(IMAGE_PATH).split('.')[0]


#         # carregar a imagem original e converter em array
#         image = load_img(IMAGE_PATH)
#         image = img_to_array(image)

#         # adicionar uma dimensão extra no array
#         image = np.expand_dims(image, axis=0)

#         # criar um gerador (generator) com as imagens do data augmentation
#         # aqui é onde são definidos os tipos de dataaugmentation
#         imgAug = ImageDataGenerator(horizontal_flip=True,
#                                     vertical_flip=True,
#                                     rotation_range=90,
#                                     fill_mode='nearest',
#                                     #novos
#                                     zoom_range=[0.3,1.0],
                                    
#                                    )

#         imgGen = imgAug.flow(image, save_to_dir=OUTPUT_PATH,
#                              save_format=formato, save_prefix=nome_da_img+'_')

#         # gerar n imagens por data augmentation
#         counter = 0
#         for (i, newImage) in enumerate(imgGen):
#             counter += 1

#             # ao gerar n imagens, parar o loop
#             if counter == numero_de_imagens:
#                 break
#     print("fim do aumento de dados na classe")

# # Visualizar o nome e o peso de cada classe
# # insira o caminho da pasta que possui as classes
# # path_raiz = '/home/rafael/Documentos/mestrado/ISIC2018CLASSIFICACAO/512-AUG/test/'
# # path_raiz = "/home/rafael/Documentos/mestrado/DERMIS/DERMIS512-SEGMED-AUG4000/"
# # path_raiz = "/home/rafael/Documentos/mestrado/DERMIS/DERMIS512-SEGMED-AUG4-TRAINTESTVAL/train/"
# # path_raiz = "/home/rafael/Documentos/mestrado/FUSAO/1FUSAO-TRAINTESTVAL602020-AUG/val/"
# path_raiz = "/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/image/ZeroPedding/"

# # lembrar que caso queira criar baseado em 1000 imagens para cada classe basta passar como argumento calcula_fator_multiplicador_classe(path_raiz,1000)
# # Se não passar nenhum valor, ele vai tentar balanciar as classes de acordo com a maior classe
# pesos = calcula_fator_multiplicador_classe(path_raiz)
# pesos

# # Criar as imagens aumentadas
# for peso in pesos:
#     classe = pesos[peso]
#     nome_classe = classe[0]
#     peso_classe = classe[1]
    
#     if peso_classe > 1: #0
#         # Criar o aumento de dados
#         path_pasta = path_raiz+nome_classe+'/'
#         formato = 'nii.gz'
#         numero_de_imagens = peso_classe
#         salva_imagens_dataaugmentation(path_pasta,formato,numero_de_imagens)
#         print(nome_classe)

# print("Fim do aumento de dados.")