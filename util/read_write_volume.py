import SimpleITK as sitk
import traceback
import glob
import os
from tqdm import tqdm

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_volume(exam_id, input_path):
    try:
        print(exam_id + ':')

        image = sitk.ReadImage(input_path)
        npyImage = sitk.GetArrayFromImage(image)

        del image
        return npyImage

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def write_volume(exam_id, npyImage, output_path):
    try:
        print(exam_id + ':')

        itkImage = sitk.GetImageFromArray(npyImage)
        sitk.WriteImage(itkImage, output_path)

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return

def image_quantization(exam_id, input_path, output_path):
    volume = read_volume(exam_id, input_path)

    # implementar código da quantização aqui

    write_volume(exam_id, volume, output_path)

def exec_read_write(src_dir, dst_dir, ext, reverse=False):
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
        output_path = dst_dir + '/' + exam_id + '_quant' + ext

        # verifica se o arquivo ja existe
        if os.path.isfile(output_path):
            print('Arquivo ' + output_path + ' ja existe.')
            continue

        exam_ids.append(exam_id)
        input_paths.append(input_path)
        output_paths.append(output_path)

    for i, exam_id in enumerate(tqdm(exam_ids,desc="Processo iniciado...")):
        # quantizacao
        image_quantization(exam_id, input_paths[i], output_paths[i])

def main():
    ext = '.nii.gz'
    main_dir = f''  #path das imagens
    src_dir = '{}'.format(main_dir)
    dst_dir = '{}/quantizadas'.format(main_dir)

    exec_read_write(src_dir, dst_dir, ext, reverse=False)

if __name__=="__main__":
    main()
