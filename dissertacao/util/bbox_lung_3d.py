import SimpleITK as sitk
import glob
import numpy as np
import traceback

# COVID-19 CT Lung and Infection Segmentation Dataset
path_src1 = glob.glob('/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/image/lung_extracted/*.gz')
path_src_lesion1 = glob.glob('/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/lesion_mask/*.gz')
path_mask_lung1 = glob.glob('/home/anatielsantos/mestrado/datasets/dissertacao/dataset1/lung_mask/*.gz')

# COVID-19 CT segmentation dataset
path_src2 = glob.glob('/home/anatielsantos/mestrado/datasets/dissertacao/dataset2/image/lung_extracted/*.gz')
path_src_lesion2 = glob.glob('/home/anatielsantos/mestrado/datasets/dissertacao/dataset2/lesion_mask/*.gz')
path_mask_lung2 = glob.glob('/home/anatielsantos/mestrado/datasets/dissertacao/dataset2/rp_lung_msk/*.gz')


def get_bounding_box_lung(image, intensity):
    image = sitk.Cast(image, sitk.sitkInt32)
    statistics = sitk.LabelStatisticsImageFilter()
    statistics.Execute(image, image)
    return statistics.GetBoundingBox(intensity)


def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def load_image(path_image, path_mask, volume):
    try:
        image = sitk.ReadImage(path_image)
        imageArr = sitk.GetArrayFromImage(image)
        imageArr_f = imageArr.astype(float)
        image = sitk.GetImageFromArray(imageArr_f)

        mask = sitk.ReadImage(path_mask)
        file_save = path_image.split("/")[-1]

        # bin mask
        mask = (mask >= 1) * 1

        x, y = 0, 0
        for i in np.unique(sitk.GetArrayFromImage(mask))[1:]:
            (min_x, max_x, min_y, max_y, min_z, max_z) = get_bounding_box_lung(
                    mask, 1
                )
            roi = image[
                min_x:max_x+1,
                min_y:max_y+1,
                min_z:max_z+1
            ]

            # bigger shape
            npyRoi = sitk.GetArrayFromImage(roi)
            if npyRoi.shape[1] > x:
                x = npyRoi.shape[1]
            if npyRoi.shape[2] > y:
                y = npyRoi.shape[2]

            # adjustImage = imadjust(npyRoi, np.amin(npyRoi), np.amax(npyRoi), 0, 3420, gamma=1)
            # truncImage = np.trunc(adjustImage).astype(np.uint32)

            # newRoi = sitk.GetImageFromArray(truncImage)

            sitk.WriteImage(roi, f"/home/anatielsantos/mestrado/datasets/dissertacao/bbox/mask/{file_save}")

        print("VOL " + str(volume + 1) + " - ROI OK")

        del image
        del mask

        return x, y

    except Exception as e:
        print("type error: " + str(e))
        print(traceback.format_exc())
        return


def get_roi_lung(path_image, path_mask):
    volume = len(path_image)
    x1, y1 = 0, 0
    for v in range(volume):
        x, y = load_image(path_image[v], path_mask[v], v)

        if x > x1:
            x1 = x
        if y > y1:
            y1 = y

    print("Maior X:", x1)
    print("Maior Y:", y1)


if __name__ == "__main__":
    get_roi_lung(path_src_lesion1, path_mask_lung1)
    get_roi_lung(path_src_lesion2, path_mask_lung2)
