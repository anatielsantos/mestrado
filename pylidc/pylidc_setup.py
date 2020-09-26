import numpy as np
import SimpleITK as sitk
import pylidc as pl
#import matplotlib.pyplot as plt
from pylidc.utils import consensus

for i in range(1010):
    if i+1 < 10:
        ct = 'LIDC-IDRI-000' + str(i+1)
    elif (i+1 >= 10 and i+1 < 100):
        ct = 'LIDC-IDRI-00' + str(i+1)
    elif (i+1 >= 100 and i+1 < 1000):
        ct = 'LIDC-IDRI-0' + str(i+1)
    else:
        ct = 'LIDC-IDRI-' + str(i+1)

    print("Pacient ID: ", str(ct))
    
    try:
        # Query for a scan, and convert it to an array volume.
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == ct).first()
        # scan1 = pl.query(pl.Annotation).filter(pl.Annotation.texture == 1)

        vol = scan.to_volume()
        nodules = scan.cluster_annotations()

        for l in range(len(nodules)):
            annotations = nodules[l]
            consensus_mask, consensus_bbox, _ = consensus(
                annotations,
                clevel=0.5,
                pad=[(0,0), (0,0), (0,0)]
            )

            k = consensus_mask.shape[-1] // 2

            # Save image and mask
            image = np.asarray(vol[consensus_bbox][:, :, k])
            mask = np.float32(np.array(consensus_mask[:, :, k]))

            img_sitk = sitk.GetImageFromArray(image)
            sitk.WriteImage(img_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/image/__vol{i}_nod{l}.nii")

            mask_sitk = sitk.GetImageFromArray(mask)
            sitk.WriteImage(mask_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/mask/__vol{i}_nod{l}.nii")
    except:
        print("ERRO")