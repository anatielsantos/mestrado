import numpy as np
import pylidc as pl
import SimpleITK as sitk
from pylidc.utils import consensus

def pylidc_roi_extract(annotators, tex):
    i = 0
    for scan in pl.query(pl.Scan):
        # annotation_groups is a list of of lists of Annotation's
        annotation_groups = scan.cluster_annotations()
        vol = scan.to_volume()

        # Next, for each annotation group, implement your criteria of what qualifies as GGO. E.g.,
        for nodule_annotations in annotation_groups:
            # Only consider nodules with 4 annotators and have >= 50% indicating GGO
            if (len(nodule_annotations) >= annotators and sum([a.texture == tex for a in nodule_annotations]) >= 1):
                consensus_mask, consensus_bbox, _ = consensus(
                    nodule_annotations,
                    clevel=0.5,
                    pad=[(5,5), (5,5), (0,0)]
                )
                k = consensus_mask.shape[-1] // 2

                # Save image and mask
                image = np.asarray(vol[consensus_bbox][:, :, k])
                mask = np.float32(np.array(consensus_mask[:, :, k]))

                img_sitk = sitk.GetImageFromArray(image)
                sitk.WriteImage(img_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/image/solid/nod{i}.nii")

                mask_sitk = sitk.GetImageFromArray(mask)
                sitk.WriteImage(mask_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/mask/solid/nod{i}.nii")
                print("Nódulo salvo")
                i += 1

pylidc_roi_extract(2, 5)