import pylidc as pl
from pylidc.utils import consensus
import SimpleITK as sitk
import numpy as np

i = 0
for scan in pl.query(pl.Scan):
        # annotation_groups is a list of of lists of Annotation's
        annotation_groups = scan.cluster_annotations()
        vol = scan.to_volume()

        # Next, for each annotation group, implement your criteria of what qualifies as GGO. E.g.,
        for nodule_annotations in annotation_groups:
            # Only consider nodules with 4 annotators and have >= 50% indicating GGO
            if (len(nodule_annotations) >= 2 and sum([a.texture == 1 for a in nodule_annotations]) >= 1):
                consensus_mask, consensus_bbox, _ = consensus(
                    nodule_annotations,
                    clevel=0.5,
                    pad=[(5,5), (5,5), (0,0)]
                )

                image = np.asarray(vol[consensus_bbox][:, :, :]).transpose(2,0,1)
                mask_image = np.float32(np.array(consensus_mask[:, :, :])).transpose(2,0,1)

                img_sitk = sitk.GetImageFromArray(image)
                sitk.WriteImage(img_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/3d/image/ggo/nod{i}.nii")

                mask_sitk = sitk.GetImageFromArray(mask_image)
                sitk.WriteImage(mask_sitk, f"/home/anatielsantos/mestrado/bases/cortes-lidc/3d/mask/ggo/nod{i}.nii")

                i += 1