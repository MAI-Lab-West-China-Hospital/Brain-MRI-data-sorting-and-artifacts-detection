# Fully automated brain MRI sorting and artifacts detection based on image contents

This is the official implementation of “A lightweight deep learning framework for automatic MRI data sorting and artifacts detection” published in Journal of Medical Systems [DOI:10.1007/s10916-023-02017-z](https://link.springer.com/article/10.1007/s10916-023-02017-z).

Brain MRI volumes with or without pathology in the following contrasts and perspectives can be classified by this model
- T2 weighted in axial, coronal and saggital planes
- T1 weighted in axial and saggital planes
- Post-contrast T1 weighted in axial, coronal and saggital planes
- FLAIR in axial and coronal planes
- Magnetic Resonance Angiography (MRA) in axial plane

The following artifacts can be detected with this model
- Motion artifact
- Alias artifact
- Metal artifact

 ![labels](labels.png)

## How to use
1. Data prepare\
Convert DICOM to NIFTI. [dcm2niix](https://github.com/rordenlab/dcm2niix) is recommended.
2. Install the dependencies\
`pip install -r requirement.txt`\
It is highly recommended to install it in a virtual environment
3. Run the script\
`python MRI_sort.py --input /PATH/TO/NIFTIs --output /PATH/TO/SORTED/NIFTIs`

## Train with own data and custom categories
