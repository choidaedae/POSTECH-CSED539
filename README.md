# POSTECH-CSED539: Wavelet Feature Upsamplers are Efficient and Competitive in Semantic Segmentation 
- POSTECH-CSED539/AIGS539 (Computer Vision, Instructor: Suha Kwak) Final Term Project
  ![image](https://github.com/choidaedae/POSTECH-CSED539/assets/105369646/4ddc3777-5054-462c-aa10-367bd9601c76)


### Contributors
- Daehyeon Choi (POSTECH)
- Taemin Lee (POSTECH)

### Dataests
- PASCAL VOC 2012 (Augemented), 512x512

## Baseline
- DeepLabV3+, with ResNet101 as an encoder

## 1. To use our feature upsamplers
#### CARAFE-v1
![image](https://github.com/choidaedae/POSTECH-CSED539/assets/105369646/3539a6a5-9346-4483-8294-8176225ae999)
- Run this code: 
```bash
python3 train_on_wavelet_carafe_v1.py
```
  
#### CARAFE-v2
- Run this code: 
```bash
python3 train_on_wavelet_carafe_v2.py
```
#### DeFUp (Deconvolutional Feature Upsampler)
![image](https://github.com/choidaedae/POSTECH-CSED539/assets/105369646/b0a90814-8429-4d8a-87c8-26f8808bd66d)

- Run this code: 
```bash
python3 train_on_wavelet_defup.py
```

## 2. To check computational efficiency of model 
1. Set model which you want to check efficiency. (In code, 'upsampler = YOUR_MODEL') 
2. Then, run this code:
```bash
python3 effiicency_check.py
```
3. you can check 1) Mean Inference Time, 2) # of Parmaters, 3) FLOPs

 
## Project Paper
[Wavelet Feature Upsamplers are Efficient and Competitive in Semantic Segmentation.pdf](https://github.com/choidaedae/POSTECH-CSED539/files/13691630/Wavelet.Feature.Upsamplers.are.Efficient.and.Competitive.in.Semantic.Segmentation.pdf)

## Thanks to ... 
- Professor, Suha Kwak (CVLab, POSTECH)
- Most of our code is based on DeeplabV3Plus-Pytorch repository.
[Github: DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)
