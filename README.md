<h2>Tensorflow-Image-Segmentation-Pre-Augmented-Hemorrhages (2025/05/23)</h2>

This is the first experiment of Image Segmentation for Hemorrhages
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a pre-augmented <a href="https://drive.google.com/file/d/1uG80TZVhV_cdSJL97LMaekx4n8gw-mBY/view?usp=sharing">
Hemorrhages-ImageMask-Dataset.zip</a>, which was derived by us from 
<a href="http://www.it.lut.fi/project/imageret">
diaretdb1_v_1_1 (http://www.it.lut.fi/project/imageret)
</a>
<br>
<br>
<b>Data Augmentation Strategy:</b><br>
 To address the limited size of <b>diaretdb1_v_1_1</b> dataset, which contains 89 images and their corresponding 
 hemorrhages ground truth, 
 we employed <a href="./generator/ImageMaskDatasetGenerator.py">an offline augmentation tool</a> to generate a pre-augmented dataset, which supports the following augmentation methods.
<br>
<li>Vertical flip</li>
<li>Horizontal flip</li>
<li>Rotation</li>
<li>Shrinks</li>
<li>Shears</li> 
<li>Deformation</li>
<li>Distortion</li>
<li>Barrel distortion</li>
<li>Pincushion distortion</li>
<br>
Please see also the following tools <br>
<li><a href="https://github.com/sarah-antillia/Image-Deformation-Tool">Image-Deformation-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Image-Distortion-Tool">Image-Distortion-Tool</a></li>
<li><a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a></li>
<br>
<hr>
<b>Actual Image Segmentation for Images of 1500x1152 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image003.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image017.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image015.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image015.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image015.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Hemorrhages Segmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the following <b>paperswithcode</b> web-site.<br>
<a href="https://paperswithcode.com/dataset/diaretdb1">https://paperswithcode.com/dataset/diaretdb1</a><br><br>
<b>DIARETDB1</b><br>
The database consists of 89 colour fundus images of which 84 contain at least mild non-proliferative<br>
 signs (Microaneurysms) of the diabetic retinopathy, and 5 are considered as normal which do not contain<br>
  any signs of the diabetic retinopathy according to all experts who participated in the evaluation. <br>
  Images were captured using the same 50 degree field-of-view digital fundus camera with varying imaging settings.<br>
   The data correspond to a good (not necessarily typical) practical situation, where the images are comparable, <br>
   and can be used to evaluate the general performance of diagnostic methods. <br>
   This data set is referred to as "calibration level 1 fundus images".
<br><br>
<b>License</b><br>
Unknown
<br>
<h3>
<a id="2">
2 Hemorrhages ImageMask Dataset
</a>
</h3>
 If you would like to train this Hemorrhages Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1uG80TZVhV_cdSJL97LMaekx4n8gw-mBY/view?usp=sharing">
Hemorrhages-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Hemorrhages
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>

On the derivation of this dataset, please refer to the following Python scripts:
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
The folder structure of the original <b>diaretdb1_v_1_1</b> is the following.<br>
<pre>
./diaretdb1_v_1_1
└─resources
    ├─example_evalresults
    ├─html
    │  └─thumbnails
    ├─images
    │  ├─ddb1_fundusimages
    │  ├─ddb1_fundusmask
    │  └─ddb1_groundtruth
    │      ├─hardexudates
    │      ├─hemorrhages
    │      ├─redsmalldots
    │      └─softexudates
    ├─testdatasets
    ├─toolkit
    └─traindatasets
</pre>
Our Hemorrhages is a 512x512 pixels pre-augmented dataset generated by the ImageMaskDatasetGenerator.py from 
1500x1152 pixels 
<b>dddb1_fundusImages</b> and their corresponding groundtruth <b>hardexudates</b>.
<pre>
./resources
  └─images
        ├─ddb1_fundusimages
        └─ddb1_groundtruth
             └─hardexudates
</pre>
<br>
<br>
<b>Hemorrhages Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/Hemorrhages_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not so large 
to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We trained Hemorrhages TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 85,86,87)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 87  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/train_console_output_at_epoch_87.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Hemorrhages.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/evaluate_console_output_at_epoch_87.png" width="720" height="auto">
<br><br>Image-Segmentation-Hemorrhages

<a href="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) to this Hemorrhages/test and dice_coef were poor as shown below.
<br>
<pre>
loss,0.3339
dice_coef,0.4562
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Hemorrhages.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image003.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image005.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image005.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image005.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image006.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image006.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image006.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image007.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image007.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image007.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image010.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image010.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/images/image020.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test/masks/image020.png" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Hemorrhages/mini_test_output/image020.jpg" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>


<h3>
References
</h3>
<b>1.The diaretdb1 diabetic retinopathy database and evaluation protocol </b><br>
T. Kauppi, V. Kalesnykiene, J.-K. Kamarainen, L. Lensu, I. Sorri, A. Raninen, et al.<br>
<a href="https://webpages.tuni.fi/vision/public_data/publications/bmvc2007_diaretdb1.pdf">
https://webpages.tuni.fi/vision/public_data/publications/bmvc2007_diaretdb1.pdf</a>

<br>
<br>
<b>2.Hard Exudates Segmentation in Diabetic Retinopathy Using DiaRetDB1</b><br>
Ma Yinghua, Yang Heng, R. Amarnath, Zeng Hui<br>
<a href="https://ieeexplore.ieee.org/document/10669034">https://ieeexplore.ieee.org/document/10669034</a>
<br>
<br>
<b>3. Detection of Early Signs of Diabetic Retinopathy Based on Textural <br>
and Morphological Information in Fundus Images</b><br>
Adrián Colomer, Jorge Igual, Valery Naranjo <br>
<a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC7071097/">
https://pmc.ncbi.nlm.nih.gov/articles/PMC7071097/
</a>
<br>
<br>
<b>4.Evaluation of fractal dimension effectiveness for damage detection in retinal background</b><br>
Adrián Colomer, Valery Naranjo, Thomas Janvier, Jose M. Mossi<br>
<a href="https://www.sciencedirect.com/science/article/pii/S0377042718300268">
https://www.sciencedirect.com/science/article/pii/S0377042718300268</a>
<br>
<br>

<b>5. Tensorflow-Image-Segmentation-Pre-Augmented-HardExudates</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-HardExudates">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Pre-Augmented-HardExudates
</a>

