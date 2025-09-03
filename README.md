<h1>Chest XRAY Segmentation</h1>
<div align="center">
    <img src="./figures/image.png" width=250px> <img src="./figures/mask.png" width=250px>
</div>
</br>

<div>
    In this project, I used UNet with an EfficientNetB3 backbone to perform binary semantic segmentation to isolate the lungs from chest X-RAY images from the <a href="https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database">COVID-19 Radiography dataset</a>. Preprocessing was performed by applying adaptive histogram equalization to enhance contrast of the low-contrast X-Ray images, followed by a resizing operation to match the input dimensions of EfficientNetB3 (300x300). Binary cross-entropy loss was used during the training step.
</div>
</br>

<div>
    During training, the average loss and dice score for each epoch were logged and visualized in the figure below. The loss appears to stabilize around epoch 8, while the dice score increased rapidly during epochs 0–1 and continued to improve more-gradually afterwards. Based on the dice score, it appears that this model generalizes excellently to the chest X-Ray images, with an approximate pixel-level accuracy of 98%. 
</div>
</br>

<div align="center">
    <img src = "./figures/figure1.png">
</div>
</br>

<div>
    The model achieved an average loss of 0.0213 on the test set, with a Dice score of 0.9824, indicating near-perfect lung segmentation in the provided X-ray images. Examples of predicted masks are shown below. For further improvement, traditional post-processing techniques such as morphological operations and the watershed algorithm could be applied to refine the segmentation results.
</div>
</br>

<div align="center">
    <img src="./figures/pred_img1.png" width=250px>
    <img src="./figures/pred1.png" width=250px>
    </br>
    <img src="./figures/pred_img2.png" width=250px>
    <img src="./figures/pred2.png" width=250px>
    </br>
    <img src="./figures/pred_img3.png" width=250px>
    <img src="./figures/pred3.png" width=250px>
    </br>
</div>

<h2>Usage:</h2>
Training:

    python main.py train train_settings.toml

</br>
Inference:

    python main.py inference inference_settings.toml
    
