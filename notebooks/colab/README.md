# Google Colaboratory Notebooks.


## ***phaseOne_trainAutoencoder.ipynb***
Training autoencoder model from scratch or training pretrained model.

- training Autoencoder on ShapeNetCore.v2 models (voxelized to shape (20,20,20) using https://www.patrickmin.com/binvox/) 
- train/test data split https://github.com/rohitgirdhar/GenerativePredictableVoxels/tree/master/dataset/ShapeNet/splits
- shapenet models were not rotated nor were the axis changed unlike in [1] (see function readModel() in https://github.com/rohitgirdhar/GenerativePredictableVoxels/blob/master/src/voxelize/transformAndVisVoxels.py)
- trained Autoencoder model: https://drive.google.com/open?id=1dhPWTz7LcJsB89Gz3-diNNd11u-wA910
  
  - approx 25 epochs, using adam optimizer with default settings 
  
- autoencoder evaluation on ShapeNet test data: ([metrics.average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html))
  
| Class  |  Our Autoencoder | [1] before Joint | [1] after Joint   | [1] PCA  |
|:-:|:-:|:-:|:-:|:-:|
|  Total | **0.980**  | 0.976 |0.976    | 0.968   |
| Bed  | **0.946**  | 0.941  | 0.938  | 0.915  |
| Cabinet | **0.997**  | 0.993  |  0.993 |  0.990 |
| Chair  | **0.972**  | 0.964  | 0.964  |  0.948 |
| Sofa  | **0.994**  |  0.991 | 0.992  |  0.986 |
| Table  | **0.971**  | **0.971**  | 0.970  |0.967   |

- autoencoder evaluation on IKEA data:

| Class  |  Our Autoencoder | 
|:-:|:-:|
| Total | 0.814 |
| Bed  | 0.785 |
| polica za knjige | 0.701|
| Chair  | 0.729 |
|  radni stol | 0.757|
| Sofa  |0.847 |
| Table | 0.802 |
| ormar   | 0.974 |


[1] https://rohitgirdhar.github.io/GenerativePredictableVoxels/

## ***phaseTwo_generateEmbeddings.ipynb***
Generate autoencoder embedding from each voxel model. Embeddings will be used as regression targets for convnet.
- voxels data: https://cmu.app.box.com/s/wb9lw48timjzz8wkj832ggw9yeccf3fl
- embeddings for each batch (ordered batch_0.h5, batch_1.h5, batch_2,h5 etc.), https://drive.google.com/open?id=1BJ8Ai7oVhwrQYH1FJARCjCPmEfwWHZue)

## ***phaseTwo_trainImageNet.ipynb***
Regress convnet MobileNetV2 to generated embeddings.
- convnet used: https://keras.io/applications/#mobilenetv2
- image data: in same batches as voxel data (https://cmu.app.box.com/s/wb9lw48timjzz8wkj832ggw9yeccf3fl)

## Work in progress...
2 ideas.
## ***phaseThree_trainJoint.ipynb***

## ***phaseThree_trainJoint_targetTensors.ipynb***
