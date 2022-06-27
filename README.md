# Plant Disease Classification

## Keras implementation for plant disease classification

### Dataset
The plantVillage dataset contains 54,305 images with of size (256, 256, 3) containing 38 classes of different plant diseases.
Training
The networks were trained using the Google’s Colab GPU with the following configurations
- Train – validation – test split = 80%, 10%, 10%
- Loss Fn : Categorical Crossentropy
- Epochs : 30
- Batch size : 32
- Optimiser : Adam
- Learning rate : 0.001

The training images were augmented by performing various rotations, shear range, zoom range and normalised using the keras Image generator function. The validation images were only normalised without any additional augmentation. 
Network
The basis of each architecture contains a convolutional layer batch normalised with ‘relu’ activation which is considered as a convolutional block. Batch normalisation speeds up the training as well act as a regulariser to some extent by reducing generalization error. The ‘relu’ activation function is the most commonly used activation in CNNs and Perceptron because it overcomes the vanishing gradient problem.

### Trial 1
In the initial network, the depth of the convolutional layers decreases towards the end. This is opposite to that of most modern architectures. Having more depth in the initial convolutional layer increases the features extracted from the input image. This reduces the number of parameters used for the network but the accuracy was not up to the mark.

### Trial 2
This network architecture uses more parameters. Unlike the Trial 1 network architecture, this network employs the conventional method having more depth at the higher layers as compared to the lower ones. This has caused an increase in accuracy as well as parameters because the thinner initial layers were able to capture more abstract details of the plant images as compared to trial 1 network.

### Trial 3
The filter sizes of the initial convolutional layers were reduced and that of the final layers increased as compared to the trial2 network. This provided a decrease in the number of parameters and hence decreased the accuracy of the network as well.

### Trial 4
This network is a more fine-tuned network as compared to trial 3 without much significant improvement in results. This network included an additional middle convolutional layer.

### Trial 5
Another convolutional layer was added at the final stage of the network as compared to trial 3 network. This had increased the parameters significantly. This layer helped the network capture higher level features of the plant images more effectively and hence significantly increased the accuracy of classification.

### Trial 6
Unlike the other architectures, trial 6 network included another dense layer at the classifier part of the network. Although it uses a similar architecture of trial 3 it achieved more accuracy with about half as many parameters compared to trial 4 network. Including the dense layer of 64 , the network was able to achieve a trade-off between the number of parameters and accuracy. This network as compared to the other network gives more emphasis on classification rather than feature extraction.

### Results
![Results](https://github.com/atulj6631//plant_disease_classification/blob/master/Results.png?raw=true)

### Conclusion
With less parameters, the memory required to store the module is less as well as the computations required for such networks is low. This enables these networks to be ported to small scale microprocessors and microcontrollers.

