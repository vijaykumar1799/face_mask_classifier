# face mask classifier


### Repository information:

To access dataset: https://www.kaggle.com/vijaykumar1799/face-mask-detection.

The python project is based on classifying whether a person is wearing, not wearing, or incorrectly wear his/her mask.
Dataset contains 3 folders labeled as their class names, each folder contains an equal number of distributed images (2994 images).


To begin, below is an image of some random set of images extracted from the dataset.

#### Note: 

Images appearing in the figure below have not defects, simply color correctionhad to applied for it to be visualized properly using matplotlib. I havent done the color mapping correction due to the fact that it is being used here for display purposes.


![Figure_1](https://user-images.githubusercontent.com/54745383/118973155-0b336b80-b97a-11eb-9e76-d84b8afd5236.png)


Inorder to classify whether a person is wearing, wearing incorrectly, or not wearing their mask correctly, a CNN model was built.

Briefly, dataset once loaded onto system for training, the data is split into training, validation, and testing data. where training data holds 70% of the original dataset, 20% is used for validation while training, and the remaining 10% to test the model once the model has completed training on the specified number of epochs.


Figure shown below represents how well the model performed. the figure displays 2 graphs over 150 iterations of training where the left graph represents the loss of training and validation data, in other words error. As for the graph on the right represents the accuracy of how well the model performed on the training and validation data.

![Figure_2](https://user-images.githubusercontent.com/54745383/118974661-bd1f6780-b97b-11eb-8dbe-dda8673f4a9c.png)


Figure shown below represents how well the trained model has performed on the testing data.

![Figure_3](https://user-images.githubusercontent.com/54745383/118976452-b1cd3b80-b97d-11eb-9f6c-5e26db21e7a7.png)


Moreover to actually test whether the model has performed well or it is not overfitting, some random images from google were extracted. by applying a cascade classifier to detect faces using openCV, faces were extracted and fed to the model for a prediction whether the face belongs to a class where the person is wearing a mask, not wearing a mask, or wearing their mask incorrectly.

##### Note: the cascade classifier didnt detect all face in the second figure shown below, so to solve this issue MTCNN should be used since it performs much well when compared to a cascade classifier.

![Figure_4](https://user-images.githubusercontent.com/54745383/118977030-65363000-b97e-11eb-9866-e38f21449c6e.png)


![Figure_5](https://user-images.githubusercontent.com/54745383/118977037-67988a00-b97e-11eb-9164-21fe2105f237.png)


![Figure_6](https://user-images.githubusercontent.com/54745383/118977050-6bc4a780-b97e-11eb-99c0-59268a11e4a2.png)


![Figure_7](https://user-images.githubusercontent.com/54745383/118977067-70895b80-b97e-11eb-9df2-4215d2a98197.png)


Overall, the trained model has performed well even when it comes to unseen data. adding to that, what can be improved is that more data should be accumulated of people wearing their masks incorrectly since there is a confusion when it comes to dealing with whether a person is wearing their mask incorrectly or not wearing. Thus improving images related to class mask_weared_incorrect should be taken into consideration.

