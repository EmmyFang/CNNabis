# CNNabis

### Motivation and Goal
It is common for tourists coming to Canada to keep a maple leaf as 
the souvenir of their visit.  However, on Oct.  18, 2018, 
Cannabis is legalized in Canada. 
Tourists are now more exposed to cannabis, so they have 
 a greater chance of mistaking a cannabis leaf for a 
 maple leaf and taking it back to their home countries, 
 where possession of such a thing could be illegal. 
 It will be very helpful for them to have a tool that helps 
 them distinguish maple leaves and cannabis leaves. <br>
 Also, most tourists know nothing about their souvenir 
 beyond the fact that it is a maple leaf.  In fact, most 
 of them aren’t even aware that there are many types of 
 maple leaves out there.  We can enrich the experience 
 of their visit by telling them which type of maple leaf 
 theirs belongs to. <br>
 To address these needs, we present 
 CNNabis, a CNN-based software that classifies 7 most common types of 
 maple leaf in North America and Cannabis leaf.  It 
 tells the tourists if their souvenir is a cannabis leaf.  If 
 not, it tells the tourists what type of maple leaf it is. 


### Requirements:
1. Python 3.6
2. pytorch
3. tkinter
4. PIL
5. torchvision
6. numpy
7. matplotlib


### User Guide
run user.py <br>
follow the prompts and select an image you want to classify <br>
it will load the pretrained model (model/best.pt) and output the probability
of the image belonging to the two most probable classes.


### main.py
train the models using the labeled data in data.zip and save the training and validation results.
By default, it uses the pretrained ResNet18 model. Make sure the data is in the same directory of main.py

### user.py 
run the applet user interface to predict on images the user selected <br>
it will load the pretrained model (model/best.pt) and give the probability
of the image belonging to the two most probable classes. 

### Show.py
show the images after augmentation 

### plot_confusion_matrix.py
plot confusion matrix 

### confustion_matrix.py
run prediction on test set and store required information to plot the confusion matrix

### Plot.py
plot the train and validation accuracy curves 

### model.py 
build a CNN model class

### model/best.pt
the pretrained model


### Final_Report.pdf
the project report



