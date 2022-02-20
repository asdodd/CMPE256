Video walking through notebook:
https://www.youtube.com/watch?v=aACebENHlrs

* runs on colab
* assumes CalTech images stored on Google drive
 - removed BACKGROUND_Google
 - removed Faces_easy

* Uses VGG-16, PCA


Possible extensions, things to work on
--------------------------------------
* use other CNNs than VGG-16
* play with PCA params, other than 300 (Albert)
 - accuracy vs N
 - time vs N
* can we do LSH of feature vector? (Sajit)
* view intermediate results of CNN
* try other image data sets (Alex)
 -trained
 -untrained
* compute metrics (recall@k, precision@k) (Tom)
* time each prediction
* run on SJSU HPC
* run local

