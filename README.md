Image-search
============

Image search and retrieval. Given a query image, we try to return the most
similar images.

Dataset: CALTECH-101

Approach: Bag of Words model (c++, openCV), GIST (matlab)

********* Compile *******************************
make  


*********** Run ********************************
./a.out 0/1 xml  
./a.out 0/1 xml img  

0/1 means train_new_model/load_saved_model
img is the query image.
