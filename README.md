# Robokeeper
This project trains a Tensorflow model using object detection model ssd_quant_mobilenet_v2 and trains it in TF1.15 to recognise subutteo footballs of varying colours.  It converts that model into a TFLite model capable of being executed on a raspberry pi, using the Coral USB Accelarator.  The model is then used in a python script which controls a Subutteo goalkeeper so that it moves to block the ball as the call comes towards it.
Much of the coding for this project has come from the excellent videos of edje electronics here: https://www.youtube.com/watch?v=aimSGOAUI8Y&t=2s 
whilst the training of the model is done in colab, based on the article by Alaa Sinjab https://towardsdatascience.com/detailed-tutorial-build-your-custom-real-time-object-detector-5ade1017fd2d
