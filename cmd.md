python3 preprocessing.py --input_dir=../ibug_landmark_68/lfpw/trainset/  --output_dir=./dataset --istrain=True --repeat=10 --img_size=112                                     --mirror_file=./Mirror68.txt
python3 DAN_V2.py -ds 1 --data_dir=./trainset/  --data_dir_test= -nlm 68 -te=60 -epe=10 -mode train
python3 DAN_V2.py -ds 1  --data_dir=./testdata --data_dir_test=  -nlm 68 -mode predict
python3 DAN_V2.py -ds 2 --data_dir=./trainset/  --data_dir_test= -nlm 68 -te=100 -epe=10 -mode train
python3 preprocessing.py --input_dir=../ibug_landmark_68/helen/testset/  --output_dir=./testdata  --istrain=False --img_size=112
python3 DAN_V2.py -ds 2 --data_dir=./testdata -nlm 68 -mode eval


https://github.com/1adrianb/face-alignment
https://github.com/goodluckcwl/Face-alignment-mobilenet-v2
https://github.com/TadasBaltrusaitis/CLM-framework
https://github.com/kylemcdonald/ofxFaceTracker

https://github.com/deepfakes/faceswap
https://github.com/shaoanlu/faceswap-GAN


https://github.com/ageitgey/face_recognition



"--train_epochs", "-te"
"--epochs_per_eval", "-epe"


resultdict = batchsize*landmarknum*2 
tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(groundtruth,resultdict['s2_ret']),-1)),-1) / tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.reduce_max(groundtruth,1),tf.reduce_min(groundtruth,1)),-1)))


tf.squared_difference: (x_g-x_p)(x_g-x_p)