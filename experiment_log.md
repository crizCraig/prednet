experiment 0001
===========================================================================================

training 154 1024 x 1024 ped images with identical test and validation sets (i know, i know)

```
Using TensorFlow backend.
Epoch 1/150
500/500 [==============================] - 5223s - loss: 0.0702 - val_loss: 0.0440
Epoch 2/150
140/500 [=======>......................] - ETA: 3482s - loss: 0.0401


500/500 [==============================] - 5184s - loss: 0.0379 - val_loss: 0.0381
Epoch 3/150
500/500 [==============================] - 5155s - loss: 0.0342 - val_loss: 0.0351
Epoch 4/150
500/500 [==============================] - 5164s - loss: 0.0312 - val_loss: 0.0331
Epoch 5/150
500/500 [==============================] - 5154s - loss: 0.0293 - val_loss: 0.0310
Epoch 6/150
500/500 [==============================] - 5142s - loss: 0.0274 - val_loss: 0.0294
Epoch 7/150
500/500 [==============================] - 5154s - loss: 0.0262 - val_loss: 0.0293
Epoch 8/150
500/500 [==============================] - 5148s - loss: 0.0251 - val_loss: 0.0272
Epoch 9/150
500/500 [==============================] - 5177s - loss: 0.0241 - val_loss: 0.0265
Epoch 10/150
500/500 [==============================] - 5194s - loss: 0.0234 - val_loss: 0.0260
Epoch 11/150
500/500 [==============================] - 5169s - loss: 0.0226 - val_loss: 0.0249
Epoch 12/150
500/500 [==============================] - 5152s - loss: 0.0222 - val_loss: 0.0246
Epoch 13/150
500/500 [==============================] - 5149s - loss: 0.0215 - val_loss: 0.0240
Epoch 14/150
500/500 [==============================] - 5158s - loss: 0.0212 - val_loss: 0.0235
Epoch 15/150
500/500 [==============================] - 5161s - loss: 0.0206 - val_loss: 0.0228
Epoch 16/150
 96/500 [====>.........................] - ETA: 3908s - loss: 0.0207^CTraceback (most recent call last):
  File "kitti_train.py", line 76, in <module>
    validation_data=val_generator, nb_val_samples=N_seq_val)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/keras/engine/training.py", line 1553, in fit_generator
    class_weight=class_weight)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/keras/engine/training.py", line 1316, in train_on_batch
    outputs = self.train_function(ins)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/keras/backend/tensorflow_backend.py", line 1900, in __call__
    feed_dict=feed_dict)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 766, in run
    run_metadata_ptr)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 964, in _run
    feed_dict_string, options, run_metadata)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1014, in _do_run
    target_list, options, run_metadata)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1021, in _do_call
    return fn(*args)
  File "/Users/craigquiter/miniconda3/envs/prednet2/lib/python2.7/site-packages/tensorflow/python/client/session.py", line 1003, in _run_fn
    status, run_metadata)
KeyboardInterrupt
```