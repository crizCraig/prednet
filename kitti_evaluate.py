'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''
from __future__ import print_function
import os

import hickle
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from kitti_settings import *

import cv2


# Anomaly of anomalies detection
# Check for non-persistent anomalies by looking at previous image and masking out regions that were anomalous.
# Likely we need to look at optical flow of two images and give some padding to previous anomalies based on that -
# So blur the residual with median blur to pad the residual, then residual -= blurred_previous_residual * optical flow with min of zero.
# As a simpler alternative, it could just be that detecting anomalies on vehicles, peds, bikes is enough.

n_plot = 40
batch_size = 10
num_test = 10
num_show = 1

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

print('Load trained model')
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

print('Create testing model to output predictions')
layer_config = train_model.layers[1].get_config()
layer_config['output_mode'] = 'prediction'
dim_ordering = layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = num_test
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(input=inputs, output=predictions)

test_generator = SequenceGenerator(test_file, test_sources, num_test, sequence_start_mode='unique',
                                   dim_ordering=dim_ordering)
X_test = test_generator.create_all()
print('X_test shape', X_test.shape, 'dimensions are num_sequnces, test_sequence_length, height, width, channels')

X_hat = test_model.predict(X_test, batch_size)
if dim_ordering == 'th':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

# View image with opencv


print('Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt')
# look at all timesteps except the first
X_test_flat = X_test[:, 1:].reshape(X_test.shape[0] * (X_test.shape[1] - 1), X_test.shape[2], X_test.shape[3], X_test.shape[4])
X_hat_flat = X_hat[:, 1:].reshape(X_hat.shape[0] * (X_hat.shape[1] - 1), X_hat.shape[2], X_hat.shape[3], X_hat.shape[4])

print('dumping results')
residuals = np.abs(X_test_flat - X_hat_flat)

hickle.dump({'X_test': X_test_flat, 'X_hat': X_hat_flat, 'residuals': residuals},
            DATA_DIR + 'results.hkl', compression='lzf')
exit()

model_error = ((X_test_flat - X_hat_flat)**2).reshape((X_test_flat.shape[0], -1)).mean(axis=1)
anomaly_idx = np.argsort(model_error)[::-1][:100]

mse_model = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )  # look at all timesteps except the first
mse_prev = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Previous Frame MSE: %f" % mse_prev)
f.close()

print('Plot some predictions')
num_rows = 3
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (num_show, num_rows * aspect_ratio))
gs = gridspec.GridSpec(num_rows, num_show)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir):
    os.mkdir(plot_save_dir)
# plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i, anomaly_i in enumerate(anomaly_idx):
    plt.subplot(gs[0])
    plt.imsave(plot_save_dir + 'actual_' + str(i).zfill(5) + '.png', X_test_flat[anomaly_i])
    plt.imshow(X_test_flat[anomaly_i], interpolation='none')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
    plt.ylabel('Actual', fontsize=10)

    plt.subplot(gs[num_show])
    plt.imsave(plot_save_dir + 'predicted_' + str(i).zfill(5) + '.png', X_hat_flat[anomaly_i])
    plt.imshow(X_hat_flat[anomaly_i], interpolation='none')
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
    plt.ylabel('Predicted', fontsize=10)

    plt.subplot(gs[num_show * 2])
    residual = np.abs(X_test_flat[anomaly_i] - X_hat_flat[anomaly_i])
    plt.imsave(plot_save_dir + 'residual_' + str(i).zfill(5) + '.png', residual)
    plt.imshow(residual, interpolation='none')  # TODO normalized squared error (x + 255) / 512 * 255.
    plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off',
                    labelleft='off')
    plt.ylabel('Residual', fontsize=10)

    plot_file = plot_save_dir + 'plot_' + str(i).zfill(5) + '.png'
    plt.savefig(plot_file)
    plt.clf()
    print('saved', plot_file)
