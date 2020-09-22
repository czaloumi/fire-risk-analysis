import tensorflow
from keras.vis.visualizaiton import visualize_activation
from keras.vis.utils import utils
import matplotlib.pyplot as plt

# Build my model
model = load_model('bestmodel.h5')
print('Model loaded.')

# Name of the layer to visualize
layer_index = utils.find_layer_idx(model, 'visualized_layer')

# Swap sigmoid with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

# Only two classes: fire or not fire
classes_to_visualize = [0, 1]
classes = {
  0: 'Fire',
  1: 'Not Fire'
}

# Visualize
for number_to_visualize in classes_to_visualize:
  visualization = visualize_activation(model, layer_index, filter_indices=number_to_visualize, input_range=(0., 1.))
  plt.imshow(visualization)
  plt.title('Target = {}'.format(classes[number_to_visualize]))
  plt.show()