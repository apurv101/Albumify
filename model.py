
# coding: utf-8

# In[1]:

# from __future__ import print_function

# import time
# from PIL import Image
# import numpy as np

# from keras import backend
# from keras.models import Model
# from keras.applications.vgg16 import VGG16

# from scipy.optimize import fmin_l_bfgs_b
# from scipy.misc import imsave

from __future__ import print_function

import numpy as np
import math
import keras.backend as K

from keras.applications import vgg16, vgg19
from keras.applications.imagenet_utils import preprocess_input

from scipy.optimize import minimize
from scipy.misc import imread, imsave, imresize
import PIL.Image


# def style_transfer(sourceImagePath, outputPath,filterPath):
#     content_image_path = sourceImagePath
#     style_image_path = filterPath
#     height = 125
#     width = 125
#     # In[4]:

#     content_weights = [0.025]
#     style_weights = [2.5]
#     # content_weights = [0.025, 0.05, 0.75, 0.1]
#     # style_weights = [2.0, 2.5, 3.0]
#     total_variation_weights = [1.0]

#     # In[5]:


#     style_image = Image.open(style_image_path)
#     style_image = style_image.resize((height, width))

#     # In[6]:


#     content_image = Image.open(content_image_path)
#     content_image = content_image.resize((height, width))

#     # In[7]:


#     content_array = np.asarray(content_image, dtype='float32')
#     content_array = np.expand_dims(content_array, axis=0)
#     print(content_array.shape)

#     style_array = np.asarray(style_image, dtype='float32')
#     style_array = np.expand_dims(style_array, axis=0)
#     print(style_array.shape)

#     # In[8]:


#     content_array[:, :, :, 0] -= 103.939
#     content_array[:, :, :, 1] -= 116.779
#     content_array[:, :, :, 2] -= 123.68
#     content_array = content_array[:, :, :, ::-1]

#     style_array[:, :, :, 0] -= 103.939
#     style_array[:, :, :, 1] -= 116.779
#     style_array[:, :, :, 2] -= 123.68
#     style_array = style_array[:, :, :, ::-1]

#     # In[13]:


#     def content_loss(content, combination):
#         return backend.sum(backend.square(combination - content))

#     # In[14]:


#     def gram_matrix(x):
#         features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
#         gram = backend.dot(features, backend.transpose(features))
#         return gram

#     # In[15]:


#     def style_loss(style, combination):
#         S = gram_matrix(style)
#         C = gram_matrix(combination)
#         channels = 3
#         size = height * width
#         return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

#     # In[16]:


#     def total_variation_loss(x):
#         a = backend.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
#         b = backend.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
#         return backend.sum(backend.pow(a + b, 1.25))

#     # In[18]:




#     def eval_loss_and_grads(x):
#         x = x.reshape((1, height, width, 3))
#         outs = f_outputs([x])
#         loss_value = outs[0]
#         grad_values = outs[1].flatten().astype('float64')
#         return loss_value, grad_values

#     class Evaluator(object):

#         def __init__(self):
#             self.loss_value = None
#             self.grads_values = None

#         def loss(self, x):
#             assert self.loss_value is None
#             loss_value, grad_values = eval_loss_and_grads(x)
#             self.loss_value = loss_value
#             self.grad_values = grad_values
#             return self.loss_value

#         def grads(self, x):
#             assert self.loss_value is not None
#             grad_values = np.copy(self.grad_values)
#             self.loss_value = None
#             self.grad_values = None
#             return grad_values

#     # In[20]:


#     content_image = backend.variable(content_array)
#     style_image = backend.variable(style_array)
#     combination_image = backend.placeholder((1, height, width, 3))

#     input_tensor = backend.concatenate([content_image,
#                                         style_image,
#                                         combination_image], axis=0)

#     model = VGG16(input_tensor=input_tensor, weights='imagenet',
#                   include_top=False)
#     layers = dict([(layer.name, layer.output) for layer in model.layers])

#     # In[21]:


#     images = list()
#     for content_weight in content_weights:
#         for style_weight in style_weights:
#             for total_variation_weight in total_variation_weights:

#                 loss = backend.variable(0.)

#                 layer_features = layers['block2_conv2']
#                 content_image_features = layer_features[0, :, :, :]
#                 combination_features = layer_features[2, :, :, :]

#                 loss += content_weight * content_loss(content_image_features, combination_features)

#                 feature_layers = ['block1_conv2', 'block2_conv2',
#                                   'block3_conv3', 'block4_conv3',
#                                   'block5_conv3']
#                 for layer_name in feature_layers:
#                     layer_features = layers[layer_name]
#                     style_features = layer_features[1, :, :, :]
#                     combination_features = layer_features[2, :, :, :]
#                     sl = style_loss(style_features, combination_features)
#                     loss += (style_weight / len(feature_layers)) * sl

#                 loss += total_variation_weight * total_variation_loss(combination_image)
#                 grads = backend.gradients(loss, combination_image)

#                 outputs = [loss]
#                 outputs += grads
#                 f_outputs = backend.function([combination_image], outputs)

#                 evaluator = Evaluator()

#                 x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

#                 iterations = 10

#                 for i in range(iterations):
#                     print('Start of iteration', i)
#                     start_time = time.time()
#                     x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
#                     print('Current loss value:', min_val)
#                     end_time = time.time()
#                     print('Iteration %d completed in %ds' % (i, end_time - start_time))

#                 x = x.reshape((height, width, 3))
#                 x = x[:, :, ::-1]
#                 x[:, :, 0] += 103.939
#                 x[:, :, 1] += 116.779
#                 x[:, :, 2] += 123.68
#                 x = np.clip(x, 0, 255).astype('uint8')
#                 images.append(Image.fromarray(x))
#                 print("With Content_weight = ", content_weight)
#                 print("With Style Weight = ", style_weight)

#     # In[22]:

#     images[0].save(outputPath)


class NeuralStyler(object):
    def __init__(self, content_image_path, style_image_path,
                 weights_filepath=None,
                 Content_loss_function_weight=1.0, Style_loss_function_weight=0.00001,
                 save_every_n_steps=10,
                 verbose=False,
                 convnet='VGG16',
                 content_layer='block5_conv1',
                 style_layers=('block1_conv1',
                               'block2_conv1',
                               'block3_conv1',
                               'block4_conv1',
                               'block5_conv1')):
        

        if content_image_path is None:
            raise ValueError('Missing content image')
        if style_image_path is None:
            raise ValueError('Missing style image')
        if convnet not in ('VGG16', 'VGG19'):
            raise ValueError('Convnet must be one of: VGG16 or VGG19')

        self.content_image_path = content_image_path
        self.style_image_path = style_image_path

        self.Content_loss_function_weight = Content_loss_function_weight
        self.Style_loss_function_weight = Style_loss_function_weight
        self.save_every_n_steps = save_every_n_steps
        self.verbose = verbose
        
        
        self.layers = style_layers if content_layer in style_layers else style_layers + (content_layer,)

        self.iteration = 0
        self.step = 0
        self.styled_image = None

        # VGG
        print('Using VGG')
        if convnet == 'VGG16':
            convnet = vgg16.VGG16(include_top=False, weights='imagenet' if weights_filepath is None else None)
        else:
            convnet = vgg19.VGG19(include_top=False, weights='imagenet' if weights_filepath is None else None)

        if weights_filepath is not None:
            print('Loading model weights from: %s' % weights_filepath)
            convnet.load_weights(filepath=weights_filepath)

        # Convnet output function
        self.get_convnet_output = K.function(inputs=[convnet.layers[0].input],
                                             outputs=[convnet.get_layer(t).output for t in self.layers])

        # Load picture image
        original_picture_image = imread(content_image_path)

        self.image_shape = (original_picture_image.shape[0], original_picture_image.shape[1], 3)
        self.e_image_shape = (1,) + self.image_shape
        self.picture_image = self.pre_process_image(original_picture_image.reshape(self.e_image_shape).astype(K.floatx()))

        print('Loading picture: %s (%dx%d)' % (self.content_image_path,
                                               self.picture_image.shape[2],
                                               self.picture_image.shape[1]))

        picture_tensor = K.variable(value=self.get_convnet_output([self.picture_image])[self.layers.index(content_layer)])

        # Load style image
        original_style_image = imread(self.style_image_path)

        print('Loading style image: %s (%dx%d)' % (self.style_image_path,
                                                   original_style_image.shape[1],
                                                   original_style_image.shape[0]))

        # Check for style image size
        if (original_style_image.shape[0] != self.picture_image.shape[1]) or \
                (original_style_image.shape[1] != self.picture_image.shape[2]):
            # Resize image
            print('Resizing style image to match picture size: (%dx%d)' %
                  (self.picture_image.shape[2], self.picture_image.shape[1]))

            original_style_image = imresize(original_style_image,
                                                 size=(self.picture_image.shape[1], self.picture_image.shape[2]),
                                             interp='lanczos')

            

        self.style_image = self.pre_process_image(original_style_image.reshape(self.e_image_shape).astype(K.floatx()))
       
    
        # Create style tensors
        style_outputs = self.get_convnet_output([self.style_image])
        style_tensors = [self.gramian(o) for o in style_outputs]

        # Compute loss function(s)
        print('Compiling loss and gradient functions')

        # Picture loss function
        picture_loss_function = 0.5 * K.sum(K.square(picture_tensor - convnet.get_layer(content_layer).output))

        # Style loss function
        style_loss_function = 0.0
        style_loss_function_weight = 1.0 / float(len(style_layers))

        for i, style_layer in enumerate(style_layers):
            style_loss_function += \
                (style_loss_function_weight *
                (1.0 / (4.0 * (style_outputs[i].shape[1] ** 2.0) * (style_outputs[i].shape[3] ** 2.0))) *
                 K.sum(K.square(style_tensors[i] - self.gramian(convnet.get_layer(style_layer).output))))

        # Composite loss function
        composite_loss_function = (self.Content_loss_function_weight * picture_loss_function) + \
                                  (self.Style_loss_function_weight * style_loss_function)

        loss_function_inputs = [convnet.get_layer(l).output for l in self.layers]
        loss_function_inputs.append(convnet.layers[0].input)

        self.loss_function = K.function(inputs=loss_function_inputs,
                                        outputs=[composite_loss_function])

        # Composite loss function gradient
        loss_gradient = K.gradients(loss=composite_loss_function, variables=[convnet.layers[0].input])

        self.loss_function_gradient = K.function(inputs=[convnet.layers[0].input],
                                                 outputs=loss_gradient)

    def fit(self, iterations=10, canvas='random', canvas_image_filepath=None, optimization_method='CG'):

        if canvas not in ('random', 'random_from_style', 'random_from_picture', 'style', 'picture', 'custom'):
            raise ValueError('Canvas must be one of: random, random_from_style, '
                             'random_from_picture, style, picture, custom')

        # Generate random image
        if canvas == 'random':
            self.styled_image = self.pre_process_image(np.random.uniform(0, 256,
                                                                         size=self.e_image_shape).astype(K.floatx()))
        elif canvas == 'style':
            self.styled_image = self.style_image.copy()
        elif canvas == 'picture':
            self.styled_image = self.picture_image.copy()
        elif canvas == 'custom':
            self.styled_image = self.pre_process_image(imread(canvas_image_filepath).
                                                       reshape(self.e_image_shape).astype(K.floatx()))
        else:
            self.styled_image = np.ndarray(shape=self.e_image_shape)

            for x in range(self.picture_image.shape[2]):
                for y in range(self.picture_image.shape[1]):
                    x_p = np.random.randint(0, self.picture_image.shape[2] - 1)
                    y_p = np.random.randint(0, self.picture_image.shape[1] - 1)
                    self.styled_image[0, y, x, :] = \
                        self.style_image[0, y_p, x_p, :] if canvas == 'random_from_style' \
                        else self.picture_image[0, y_p, x_p, :]

        bounds = None

        # Set bounds if the optimization method supports them
        if optimization_method in ('L-BFGS-B', 'TNC', 'SLSQP'):
            bounds = np.ndarray(shape=(self.styled_image.flatten().shape[0], 2))
            bounds[:, 0] = -128.0
            bounds[:, 1] = 128.0

        print('Starting optimization with method: %r' % optimization_method)

        for _ in range(iterations):
            self.iteration += 1

            if self.verbose:
                print('Starting iteration: %d' % self.iteration)

            minimize(fun=self.loss, x0=self.styled_image.flatten(), jac=self.loss_gradient,
                     callback=self.callback, bounds=bounds, method=optimization_method)

            self.save_image(self.styled_image)

    def loss(self, image):
        outputs = self.get_convnet_output([image.reshape(self.e_image_shape).astype(K.floatx())])
        outputs.append(image.reshape(self.e_image_shape).astype(K.floatx()))

        v_loss = self.loss_function(outputs)[0]

        if self.verbose:
            print('\tLoss: %.2f' % v_loss)

        # Check whether loss has become NaN
        if math.isnan(v_loss):
            print('NaN Loss function value')

        return v_loss

    def loss_gradient(self, image):
        return np.array(self.loss_function_gradient([image.reshape(self.e_image_shape).astype(K.floatx())])).\
            astype('float64').flatten()

    def callback(self, image):
        self.step += 1
        self.styled_image = image.copy()

        if self.verbose:
            print('Optimization step: %d/%d' % (self.step, self.iteration))

        if self.step == 1 or self.step % self.save_every_n_steps == 0:
            self.save_image(image)

    def save_image(self, image):
        imsave(self.destination_folder + 'img_' + str(self.step) + '_' + str(self.iteration) + '.jpg',
               self.post_process_image(image.reshape(self.e_image_shape).copy()))

    @staticmethod
    def gramian(filters):
        c_filters = K.batch_flatten(K.permute_dimensions(K.squeeze(filters, axis=0), pattern=(2, 0, 1)))
        return K.dot(c_filters, K.transpose(c_filters))

    @staticmethod
    def pre_process_image(image):
        return preprocess_input(image)

    @staticmethod
    def post_process_image(image):
        image[:, :, :, 0] += 103.939
        image[:, :, :, 1] += 116.779
        image[:, :, :, 2] += 123.68
        return np.clip(image[:, :, :, ::-1], 0, 255).astype('uint8')[0]



def style_transfer():
    print('start')
    neural_styler = NeuralStyler(content_image_path='input/maggie-grace-portrait-wallpapers_14105_1024x768.jpg',
                                     style_image_path='output/darksideofthemoon.jpg',

                                     # If you have a local copy of Keras VGG16/19 weights
                                     # weights_filepath='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

                                     Content_loss_function_weight=0.4,
                                     Style_loss_function_weight=0.6,
                                     verbose=True,
                                     content_layer='block4_conv1',
                                     style_layers=('block1_conv1',
                                                   'block2_conv1',
                                                   'block3_conv1',
                                                   'block4_conv1',
                                                   'block5_conv1'))

        # Create styled image
        #neural_styler.fit(canvas='picture', optimization_method='L-BFGS-B')
        # or
        # neural_styler.fit(canvas='picture', optimization_method='CG')

        # Try also
        #
    neural_styler.fit(canvas='random_from_style', optimization_method='TNC')
        # and
        # neural_styler.fit(canvas='style')
        #
        # with different optimization algorithms (CG, etc.)
