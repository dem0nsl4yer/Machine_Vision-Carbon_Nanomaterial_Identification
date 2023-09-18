import tensorflow
from PIL import Image
import requests
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import tensorflow as tf


def predict_class(model, images, show = True):
  for img in images:
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    #q_list = ['substrate','mdiamond','nanotube','graphene','qcarbon','ndiamond','cvddiamond']
    q_list = ['acarbon', 'cvddiamond', 'graphene', 'mdiamond', 'ndiamond', 'qcarbon', 'substrate']
    q_list.sort()
    pred_value = q_list[index]
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()
        print(pred_value)

def get_attribution(model, allo):
    img = image.load_img(allo, target_size=(299, 299))
    img = image.img_to_array(img)
    img /= 255.
    f, ax = plt.subplots(1, 3, figsize=(15, 15))
    #print(img.shape)
    ax[0].imshow(img)

    img = np.expand_dims(img, axis=0)

    last_conv_layer = model.get_layer("mixed10")
    model_grad = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
    with tf.GradientTape() as gtape:
        preds, conv_layer_output = model_grad(img)
        class_id = np.argmax(preds[0])
        class_output = preds[:, class_id]

    grads = gtape.gradient(class_output, conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(conv_layer_output * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #print(heatmap.shape)
    #heatmap = cv2.resize(heatmap, (img.shape[2], img.shape[1]))  # Resize heatmap to match input image size

    ax[1].imshow(heatmap[0], cmap='jet', alpha=0.5)  # Display heatmap with transparency
    ax[1].set_title("Heat map")
    
    act_img = cv2.imread(allo)
    heatmap_resized = cv2.resize(heatmap[0], (act_img.shape[1], act_img.shape[0]))
    heatmap_rescaled = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(act_img, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite('classactivation.png', superimposed)

    img_act = Image.open('classactivation.png')
    ax[2].imshow(img_act)
    #ax[2].imshow(img[0])
    ax[2].imshow(heatmap_resized, cmap='jet', alpha=0.5, interpolation='bilinear')
    ax[2].set_title("Class Activation")
    plt.show()
    return preds