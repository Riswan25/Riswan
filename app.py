from flask import Flask, render_template, request

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

app = Flask(__name__)
model = load_model('batukertasgunting.hdf5')

@app.route('/')
def website():
    return render_template('gunting_kertas_batu.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)
    
    image = load_img(image_path, target_size=(150,150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = np.vstack([image])
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocess_input(image)
    classes = model.predict(image, batch_size=10)
    #label = decode_predictions(classes)
    #label = label[0][0]

    if classes[0, 0] == 1:
        hasil = 'Tangan Anda Berbentuk Kertas'
    elif classes[0, 1] == 1:
        hasil = 'Tangan Anda Berbentuk Batu'
    elif classes[0, 2] == 1:
        hasil = 'Tangan Anda Berbentuk Gunting'
    else:
        hasil = 'Gambar Tidak Jelas'

    #classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('gunting_kertas_batu.html', prediction=hasil)

if __name__ == '__main__':
    app.run(port=3000, debug=True)