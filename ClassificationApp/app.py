import os
import io
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, send_file, Response
import model

app = Flask(__name__)

# create upload folder
upload_folder = 'uploads/'
if not os.path.exists(upload_folder):
  os.mkdir(upload_folder)

# configure upload folder
app.config['UPLOAD_FOLDER'] = upload_folder

# configure allowed extensions
allowed_extensions = ['jpeg', 'png']

def check_file_extensions(filename):
  return filename.split('.')[-1] in allowed_extensions

# initialize model
PATH = 'WBC_model.pt'
subtype_index = ['Eosonophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
device = torch.device('cpu')
cnn_model = model.cnn_model
cnn_model.load_state_dict(torch.load(PATH, map_location=device))
cnn_model.eval()

def transform_image(image_bytes):
  transform = transforms.Compose([transforms.Resize((120,120)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                  (0.5, 0.5, 0.5), 
                                  (0.5, 0.5, 0.5))])
  image = Image.open(io.BytesIO(image_bytes))
  return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
  tensor = transform_image(image_bytes=image_bytes)
  outputs = cnn_model.forward(tensor)
  _, predicted = outputs.max(1)
  predicted_idx = predicted.item()
  return predicted_idx

@app.route('/')
def main():
  return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
  class_array = np.array([])
  if request.method == 'POST':
    files = request.files.getlist('files')
    for file in files:
      print(file.filename)
      if check_file_extensions(file.filename):
        img_bytes = file.read()
        class_name = get_prediction(image_bytes=img_bytes)
        class_array = np.append(class_array, class_name)
      else: 
        class_array = np.append(class_array, 4)
    class_count_dict = {
      "Eosonophil": np.count_nonzero(class_array == 0),
      "Lymphocyte": np.count_nonzero(class_array == 1),
      "Monocyte": np.count_nonzero(class_array == 2),
      "Neutrophil": np.count_nonzero(class_array == 3),
      "Error count": np.count_nonzero(class_array == 4)
    }
  return render_template('predict.html', class_count_dict=class_count_dict)

if __name__ == '__main__':
  app.run(host="localhost", port=8000, debug=True)
