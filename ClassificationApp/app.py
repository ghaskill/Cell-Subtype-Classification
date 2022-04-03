import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import model
#app
app = Flask(__name__)

PATH = 'WBC_model.pt'
subtype_index = ['Eisonophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
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
  return subtype_index[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    img_bytes = file.read()
    class_name = get_prediction(image_bytes=img_bytes)
    return jsonify({'class_name': class_name})

if __name__ == '__main__':
  app.run(host="localhost", port=8000, debug=True)
