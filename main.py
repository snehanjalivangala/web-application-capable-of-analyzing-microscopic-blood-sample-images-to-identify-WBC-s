# pip install flask
# pip install torch
# pip install opencv-python

import os

import torch

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

print("Starting..")
from flask import Flask,render_template,request,redirect,send_file,url_for,flash
import os
from werkzeug.utils import secure_filename

import cv2

IMAGE_SIZE = (224, 224, 3)


TRAINED_YOLO = "best.pt" #Pretraine Model Filename
model_image_size = 640 #Image Size with which Model was Trained


model,imgsz,device = None, None, None

def yolo_detect(source, conf_thres=0.30, iou_thres=0.45, lite = False):
    global model,imgsz,device, model_image_size

    if model is None:# If model is not Loaded
        ###################### Load Model for Detection  ############################# 
        device = select_device(" ")
        model = DetectMultiBackend(TRAINED_YOLO, device=device)
        imgsz = check_img_size(model_image_size, s=model.stride)  # check image size

        # print("Names:", model.names,)
        print("Device:", device.type)

        if model.pt and device.type != 'cpu' :
            print("Activated Half Precision for GPU")
            model.model.half()# half precision only supported by PyTorch on CUDA
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))# warmup     
        else:
            model.model.float()

    loaded_image = LoadImages(source, img_size=imgsz, stride=model.stride)

    for path, im, raw, vid_cap, _ in loaded_image:
        im = torch.from_numpy(im).to(device)
        if device.type != 'cpu':
            im = im.half()
        else:
            im = im.float()  # uint8 to float
        
        im /= 255  # 0.0 - 255.0 to 0.0 - 1.0 #Normalise Image
        if len(im.shape) == 3:
            im = im[None] # [R][G][B] -> Grey scale format

        model_preds = model(im) # Predictions from Model
        final_preds = non_max_suppression(model_preds, conf_thres, iou_thres,max_det=100)

        coords = list()
        class_list = list()

        # Process predictions
        for detection in final_preds:  # per image
            annotator = Annotator(raw, line_width=3, example=str(model.names))
            if len(detection):
                detection[:, :4] = scale_coords(im.shape[2:], detection[:, :4], raw.shape).round()# Rescale boxesto raw size
              
                for *xyxy, conf, cls in reversed(detection):
                    c = int(cls)  # integer class
                    label = f'{model.names[c]} {conf:.2f}' #f'{names[c]}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    xyxy.append(conf)
                    coords.append(xyxy)
                    class_list.append(model.names[c])
            annotated_image = annotator.result()

    if lite: #Only return Annotated Output
        return annotated_image
    
    #Return Annoated with Prediction Details
    return annotated_image, coords, class_list
            


app=Flask(__name__)
app.secret_key="secure"
app.config['UPLOAD_FOLDER'] = './static/uploads/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',methods=["post","get"])
def first_page():
    if request.method=="POST":
        global image_name,image_data

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # op = predict('static/uploads/'+filename)
            image,cd,cl = yolo_detect(source = app.config['UPLOAD_FOLDER']+filename)

            wbc = []
            rbc = []
            platelets = []
            for i in cl:
                if i == "WBC":
                    wbc.append(i)
                if i == "RBC":
                    rbc.append(i)
                if i == "Platelets":
                    platelets.append(i)

            solution = f"Estimated WBC count: {len(wbc)} \n"
            solution += f"Estimated RBC count: {len(rbc)} \n"
            solution += f"Estimated Platelets count: {len(platelets)} \n"

            cv2.imwrite(app.config['UPLOAD_FOLDER']+"op.jpg", image)

            return render_template("data_page.html",
                           filename='op.jpg', result = "Analysis Report", solution = solution.split("\n"))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)

    else:
        return render_template("form_page.html")


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)



                

if __name__ == '__main__':

    app.run(debug=True, host="0.0.0.0")
