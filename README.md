# PPE Detection
Detect Personal Protective Equipment (PPE) on industrial workers using YOLOv8. The model identifies and localizes 9 classes mentioned below. The dataset used is a custom sourced dataset which was used for training the YOLOv8 model using Roboflow api. The link for the steps to train a custom YOLOv8 model using roboflow: https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/

## Classes
0. withHelmet
1. withoutHelmet
2. withGoggle
3. withoutGoggle
4. withGlove
5. withoutGlove
6. withoutShoe
7. withShoe
8. Mobile

## Install the required dependencies
```python
pip3 install requirements.txt
```

## Custom plotting function
``` python
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
```
## Option to save the inferenced video 
``` python
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)   
size = (frame_width, frame_height)
video = cv2.VideoWriter('output1.mp4', 
                         cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
```


