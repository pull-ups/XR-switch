from ultralytics import YOLO
# ultralytics.checks()

model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)
results = model.train(data="./switch_data_v2/data.yaml", epochs=100, imgsz=640)



#sbatch -p suma_rtx4090 -q big_qos --gres=gpu:1 train.sh
