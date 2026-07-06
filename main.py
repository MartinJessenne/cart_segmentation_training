from ultralytics import YOLO

def main():
    model = YOLO("yolo26n-seg.pt")  # pretrained checkpoint, auto-downloads on first use
    
    result = model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=800,       # closer to native than 640, still much cheaper than 1280
    rect=True,       # exploit your fixed 1280x800 aspect ratio, less wasted padding
    batch=-1,
    device=0,
    patience=20,
)

if __name__ == "__main__":
    main()
