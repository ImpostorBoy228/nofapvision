# inference_camera.py
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from model import get_model

def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    # Для совместимости делаем модель с 2 классами (если нужно, можно восстановить num_classes из ckpt)
    model = get_model(num_classes=len(ckpt.get('class_to_idx', {0:0})), pretrained=False, device=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    class_to_idx = ckpt.get('class_to_idx', { 'fap':0, 'nofap':1 })
    # invert mapping
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    return model, idx_to_class

def preprocess_frame(frame, img_size=224, device='cpu'):
    from PIL import Image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    tensor = transform(pil).unsqueeze(0).to(device)
    return tensor

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--cam', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, idx_to_class = load_model(args.model, device)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Не могу открыть камеру")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inp = preprocess_frame(frame, img_size=args.img_size, device=device)
        with torch.no_grad():
            out = model(inp)
            probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0]
            pred = int(out.argmax(dim=1).cpu().numpy()[0])
            label = idx_to_class.get(pred, str(pred))
            score = probs[pred]
        # overlay
        text = f"{label} {score*100:.1f}%"
        cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow('Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
