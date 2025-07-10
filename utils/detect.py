import argparse
import torchvision
import torch
import PIL.Image
from PIL import ImageDraw, ImageFont
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def detect(weights, source, output, confidence_threshold=0.5, fps=30):
  model = torchvision.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(model.fc.in_features, 3) 
  
  model.load_state_dict(torch.load(weights, map_location=device))
  
  model.eval() 
  model = model.to(device)

  font = None
  try:
      font = ImageFont.truetype("arial.ttf", 20)
  except IOError:
      try:
          font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
      except IOError:
          print("Warning: Could not load font. Using default PIL font.")
          font = ImageFont.load_default()

  # 模型输入尺寸 (224x224)
  MODEL_INPUT_SIZE = (224, 224)
  # 中间处理尺寸
  PROCESS_SIZE = (640, 480)
  # 截取下半部分的起始Y坐标和高度
  CROP_HEIGHT = 224
  CROP_START_Y = PROCESS_SIZE[1] - CROP_HEIGHT # 480 - 224 = 256

  # transform 是针对 640x224 的裁剪区域
  transform_for_inference = transforms.Compose([
      transforms.Resize(MODEL_INPUT_SIZE), # 缩放到模型要求的 224x224
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  def process_frame(frame_pil_orig, draw_on_image_pil):
      orig_width, orig_height = frame_pil_orig.size

      # 调整大小到 PROCESS_SIZE大小 (640x480)
      # 使用 PIL.Image.LANCZOS 进行高质量缩放
      resized_img_pil = frame_pil_orig.resize(PROCESS_SIZE, PIL.Image.LANCZOS)

      # 裁剪底部224 640x224
      # crop(left, upper, right, lower)
      cropped_img_pil = resized_img_pil.crop((0, CROP_START_Y, PROCESS_SIZE[0], PROCESS_SIZE[1]))

      # 对裁剪后的图像进行推理变换
      image_tensor = transform_for_inference(cropped_img_pil).unsqueeze(dim=0).to(device)

      with torch.no_grad():
          outputs = model(image_tensor)
      
      pred_xy = outputs[0, :2].cpu().numpy()
      pred_conf_logit = outputs[0, 2].item()
      confidence = torch.sigmoid(torch.tensor(pred_conf_logit)).item()
      
      # 坐标去规范化并转换回原始图像坐标
      x_norm = pred_xy[0]
      y_norm = pred_xy[1] # 这里的y_norm是相对于 640x224 裁剪区域的 y 轴归一化坐标

      # 将模型预测的 (x_norm, y_norm) 从 [-1, 1] 映射回 640x224 裁剪区域内的像素坐标
      # 模型输入是 224x224，但训练时XYDataset中get_x/get_y的width是224，x轴缩放了640/224
      # x_pixel_in_cropped_224 = (x_norm * (224/2) + 224/2) 
      # y_pixel_in_cropped_224 = 224 - (y_norm * (224/2) + 224/2) # 原始 get_y 翻转了Y轴
      
      # 映射回 640x224 区域的像素坐标
      # x 轴：(归一化x * (模型输入width/2) + 模型输入width/2) * (裁剪区域width / 模型输入width)
      # 注意：模型输入是224宽度，但get_x中value映射到了640宽度
      # 所以这里推导的x是关于640的，但是被输入到模型时被缩放到了224
      # 反推时，这里的 x_pixel_in_cropped_640 是相对于 640x224 裁剪区域的 x 像素
      x_pixel_in_cropped_640 = (x_norm * (224/2) + 224/2) * (PROCESS_SIZE[0] / MODEL_INPUT_SIZE[0]) # (640/224)
      # y 轴：是相对于 224 裁剪区域的高度
      y_pixel_in_cropped_224_inverted = (y_norm * (MODEL_INPUT_SIZE[1]/2) + MODEL_INPUT_SIZE[1]/2) # Y归一化转为[0,224]范围的，但还是翻转的
      y_pixel_in_cropped_224 = MODEL_INPUT_SIZE[1] - y_pixel_in_cropped_224_inverted # 再次反转Y轴到正常Y轴

      # 将坐标从 640x224 裁剪区域转换回 640x480 的中间处理区域
      # x_pixel_in_process_size = x_pixel_in_cropped_640
      # y_pixel_in_process_size = y_pixel_in_cropped_224 + CROP_START_Y
      
      # 因为裁剪区域的宽度是 640，所以 x 坐标可以直接用
      x_on_original_resized = int(x_pixel_in_cropped_640)
      # y 坐标需要加上裁剪前的偏移量 CROP_START_Y (480-224 = 256)
      y_on_original_resized = int(y_pixel_in_cropped_224 + CROP_START_Y)

      # 如果原始图像尺寸与 PROCESS_SIZE (640x480) 不同，还需要进行一次缩放
      # 将坐标从 PROCESS_SIZE (640x480) 映射回原始图像尺寸
      scale_x = orig_width / PROCESS_SIZE[0]
      scale_y = orig_height / PROCESS_SIZE[1]
      
      final_x = int(x_on_original_resized * scale_x)
      final_y = int(y_on_original_resized * scale_y)

      text_to_display = f"Conf: {confidence:.2f}"
      draw_on_image_pil.text((10, 10), text_to_display, fill=(0, 255, 0), font=font) 

      if confidence >= confidence_threshold:
          draw_size = 5 
          for i in range(max(0, final_x - draw_size), min(orig_width, final_x + draw_size + 1)):
              for j in range(max(0, final_y - draw_size), min(orig_height, final_y + draw_size + 1)):
                  draw_on_image_pil.point((i, j), (255, 0, 0)) # Red
          print(f"  Predicted (x, y) on original frame: ({final_x}, {final_y}), Confidence: {confidence:.4f}")
      else:
          print(f"  Confidence {confidence:.4f} < threshold {confidence_threshold:.2f}. Not drawing midline.")
      
      return final_x, final_y, confidence

  is_video = source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

  if not is_video: 
      print(f"Processing single image: {source}")
      image_raw = PIL.Image.open(source).convert("RGB")
      imagedraw = ImageDraw.Draw(image_raw)
      
      process_frame(image_raw, imagedraw) 
      
      image_raw.save(output)

  else: 
      print(f"Processing video: {source}")
      cap = cv2.VideoCapture(source)
      if not cap.isOpened():
          print(f"Error: Could not open video file {source}")
          return

      frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      
      fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
      out = cv2.VideoWriter(output, fourcc, fps, (frame_width, frame_height))

      frame_count = 0
      while True:
          ret, frame = cap.read()
          if not ret:
              break 

          frame_count += 1
          print(f"\rProcessing frame {frame_count}...", end="")

          img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image_pil = PIL.Image.fromarray(img_rgb)
          imagedraw = ImageDraw.Draw(image_pil)

          process_frame(image_pil, imagedraw)
          
          final_frame_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
          out.write(final_frame_bgr)

      print(f"\nFinished processing {frame_count} frames. Output video saved to {output}")
      cap.release()
      out.release()
      cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./best_line_follower_model_xy_conf.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='image.jpg', help='source file (image or video)')
    parser.add_argument('--output', type=str, default='output_media', help='output file name (e.g., output.jpg or output.mp4)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Minimum confidence to draw midline')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for output video (ignored for images)')
    args = parser.parse_args()

    if args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and not (args.output.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))):
        args.output = os.path.splitext(args.output)[0] + ".mp4"
    elif not args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')) and not (args.output.lower().endswith(('.jpg', '.png', '.jpeg'))):
        args.output = os.path.splitext(args.output)[0] + ".jpg"

    with torch.no_grad():
        detect(args.weights, args.source, args.output, args.confidence_threshold, args.fps)
