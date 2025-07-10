import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_x(value, width):
	"""Gets the x value from the image filename, assuming value is a float pixel coord."""
	return (value * 224.0 / 640.0 - width / 2) / (width / 2)

def get_y(value, height):
	"""Gets the y value from the image filename, assuming value is a float pixel coord."""
	return ((224 - value) - height / 2) / (height / 2)

class XYDataset(torch.utils.data.Dataset):
	
	def __init__(self, directory, random_hflips=False):
		self.directory = directory
		self.random_hflips = random_hflips
		self.image_paths = glob.glob(os.path.join(self.directory, 'image', '*.jpg'))
		self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		image_path = self.image_paths[idx]
		
		image = PIL.Image.open(image_path).convert("RGB")
		
		label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
		label_path = os.path.join(self.directory, 'label', label_filename)

		raw_value1, raw_value2 = 0.0, 0.0 # 保持为float类型
		has_midline_flag = 0

		if os.path.exists(label_path):
			with open(label_path, 'r') as label_file:
				content = label_file.read().strip()
				values = content.split()
				if len(values) == 3:
					try:
						# 尝试将值转换为浮点数，包括 "NaN" 字符串
						raw_value1 = float(values[0]) if values[0].lower() != "nan" else np.nan
						raw_value2 = float(values[1]) if values[1].lower() != "nan" else np.nan
						has_midline_flag = int(values[2])
					except ValueError:
						print(f"Warning: Could not parse label values in {label_path}. Expected numbers or 'NaN', got unexpected data. Using default 0,0,0.")
				else:
					print(f"Warning: Unexpected label format in {label_path}. Expected 3 values, got {len(values)}. Using default 0,0,0.")
		else:
			print(f"Warning: Label file not found for {image_path}. Assuming no midline (0,0,0).")

		# 只有当 has_midline_flag 为 1 时才计算有效的 x, y 坐标
		# 否则，x, y 设为占位符
		if has_midline_flag == 1:
			x = float(get_x(raw_value1, 224))
			y = float(get_y(raw_value2, 224))
		else:
			# 如果没有中线，X Y坐标可以是任意值，因为它们不会被用于损失计算
			x, y = 0.0, 0.0
	  
		if self.random_hflips and has_midline_flag == 1: # 只有有中线时才考虑翻转和坐标变更
			if float(np.random.rand(1)) > 0.5:
				image = transforms.functional.hflip(image)
				x = -x
		
		image = self.color_jitter(image)
		image = transforms.functional.resize(image, (224, 224))
		image = transforms.functional.to_tensor(image)
		
		image = transforms.functional.normalize(image,
												[0.485, 0.456, 0.406],
												[0.229, 0.224, 0.225])
		
		return image, torch.tensor([x, y]).float(), torch.tensor([float(has_midline_flag)]).float()

def main(train_dataset_path, test_dataset_path):
	train_dataset = XYDataset(train_dataset_path, random_hflips=False)
	test_dataset = XYDataset(test_dataset_path, random_hflips=False)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=32,
		shuffle=True,
		num_workers=4
	)

	test_loader = torch.utils.data.DataLoader(
		test_dataset,
		batch_size=32,
		shuffle=False,
		num_workers=4
	)

	model = models.resnet18(pretrained=True)
	model.fc = torch.nn.Linear(model.fc.in_features, 3)
	model = model.to(device)

	NUM_EPOCHS = 100
	BEST_MODEL_PATH = './best_line_follower_model_xy_conf.pt'
	best_loss = 1e9

	optimizer = optim.Adam(model.parameters())

	criterion_coords = nn.MSELoss()
	criterion_midline_conf = nn.BCEWithLogitsLoss()

	for epoch in range(NUM_EPOCHS):
		model.train()
		train_total_loss = 0.0
		train_coord_loss_epoch = 0.0
		train_conf_loss_epoch = 0.0
		
		for images, target_xy, target_has_midline in train_loader:
			images = images.to(device)
			target_xy = target_xy.to(device)
			target_has_midline = target_has_midline.to(device)

			optimizer.zero_grad()
			outputs = model(images)
			
			pred_xy = outputs[:, :2]
			pred_conf_logit = outputs[:, 2]

			loss_conf = criterion_midline_conf(pred_conf_logit, target_has_midline.squeeze(1))

			has_midline_mask = (target_has_midline.squeeze(1) == 1).bool()

			loss_coords = torch.tensor(0.0, device=device)
			if has_midline_mask.any():
				valid_pred_xy = pred_xy[has_midline_mask]
				valid_target_xy = target_xy[has_midline_mask]
				loss_coords = criterion_coords(valid_pred_xy, valid_target_xy)
			
			combined_loss = loss_coords + loss_conf # 保持损失权重为 1:1

			combined_loss.backward()
			optimizer.step()

			train_total_loss += combined_loss.item()
			train_coord_loss_epoch += loss_coords.item()
			train_conf_loss_epoch += loss_conf.item()

		avg_train_total_loss = train_total_loss / len(train_loader)
		avg_train_coord_loss = train_coord_loss_epoch / len(train_loader)
		avg_train_conf_loss = train_conf_loss_epoch / len(train_loader)
		
		model.eval()
		test_total_loss = 0.0
		test_coord_loss_epoch = 0.0
		test_conf_loss_epoch = 0.0

		with torch.no_grad():
			for images, target_xy, target_has_midline in test_loader:
				images = images.to(device)
				target_xy = target_xy.to(device)
				target_has_midline = target_has_midline.to(device)

				outputs = model(images)
				pred_xy = outputs[:, :2]
				pred_conf_logit = outputs[:, 2]

				loss_conf = criterion_midline_conf(pred_conf_logit, target_has_midline.squeeze(1))
				
				has_midline_mask = (target_has_midline.squeeze(1) == 1).bool()
				
				loss_coords = torch.tensor(0.0, device=device)
				if has_midline_mask.any():
					valid_pred_xy = pred_xy[has_midline_mask]
					valid_target_xy = target_xy[has_midline_mask]
					loss_coords = criterion_coords(valid_pred_xy, valid_target_xy)
				
				combined_loss = loss_coords + loss_conf

				test_total_loss += combined_loss.item()
				test_coord_loss_epoch += loss_coords.item()
				test_conf_loss_epoch += loss_conf.item()

		avg_test_total_loss = test_total_loss / len(test_loader)
		avg_test_coord_loss = test_coord_loss_epoch / len(test_loader)
		avg_test_conf_loss = test_conf_loss_epoch / len(test_loader)
		
		print(f"Epoch {epoch+1}/{NUM_EPOCHS}:")
		print(f"  Train: Total Loss: {avg_train_total_loss:.8f}, Coord Loss: {avg_train_coord_loss:.8f}, Conf Loss: {avg_train_conf_loss:.8f}")
		print(f"  Test:  Total Loss: {avg_test_total_loss:.8f}, Coord Loss: {avg_test_coord_loss:.8f}, Conf Loss: {avg_test_conf_loss:.8f}")

		if avg_test_total_loss < best_loss:
			print(f"Saving new best model with Test Total Loss: {avg_test_total_loss:.4f}")
			torch.save(model.state_dict(), BEST_MODEL_PATH)
			best_loss = avg_test_total_loss

if __name__ == '__main__':
	train_dataset_path = 'line_follow_dataset/train'
	test_dataset_path = 'line_follow_dataset/test'

	main(train_dataset_path, test_dataset_path)
