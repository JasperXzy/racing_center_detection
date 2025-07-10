import torchvision
import torch

def main(args=None):
  model = torchvision.models.resnet18(pretrained=False)
  model.fc = torch.nn.Linear(512, 3) 
  
  model_weights_path = './best_line_follower_model_xy_conf.pt' 
  
  model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
  
  device = torch.device('cpu') 
  model = model.to(device)
  
  model.eval() 

  x = torch.randn(1, 3, 224, 224, requires_grad=True).to(device) 

  onnx_output_path = "./best_line_follower_model_xy_conf.onnx"
  
  torch.onnx.export(model,
                    x,                          
                    onnx_output_path,           
                    export_params=True,         
                    opset_version=11,           
                    do_constant_folding=True,  
                    input_names=['input'],      
                    output_names=['output'],    
                   )

  print(f"ONNX model exported to {onnx_output_path}")

if __name__ == '__main__':
  main()
