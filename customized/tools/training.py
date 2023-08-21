from base64 import decode
import os
import subprocess
import sys

sys.path.append(os.getcwd())
import yaml
from easydict import EasyDict as easydict

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'

def bash_command(cmd):
     result = subprocess.Popen(['/bin/bash', '-c', cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
     #while result.poll() is None:
     #     line = result.stdout.readline()
     #     line = line.strip()
     #     if line:
     #          print(line.decode("utf-8",'ignore'))
     std_output, std_error = result.communicate()
     return result.returncode, std_output.decode("utf-8"), std_error.decode("utf-8")

def convert_onnx(model, image_size):
     weights_path = '/home/ubuntu/Mask-Det_Damo_Yolo/customized/results/weights/{}/{}'.format(model, 'latest_ckpt.pth')
     output_folder = '/home/ubuntu/Mask-Det_Damo_Yolo/customized/results/weights/onnx'

     convert_command = 'python3 /home/ubuntu/Mask-Det_Damo_Yolo/customized/models/damo_yolo/convert.py \
                         -f /home/ubuntu/AIDMS_Model_Damo_Yolo/customized/damo-yolo/configs/{}.py \
                         -c {} --batch_size 1 --img_size {}'.format(model, weights_path, image_size)
     
     return_code, stdout, stderr = bash_command(convert_command)
     if return_code:
          print(stderr)
     else:
          print(stdout)


if __name__ == '__main__':
     ''' 
     read yaml sample code
     model : selection backbone
     batch : setting yaml batch
     iterations_for_each_class : setting yaml batch
     learning_rate : setting learning rate 
     or other :
     '''
     with open('/home/ubuntu/Mask-Det_Damo_Yolo/customized/hyperparameters/parameters.yaml', 'r') as para:
          hyperparameters = yaml.load(para, Loader=yaml.FullLoader)
          hyperparameters = hyperparameters['hyperparameters']['training']
          hyperparameters = easydict(hyperparameters)

     model = hyperparameters.default_para.model.value[0].select_key
     batch = hyperparameters.default_para.batch.value
     epochs = hyperparameters.default_para.epochs.value
     height_img = hyperparameters.advanced_para.image_size.Height.value
     width_img = hyperparameters.advanced_para.image_size.Width.value
     image_size = height_img if height_img > width_img else width_img
     learning_rate = hyperparameters.advanced_para.learning_rate.value

     print('Select model name : {}'.format(model))

     return_code, stdout, stderr = bash_command(f'python3 /home/ubuntu/Mask-Det_Damo_Yolo/customized/models/voc2coco.py')
     if return_code:
          print(stderr)
     else:
          print(stdout)


     if model in ['damoyolo_tinynasL20_T','damoyolo_tinynasL18_Nm','damoyolo_tinynasL18_Ns','damoyolo_tinynasL20_N','damoyolo_tinynasL20_Nl',
                  'damoyolo_tinynasL25_S','damoyolo_tinynasL35_M','damoyolo_tinynasL45_L']:
          training_command = 'python3 -m torch.distributed.launch --nproc_per_node=1 \
               /home/ubuntu/Mask-Det_Damo_Yolo/customized/damo-yolo/train.py \
               -f /home/ubuntu/Mask-Det_Damo_Yolo/customized/models/damo-yolo/configs/{}.py'.format(model)
          

     return_code, std_out, std_err = bash_command(training_command)

     if return_code:
          print(stderr)
          print('Failed to train model in training.')
          sys.exit(1)

     else:
          print(stdout)
     print('Success to train model.')
     
     convert_onnx(model, image_size)

     print('Success convert onnx model.')
