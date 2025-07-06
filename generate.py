"""
	Inference Stage 2
"""

import os, torch, random, cv2, torchvision, subprocess, librosa, datetime, tempfile, face_alignment
import numpy as np
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Disable albumentations regretable behavior that checks internet
import albumentations as A
import albumentations.pytorch.transforms as A_pytorch

from tqdm import tqdm
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor

from .models.float.FLOAT import FLOAT
from .options.base_options import  BaseOptionsJson


class DataProcessor:
	def __init__(self, opt):
		self.opt = opt
		self.fps = opt.fps
		self.sampling_rate = opt.sampling_rate
		self.input_size = opt.input_size

		self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

		# wav2vec2 audio preprocessor
		self.wav2vec_preprocessor = Wav2Vec2FeatureExtractor.from_pretrained(opt.wav2vec_model_path, local_files_only=True)

		# image transform 
		self.transform = A.Compose([
				A.Resize(height=opt.input_size, width=opt.input_size, interpolation=cv2.INTER_AREA),
				A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
				A_pytorch.ToTensorV2(),
			])

	@torch.no_grad()
	def process_img(self, img:np.ndarray) -> np.ndarray:
		mult = 360. / img.shape[0]

		resized_img = cv2.resize(img, dsize=(0, 0), fx = mult, fy = mult, interpolation=cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)        
		bboxes = self.fa.face_detector.detect_from_image(resized_img)
		bboxes = [(int(x1 / mult), int(y1 / mult), int(x2 / mult), int(y2 / mult), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.95]
		bboxes = bboxes[0] # Just use first bbox

		bsy = int((bboxes[3] - bboxes[1]) / 2)
		bsx = int((bboxes[2] - bboxes[0]) / 2)
		my  = int((bboxes[1] + bboxes[3]) / 2)
		mx  = int((bboxes[0] + bboxes[2]) / 2)
		
		bs = int(max(bsy, bsx) * 1.6)
		img = cv2.copyMakeBorder(img, bs, bs, bs, bs, cv2.BORDER_CONSTANT, value=0)
		my, mx  = my + bs, mx + bs  	# BBox center y, bbox center x
		
		crop_img = img[my - bs:my + bs,mx - bs:mx + bs]
		crop_img = cv2.resize(crop_img, dsize = (self.input_size, self.input_size), interpolation = cv2.INTER_AREA if mult < 1. else cv2.INTER_CUBIC)
		return crop_img

	def default_img_loader(self, path) -> np.ndarray:
		img = cv2.imread(path)
		return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	def default_aud_loader(self, path: str) -> torch.Tensor:
		speech_array, sampling_rate = librosa.load(path, sr = self.sampling_rate)
		return self.wav2vec_preprocessor(speech_array, sampling_rate = sampling_rate, return_tensors = 'pt').input_values[0]


	def preprocess(self, ref_path:str, audio_path:str, no_crop:bool) -> dict:
		s = self.default_img_loader(ref_path)
		if not no_crop:
			s = self.process_img(s)
		s = self.transform(image=s)['image'].unsqueeze(0)
		a = self.default_aud_loader(audio_path).unsqueeze(0)
		return {'s': s, 'a': a, 'p': None, 'e': None}


class InferenceAgent:
	def __init__(self, opt):
		torch.cuda.empty_cache()
		self.opt = opt
		self.rank = opt.rank
		
		# Load Model
		self.load_model()
		self.load_weight(opt.ckpt_path, rank=self.rank)
		self.G.to(self.rank)
		self.G.eval()

		# Load Data Processor
		self.data_processor = DataProcessor(opt)

	def load_model(self) -> None:
		self.G = FLOAT(self.opt)

	def load_weight(self, checkpoint_path: str, rank: int) -> None:
		state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
		with torch.no_grad():
			for model_name, model_param in self.G.named_parameters():
				if model_name in state_dict:
					model_param.copy_(state_dict[model_name].to(rank))
				elif "wav2vec2" in model_name: pass
				else:
					print(f"! Warning; {model_name} not found in state_dict.")

		del state_dict

	def save_video(self, vid_target_recon: torch.Tensor, video_path: str, audio_path: str) -> str:
		with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
			temp_filename = temp_video.name
			vid = vid_target_recon.permute(0, 2, 3, 1)
			vid = vid.detach().clamp(-1, 1).cpu()
			vid = ((vid + 1) / 2 * 255).type('torch.ByteTensor')
			torchvision.io.write_video(temp_filename, vid, fps=self.opt.fps)			
			if audio_path is not None:
				with open(os.devnull, 'wb') as f:
					command =  "ffmpeg -i {} -i {} -c:v copy -c:a aac {} -y".format(temp_filename, audio_path, video_path)
					subprocess.call(command, shell=True, stdout=f, stderr=f)
				if os.path.exists(video_path):
					os.remove(temp_filename)
			else:
				os.rename(temp_filename, video_path)
			return video_path

	@torch.no_grad()
	def run_inference(
		self,
		res_video_path: str,
		ref_path: str,
		audio_path: str,
		a_cfg_scale: float	= 2.0,
		r_cfg_scale: float	= 1.0,
		e_cfg_scale: float	= 1.0,
		emo: str 			= 'S2E',
		nfe: int			= 10,
		no_crop: bool 		= False,
		seed: int			= 25,
		verbose: bool 		= False
	) -> str:
		data = self.data_processor.preprocess(ref_path, audio_path, no_crop = no_crop)
		# print(f"> [Done] Preprocess.")

		# inference
		d_hat = self.G.inference(
			data 		= data,
			a_cfg_scale = a_cfg_scale,
			r_cfg_scale = r_cfg_scale,
			e_cfg_scale = e_cfg_scale,
			emo 		= emo,
			nfe			= nfe,
			seed		= seed
			)['d_hat']


		# res_video_path = self.save_video(d_hat, res_video_path, audio_path)
		# if verbose: print(f"> [Done] result saved at {res_video_path}")

		images_bhwc = d_hat.squeeze(0).permute(0, 2, 3, 1)
		images_bhwc = images_bhwc.detach().clamp(-1, 1).cpu()
		images_bhwc = ((images_bhwc + 1) / 2) 
		return images_bhwc