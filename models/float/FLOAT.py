import torch, math
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint
from transformers import Wav2Vec2Config
from transformers.modeling_outputs import BaseModelOutput

from ...models.wav2vec2 import Wav2VecModel
from ...models.wav2vec2_ser import Wav2Vec2ForSpeechClassification

from ...models.basemodel import BaseModel
from .generator import Generator
from .FMT import FlowMatchingTransformer
import time

from comfy.utils import ProgressBar

######## Main Phase 2 model ########		
class FLOAT(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt

		self.num_frames_for_clip = int(self.opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(self.opt.num_prev_frames)

		# motion latent auto-encoder
		self.motion_autoencoder = Generator(size = opt.input_size, style_dim = opt.dim_w, motion_dim = opt.dim_m)
		self.motion_autoencoder.requires_grad_(False)

		# condition encoders
		self.audio_encoder 		= AudioEncoder(opt)
		self.emotion_encoder	= Audio2Emotion(opt)

		# FMT; Flow Matching Transformer
		self.fmt = FlowMatchingTransformer(opt)
		
		# ODE options
		self.odeint_kwargs = {
			'atol': self.opt.ode_atol,
			'rtol': self.opt.ode_rtol,
			'method': self.opt.torchdiffeq_ode_method
		}
	
	######## Motion Encoder - Decoder ########
	@torch.no_grad()
	def encode_image_into_latent(self, x: torch.Tensor) -> list:
		x_r, _, x_r_feats = self.motion_autoencoder.enc(x, input_target=None)
		x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
		return x_r, x_r_lambda, x_r_feats

	@torch.no_grad()
	def encode_identity_into_motion(self, x_r: torch.Tensor) -> torch.Tensor:
		x_r_lambda = self.motion_autoencoder.enc.fc(x_r)
		r_x = self.motion_autoencoder.dec.direction(x_r_lambda)
		return r_x

	@torch.no_grad()
	def decode_latent_into_image(self, s_r: torch.Tensor , s_r_feats: list, r_d: torch.Tensor) -> dict:
		# print("starting decoding")
		# start = time.time()
		
		T = r_d.shape[1] # r_d is on GPU
		pbar = ProgressBar(T) # only for decoding latents, encoding + inference is pretty fast
		d_hat_list_cpu = [] # List to accumulate frames on CPU
		for t in range(T):
			s_r_d_t = s_r + r_d[:, t] # Operation on GPU
			img_t_gpu, _ = self.motion_autoencoder.dec(s_r_d_t, alpha = None, feats = s_r_feats) # img_t_gpu is on GPU
			d_hat_list_cpu.append(img_t_gpu.cpu()) # Move the generated frame to CPU and add it to the list
			
			# Free intermediate GPU tensors as soon as possible within the loop
			del s_r_d_t, img_t_gpu
			if torch.cuda.is_available() and (t % 10 == 0 or t == T -1) : # Empty cache periodically or at the end
				torch.cuda.empty_cache()
				
			pbar.update(1)

		# s_r, r_d, s_r_feats are arguments passed to this function and are on GPU.
		# Their cleanup will happen in the calling method (inference) after this function returns.

		d_hat_stacked_cpu = torch.stack(d_hat_list_cpu, dim=1).squeeze() # Stacking happens on CPU
		# end = time.time()
		
		# print(end-start, "decoding done")
		# The result 'd_hat' is now a CPU tensor.
		return {'d_hat': d_hat_stacked_cpu}


	######## Motion Sampling and Inference ########
	@torch.no_grad()
	def sample(
		self,
		data: dict,
		a_cfg_scale: float = 1.0,
		r_cfg_scale: float = 1.0,
		e_cfg_scale: float = 1.0,
		emo: str = None,
		nfe: int = 10,
		seed_arg: int = None # Renamed seed to avoid conflict with self.opt.seed
	) -> torch.Tensor:

		r_s, a_cpu = data['r_s'], data['a'] # r_s is on GPU, a_cpu is on CPU
		B = a_cpu.shape[0]

		# make time 
		time_steps = torch.linspace(0, 1, nfe, device=self.opt.rank) # Use nfe from arguments
		
		# encoding audio first with whole audio
		a_gpu = a_cpu.to(self.opt.rank)
		T = math.ceil(a_gpu.shape[-1] * self.opt.fps / self.opt.sampling_rate)
		wa = self.audio_encoder.inference(a_gpu, seq_len=T) # wa is on GPU

		# encoding emotion first
		emo_idx = self.emotion_encoder.label2id.get(str(emo).lower(), None)
		if emo_idx is None:
			we = self.emotion_encoder.predict_emotion(a_gpu).unsqueeze(1) # we is on GPU
		else:
			we = F.one_hot(torch.tensor(emo_idx, device = a_gpu.device), num_classes = self.opt.dim_e).unsqueeze(0).unsqueeze(0) # we is on GPU

		del a_gpu # Free VRAM from a_gpu
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		# r_s is already on GPU
		# wa is on GPU
		# we is on GPU

		sample_list = []
		# sampleing chunk by chunk
		for t_loop_idx in range(0, int(math.ceil(T / self.num_frames_for_clip))):
			if self.opt.fix_noise_seed:
				current_seed = self.opt.seed if seed_arg is None else seed_arg # Use seed_arg from arguments
				g = torch.Generator(self.opt.rank)
				g.manual_seed(current_seed)
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank, generator = g)
			else:
				x0 = torch.randn(B, self.num_frames_for_clip, self.opt.dim_w, device = self.opt.rank)

			if t_loop_idx == 0: # should define the previous
				prev_x_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
				prev_wa_t = torch.zeros(B, self.num_prev_frames, self.opt.dim_w).to(self.opt.rank)
			else:
				prev_x_t = sample_t[:, -self.num_prev_frames:]
				prev_wa_t = wa_t_chunk[:, -self.num_prev_frames:] # Use wa_t_chunk from previous iteration
			
			wa_t_chunk = wa[:, t_loop_idx * self.num_frames_for_clip: (t_loop_idx+1)*self.num_frames_for_clip]

			if wa_t_chunk.shape[1] < self.num_frames_for_clip: # padding by replicate
				wa_t_chunk = F.pad(wa_t_chunk, (0, 0, 0, self.num_frames_for_clip - wa_t_chunk.shape[1]), mode='replicate')

			def ode_fn(tt, zt):
				out = self.fmt.forward_with_cfv(
						t 			= tt.unsqueeze(0),
						x 			= zt,
						wa 			= wa_t_chunk,
						wr 			= r_s,
						we 			= we, 
						prev_x 		= prev_x_t, 	
						prev_wa 	= prev_wa_t,
						a_cfg_scale = a_cfg_scale,
						r_cfg_scale = r_cfg_scale,
						e_cfg_scale = e_cfg_scale
						)

				out_current = out[:, self.num_prev_frames:]
				return out_current

			# solve ODE
			trajectory_t = odeint(ode_fn, x0, time_steps, **self.odeint_kwargs) # Use time_steps
			sample_t = trajectory_t[-1]
			# trajectory_t can be large, delete it if only the last step is needed
			del trajectory_t
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

			sample_list.append(sample_t)
		final_sample = torch.cat(sample_list, dim=1)[:, :T]
		return final_sample

	@torch.no_grad()
	def inference(
		self,
		data: dict,
		a_cfg_scale = None,
		r_cfg_scale = None,
		e_cfg_scale = None,
		emo			= None,
		nfe			= 10,
		seed		= None,
	) -> dict:

		s_cpu, a_cpu = data['s'], data['a'] # s_cpu and a_cpu are on CPU from DataProcessor

		s_img_gpu = s_cpu.to(self.opt.rank) # Source image on GPU
		# print("starting encoding")
		# start = time.time()
		# s_r, s_r_feats are for the final image decoding and come from s_img_gpu
		# r_s_lambda_from_s_img is the motion lambda calculated from s_img_gpu
		s_r, r_s_lambda_from_s_img, s_r_feats = self.encode_image_into_latent(s_img_gpu)
		# end = time.time()
		# print(end-start, "encoding end")
		del s_img_gpu # Free VRAM from the source GPU image
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		r_s_style_source_gpu = None # To track if we create a temporary tensor for the style
		# Determine r_s based on whether s_r is already provided in data
		# s_r would be on GPU if pre-calculated, r_s_lambda is already on GPU
		if 's_r' in data and data['s_r'] is not None:
			# The motion style r_s comes from an external reference data['s_r']
			r_s_style_source_gpu = data['s_r'].to(self.opt.rank)
			r_s = self.encode_identity_into_motion(r_s_style_source_gpu)
			# r_s_lambda_from_s_img is not used to calculate r_s in this branch
		else:
			# The motion style r_s comes from the source image s_cpu (via r_s_lambda_from_s_img)
			r_s = self.motion_autoencoder.dec.direction(r_s_lambda_from_s_img)
		
		# r_s is now defined and on GPU.

		# Prepare data for the sample method.
		# 'a' (audio) will be handled by the sample method (moved to GPU there).
		# 'r_s' (reference style) is on GPU.
		# We update/add 'r_s' to the data dictionary that will be passed to self.sample.
		# Other elements from the original `data` dict are passed along.
		data_for_sample = data.copy() # Avoid modifying the input dict directly if it's used elsewhere, though original code modifies it.
		data_for_sample['r_s'] = r_s 
		data_for_sample['a'] = a_cpu # Ensure 'a' is the CPU tensor for sample method


		# set conditions
		if a_cfg_scale is None: a_cfg_scale = self.opt.a_cfg_scale
		if r_cfg_scale is None: r_cfg_scale = self.opt.r_cfg_scale
		if e_cfg_scale is None: e_cfg_scale = self.opt.e_cfg_scale
		
		# print("starting actual inference")
		# start = time.time()
		sample_result = self.sample(
			data_for_sample, # Pass the prepared dictionary
			a_cfg_scale = a_cfg_scale, 
			r_cfg_scale = r_cfg_scale, 
			e_cfg_scale = e_cfg_scale, 
			emo = emo, 
			nfe = nfe, 
			seed_arg = seed # Pass seed as seed_arg
		)
		# end = time.time()
		# print(end-start, "actual inference")
		# s_r, s_r_feats (from the source image s_cpu) are used for decoding. sample_result is on GPU.
		data_out_dict = self.decode_latent_into_image(s_r = s_r, s_r_feats = s_r_feats, r_d = sample_result)
        # data_out_dict['d_hat'] is now on CPU thanks to changes in decode_latent_into_image

		# Final cleanup of GPU tensors no longer needed
		del s_r, s_r_feats, sample_result, r_s
		if r_s_style_source_gpu is not None:
			del r_s_style_source_gpu
		# r_s_lambda_from_s_img was used (or not used) to calculate r_s. It can be deleted.
		del r_s_lambda_from_s_img
            
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		return data_out_dict # {'d_hat': tensor_on_CPU}




################ Condition Encoders ################
class AudioEncoder(BaseModel):
	def __init__(self, opt):
		super().__init__()
		self.opt = opt
		self.only_last_features = opt.only_last_features
		
		self.num_frames_for_clip = int(opt.wav2vec_sec * self.opt.fps)
		self.num_prev_frames = int(opt.num_prev_frames)

		self.wav2vec2 = Wav2VecModel.from_pretrained(opt.wav2vec_model_path, local_files_only = True)
		self.wav2vec2.feature_extractor._freeze_parameters()

		for name, param in self.wav2vec2.named_parameters():
			param.requires_grad = False

		audio_input_dim = 768 if opt.only_last_features else 12 * 768

		self.audio_projection = nn.Sequential(
			nn.Linear(audio_input_dim, opt.dim_w),
			nn.LayerNorm(opt.dim_w),
			nn.SiLU()
			)

	def get_wav2vec2_feature(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		a = self.wav2vec2(a, seq_len=seq_len, output_hidden_states = not self.only_last_features)
		if self.only_last_features:
			a = a.last_hidden_state
		else:
			a = torch.stack(a.hidden_states[1:], dim=1).permute(0, 2, 1, 3)
			a = a.reshape(a.shape[0], a.shape[1], -1)
		return a

	def forward(self, a:torch.Tensor, prev_a:torch.Tensor = None) -> torch.Tensor:
		if prev_a is not None:
			a = torch.cat([prev_a, a], dim = 1)
			if a.shape[1] % int( (self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int((self.num_frames_for_clip + self.num_prev_frames) * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode='replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip + self.num_prev_frames)
		else:
			if a.shape[1] % int( self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) != 0:
				a = F.pad(a, (0, int(self.num_frames_for_clip * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
			a = self.get_wav2vec2_feature(a, seq_len = self.num_frames_for_clip)
	
		return self.audio_projection(a) # frame by frame

	@torch.no_grad()
	def inference(self, a: torch.Tensor, seq_len:int) -> torch.Tensor:
		if a.shape[1] % int(seq_len * self.opt.sampling_rate / self.opt.fps) != 0:
			a = F.pad(a, (0, int(seq_len * self.opt.sampling_rate / self.opt.fps) - a.shape[1]), mode = 'replicate')
		a = self.get_wav2vec2_feature(a, seq_len=seq_len)
		return self.audio_projection(a)



class Audio2Emotion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.wav2vec2_for_emotion = Wav2Vec2ForSpeechClassification.from_pretrained(opt.audio2emotion_path, local_files_only=True)
        self.wav2vec2_for_emotion.eval()
        
		# seven labels
        self.id2label = {0: "angry", 1: "disgust", 2: "fear", 3: "happy",
						4: "neutral", 5: "sad", 6: "surprise"}

        self.label2id = {v: k for k, v in self.id2label.items()}

    @torch.no_grad()
    def predict_emotion(self, a: torch.Tensor, prev_a: torch.Tensor = None) -> torch.Tensor:
        if prev_a is not None:
            a = torch.cat([prev_a, a], dim=1)
        logits = self.wav2vec2_for_emotion.forward(a).logits
        return F.softmax(logits, dim=1) 	# scores

#######################################################
