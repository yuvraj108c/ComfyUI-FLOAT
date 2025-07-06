import json
from types import SimpleNamespace

def dict_to_obj(d):
    return json.loads(json.dumps(d), object_hook=lambda d: SimpleNamespace(**d))

# use custom dictionary instead of original parser for arguments
BaseOptionsJson = dict_to_obj({
  "pretrained_dir": "./checkpoints",
  "seed": 15,
  "fix_noise_seed": False,
  "input_size": 512,
  "input_nc": 3,
  "fps": 25.0,
  "sampling_rate": 16000,
  "audio_marcing": 2,
  "wav2vec_sec": 2.0,
  "wav2vec_model_path": "./checkpoints/wav2vec2-base-960h",
  "audio2emotion_path": "./checkpoints/wav2vec-english-speech-emotion-recognition",
  "attention_window": 2,
  "only_last_features": False,
  "average_emotion": False,
  "audio_dropout_prob": 0.1,
  "ref_dropout_prob": 0.1,
  "emotion_dropout_prob": 0.1,
  "style_dim": 512,
  "dim_a": 512,
  "dim_w": 512,
  "dim_h": 1024,
  "dim_m": 20,
  "dim_e": 7,
  "fmt_depth": 8,
  "num_heads": 8,
  "mlp_ratio": 4.0,
  "no_learned_pe": False,
  "num_prev_frames": 10,
  "max_grad_norm": 1.0,
  "ode_atol": 1e-5,
  "ode_rtol": 1e-5,
  "nfe": 10,
  "torchdiffeq_ode_method": "euler",
  "a_cfg_scale": 2.0,
  "e_cfg_scale": 1.0,
  "r_cfg_scale": 1.0,
  "n_diff_steps": 500,
  "diff_schedule": "cosine",
  "diffusion_mode": "sample"
})