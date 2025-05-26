import folder_paths
import os
import comfy.model_management as mm
import time
import torchaudio
import torchvision.utils as vutils

from .generate import InferenceAgent
from .options.base_options import BaseOptionsJson

class LoadFloatModels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['float.pth'],)
            },
        }

    RETURN_TYPES = ("FLOAT_PIPE",)
    RETURN_NAMES = ("float_pipe",)
    FUNCTION = "loadmodel"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Models are auto-downloaded to /ComfyUI/models/float"

    def loadmodel(self, model):
        # download models if not exist
        float_models_dir = os.path.join(folder_paths.models_dir, "float")
        os.makedirs(float_models_dir, exist_ok=True)

        wav2vec2_base_960h_models_dir = os.path.join(float_models_dir,"wav2vec2-base-960h") 
        wav2vec_english_speech_emotion_recognition_models_dir = os.path.join(float_models_dir,"wav2vec-english-speech-emotion-recognition") 
        float_model_path = os.path.join(float_models_dir,"float.pth")

        if not os.path.exists(float_model_path) or not os.path.isdir(wav2vec2_base_960h_models_dir) or not os.path.isdir(wav2vec_english_speech_emotion_recognition_models_dir):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="yuvraj108c/float", local_dir=float_models_dir, local_dir_use_symlinks=False)

        # use custom dictionary instead of original parser for arguments
        opt = BaseOptionsJson
        opt.rank, opt.ngpus  = 0,1
        opt.ckpt_path = float_model_path
        opt.pretrained_dir = float_models_dir
        opt.wav2vec_model_path = wav2vec2_base_960h_models_dir
        opt.audio2emotion_path = wav2vec_english_speech_emotion_recognition_models_dir
        agent = InferenceAgent(opt)

        return (agent,)

class FloatProcess:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ref_image": ("IMAGE",),
                "ref_audio": ("AUDIO",),
                "float_pipe": ("FLOAT_PIPE",),
                "a_cfg_scale": ("FLOAT", {"default": 2.0,"min": 1.0, "step": 0.1}),
                "r_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "e_cfg_scale": ("FLOAT", {"default": 1.0,"min": 1.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 25, "step": 1}),
                "emotion": (['none', 'angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'], {"default": "none"}),
                "crop": ("BOOLEAN",{"default":False},),
                "seed": ("INT", {"default": 62064758300528, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "floatprocess"
    CATEGORY = "FLOAT"
    DESCRIPTION = "Float Processing"

    def floatprocess(self, ref_image, ref_audio, float_pipe, a_cfg_scale, r_cfg_scale, e_cfg_scale, fps, emotion, crop, seed):
        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)

        # save image
        if ref_image.shape[0] != 1:
            raise Exception("Only a single image is supported.")
        ref_image_bchw = ref_image.permute(0, 3, 1, 2)
        image_save_path = os.path.join(temp_dir, f"{int(time.time())}.png")
        vutils.save_image(ref_image_bchw[0], image_save_path)
        
        float_pipe.G.to(float_pipe.rank)

        float_pipe.opt.fps = fps
        images_bhwc = float_pipe.run_inference(
            None,
            image_save_path,
            ref_audio,
            a_cfg_scale = a_cfg_scale,
            r_cfg_scale = r_cfg_scale,
            e_cfg_scale = e_cfg_scale,
            emo 		= None if emotion == "none" else emotion,
            no_crop 	= not crop,
            seed 		= seed
        )

        float_pipe.G.to(mm.unet_offload_device())

        return (images_bhwc,)