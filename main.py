## cli management
import click
from pathlib import Path

## core model data management
import librosa
import numpy as np
import numpy.typing as npt
import pympi
import torch
import whisperx

## logging
from contextlib import redirect_stdout
import datetime
import io
import logging
from transcribe.logging import make_loggers, make_file_handler, err_log
from tqdm import tqdm
import warnings

## typing
from typing import Literal

## globals
f = io.StringIO()
logger = make_loggers("transcribe")
logger.setLevel(logging.INFO)

## constants
SR = 16000

@err_log(logger)
def get_model(device:str|None = None):
    if device == "cuda":
        compute_type = "float16"
    elif device == "cpu":
        compute_type = "int8"
    elif torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    logger.info(f"Loading model for {device}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(f):
            model = whisperx.load_model(
                whisper_arch = "large-v2", 
                device = device, 
                compute_type = compute_type,
                language = "en",
                vad_method = "silero",
                asr_options = {"suppress_numerals": True}
            )

    return model

@err_log(logger)
def get_pairs(path:Path|str) -> list[list[Path]]:
    path = Path(path)
    if path.is_dir():
        pairs = process_dir(path)
        return pairs
    
    wav = path.with_suffix(".wav")
    eaf = path.with_suffix(".eaf")

    if wav.exists() and eaf.exists():
        return [[wav, eaf]]
    
    if not wav.exists():
        logger.warning(f"No wav file called {str(wav)}")
    
    if not eaf.exists():
        logger.warning(f"No eaf file called {str(eaf)}")

    return []

@err_log(logger)
def process_dir(path:Path) -> list[list[Path]]:
    logger.info(f"Finding wav/eaf pairs in {str(path)}")
    wavs = list(
        path.glob("*.wav")
    )

    eaf_potential = [
        p.with_suffix(".eaf")
        for p in wavs
    ]

    pairs = [
        [wav, eaf]
        for wav, eaf in zip(wavs, eaf_potential)
        if eaf.exists()
    ]
   
    logger.info(f"{len(pairs)} wav/eaf pairs found in {str(path)}")
    
    return pairs

@err_log(logger)
def transcribe_pair(pair: list[Path], model):
    wav = pair[0].with_suffix(".wav")
    eaf_file = pair[0].with_suffix(".eaf")
    parent = wav.parent
    transcript_dir = parent.joinpath("transcript")
    if not transcript_dir.exists():
        transcript_dir.mkdir()
    logger.info(f"Loading audio {str(wav)}")
    audio, _ = librosa.load(path = wav, sr=SR)
    eaf = pympi.Eaf(eaf_file)

    transcript = make_empty_transcript(eaf)
    speakers = eaf.get_tier_names()
    fhandler = make_file_handler(wav.with_suffix(".log"))
    logger.addHandler(fhandler)
    logger.info(f"Transcribing {wav.with_suffix('').name}")
    for s in speakers:
        logger.info(f"Transcribing {s}")
        clips = eaf.get_annotation_data_for_tier(id_tier = s)
        transcribe_clips(audio, clips, transcript, s, model)
    logger.info(f"Writing transcription")
    transcript.to_file(
        transcript_dir.joinpath(
            eaf_file.name
        )
    )
    logger.removeHandler(fhandler)
    pass

@err_log(logger)
def transcribe_clips(
        audio:npt.NDArray,
        clips:list[tuple[int, int, str]],
        transcript: pympi.Eaf,
        s: str,
        model
    ):
    for c in tqdm(clips, bar_format='{desc}: {percentage:3.0f}%'):

        start_idx = int(c[0]/1000 * SR)
        end_idx = int(c[1]/1000 * SR)
        audio_segment = audio[start_idx:end_idx]
        try:
            with redirect_stdout(f):
                result:dict[str, list[dict[str,float|str]]] = model.transcribe(audio_segment)
            segments = result["segments"]
            for seg in segments:
                logging.debug(f"{c[0]/1000},{c[1]/1000}, {seg['text']}")
                text = str(seg["text"])
                transcript.add_annotation(
                    id_tier=s,
                    start = c[0],
                    end = c[1],
                    value = text
                )            
        except:
            logger.warning(f"Problem transcribing clip starting at {c[0]/1000}")
            return

@err_log(logger)
def make_empty_transcript(eaf:pympi.Eaf) -> pympi.Eaf :
    speakers = eaf.get_tier_names()    
    transcript = pympi.Eaf()
    for s in speakers:
        transcript.add_tier(tier_id = s)
    transcript.remove_tier(id_tier = "default")
    return transcript


@click.command()
@click.argument(
    "PATH",
    type = click.Path(
        file_okay = True,
        dir_okay = True,
        path_type=Path
    )
)
@click.option(
    "--cache",
    type = click.Choice(
        choices = [
            "cuda",
            "cpu"
        ]
    )
)
@click.option(
    "--debug",
    is_flag = True
)
def main(path:Path|str, cache:Literal["cuda", "gpu"]|None = None, debug:bool = False):

    if debug:
        logging.basicConfig(level = logging.DEBUG)
    if cache:  
        logger.info(f"Caching models for {cache}")
        model = get_model(cache)
        return
    
    fhandler = make_file_handler(
        Path(str(datetime.datetime.now()).replace(" ", "_") + ".log")
    )
    fhandler.setLevel(logging.INFO)
    logger.addHandler(fhandler)
    
    path = Path(path)
    pairs = get_pairs(path)
    if len(pairs) < 1:
        return
    
    model = get_model()

    for p in pairs:
        transcribe_pair(p, model)

if __name__ == "__main__":
    main()
