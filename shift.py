import argparse
import sys
import pathlib
import loguru

import torch

from modules.shifter import Shift, BatchProcessor
from modules.shifter.utils import *

DEFAULT = {
    "nsf_hifigan": "checkpoints/nsf_hifigan/model",
    "rmvpe": "checkpoints/rmvpe/model.pt"
}

def setup_logger(verbose: bool = False, quiet: bool = False):
    loguru.logger.remove()

    if quiet:
        level = "WARNING"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    
    loguru.logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level=level, colorize=True,)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--key_shift", type=float, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--format", type=str, default="wav", choices=["wav", "flac", "mp3"])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--silent", action="store_true", required=False)
    parser.add_argument("--add_suffix", action="store_true")

    return parser.parse_args()

def validate_paths(arguments) -> bool:
    if not pathlib.Path(arguments.input).exists():
        loguru.logger.error(f"Input not found: {arguments.input}")
        
        return False
    
    if not pathlib.Path(DEFAULT["nsf_hifigan"]).exists():
        loguru.logger.error(f"NSF HiFiGAN model not found: {DEFAULT['nsf_hifigan']}")
        loguru.logger.info("Please download vocoder and place it in checkpoints/nsf_hifigan/")
        loguru.logger.info("TIP: FishAudio NSF HiFiGAN is recommended over NSF HiFiGAN by team OpenVPI")

        return False
    
    NSF_HiFiGAN_config = pathlib.Path(DEFAULT["nsf_hifigan"]).parent / "config.json"
    if not NSF_HiFiGAN_config.exists():
        loguru.logger.error(f"NSF HiFiGAN's config.json not found: {NSF_HiFiGAN_config}")

        return False
    
    if not pathlib.Path(DEFAULT["rmvpe"]).exists():
        loguru.logger.error(f"RMVPE model not found: {DEFAULT['rmvpe']}")
        loguru.logger.info("Please download pitch extractor and place it in checkpoints/rmvpe/")
        loguru.logger.info("TIP: yxlllc's is recommended over lj1995's RMVPE (co-author of RMVPE)")

        return False
    
    return True

def main():
    arguments = parse_arguments()

    setup_logger(arguments.verbose, arguments.quiet)

    loguru.logger.info("=== PitchShift AI ===")
    loguru.logger.info("by onnx4144")
    loguru.logger.info("v2")
    loguru.logger.info("=====================")

    if not validate_paths(arguments):
        return 1
    
    input_path = pathlib.Path(arguments.input)
    output_path = pathlib.Path(arguments.output)

    try:
        is_batch = input_path.is_dir() or (input_path.is_file() and output_path.is_dir())

        if is_batch:
            loguru.logger.info("Selected mode: Batch")

            if arguments.add_suffix:
                suffix = shift_suffix(arguments.key_shift)

            processor = BatchProcessor(
                nsf_hifigan=DEFAULT["nsf_hifigan"],
                pitch_extractor=DEFAULT["rmvpe"],
                device="cuda" if torch.cuda.is_available() else "cpu",
                sample_rate=44100,
            )

            results = processor.process(
                input_path=str(input_path),
                output_path=str(output_path),
                key_shift=arguments.key_shift,
                recursive=arguments.recursive,
                overwrite=arguments.overwrite,
                output_format=arguments.format,
                silent=arguments.silent,
                add_suffix=arguments.add_suffix,
            )

            failed = sum(1 for result in results if result.status.name == "FAILED")
            if failed > 0:
                return 1
            
        else:
            loguru.logger.info("Selected mode: Single")

            Shift.shift_audio(
                input=str(input_path),
                output=str(output_path),
                key_shift=arguments.key_shift,
                nsf_hifigan=DEFAULT["nsf_hifigan"],
                pitch_extractor=DEFAULT["rmvpe"],
                device=arguments.device,
                sample_rate=44100,
            )

        return 0
    
    except KeyboardInterrupt:
        loguru.logger.warning("Interrupted by user")

        return 130
    
    except Exception as e:
        loguru.logger.exception(f"Fatal error: {e}")

        return 1
    
if __name__ == "__main__":
    sys.exit(main())
