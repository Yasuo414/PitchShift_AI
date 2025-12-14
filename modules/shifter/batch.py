import pathlib
import dataclasses
import typing
import enum
import loguru
import tqdm
import os

from modules.shifter.shift import Shift
from modules.shifter.utils import pitch_shift, shift_suffix

EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma"
}

class Status(enum.Enum):
    SUCCESS = "success"
    SKIPPED = "skipped"
    FAILED = "failed"

@dataclasses.dataclass
class Result:
    input_path: pathlib.Path
    output_path: pathlib.Path
    status: Status
    error: typing.Optional[str] = None

def get_suffix_string(key_shift: float) -> str:
    if key_shift == int(key_shift):
        return f"_{int(key_shift)}x"
    else:
        return f"_{key_shift}x"

class BatchProcessor:
    def __init__(self, nsf_hifigan: str, pitch_extractor: str = None, device: str = "cuda", sample_rate: int = 44100):
        self.shifter = Shift(nsf_hifigan, pitch_extractor, device, sample_rate)
        self.sample_rate = sample_rate
    
    @staticmethod
    def find_audio_files(path: str, recursive: bool = False) -> typing.List[pathlib.Path]:
        path = pathlib.Path(path)

        if path.is_file():
            if path.suffix.lower() in EXTENSIONS:
                return [path]
            
            return []
        
        files = []
        pattern = "**/*" if recursive else "*"

        for extension in EXTENSIONS:
            files.extend(path.glob(f"{pattern}{extension}"))
            files.extend(path.glob(f"{pattern}{extension.upper()}"))
        
        return sorted(set(files))
    
    def process(self, input_path: str, output_path: str, key_shift: float, recursive: bool = False, overwrite: bool = False, output_format: str = "wav", progress_callback: typing.Optional[typing.Callable[[int, int, str], None]] = None, silent: bool = True, add_suffix: bool = False) -> typing.List[Result]:
        input_path = pathlib.Path(input_path)
        output_path = pathlib.Path(output_path)

        files = self.find_audio_files(str(input_path), recursive)

        if not files:
            loguru.logger.warning(f"No audio files found in: {input_path}")

            return []
        
        loguru.logger.info(f"Found {len(files)} audio file(s)")
        loguru.logger.info(f"Pitch shift: {pitch_shift(key_shift)}")

        if add_suffix:
            suffix_string = get_suffix_string(key_shift)
            loguru.logger.debug(f"Suffix string: '{suffix_string}' (type: {type(suffix_string).__name__})")
        else:
            suffix_string = ""

        tasks = []
        for file in files:
            if input_path.is_file():
                if add_suffix:
                    out = output_path.parent / f"{output_path.stem}{suffix_string}{output_path.suffix}"
                else:
                    out = output_path
            else:
                relative = file.relative_to(input_path)
                if add_suffix:
                    new_name = f"{relative.stem}{suffix_string}.{output_format}"
                    out = output_path / relative.parent / new_name
                else:
                    out = output_path / relative.with_suffix(f".{output_format}")
            
            should_process = overwrite or not out.exists()
            tasks.append((file, out, should_process))
        
        results = []
        success_count = 0
        skip_count = 0
        fail_count = 0

        with tqdm.tqdm(total=len(tasks), desc="Processing", unit="file") as progress_bar:
            for index, (inp, out, should_process) in enumerate(tasks):
                if not should_process:
                    results.append(Result(inp, out, Status.SKIPPED))
                    skip_count += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        "Success": success_count,
                        "Skipped": skip_count,
                        "Failed": fail_count
                    })
                    continue

                try:
                    os.makedirs(out.parent, exist_ok=True)

                    if silent:
                        self.shifter.process_file_silent(str(inp), str(out), key_shift)
                    else:
                        self.shifter.process_file(str(inp), str(out), key_shift)

                    results.append(Result(inp, out, Status.SUCCESS))
                    success_count += 1
                except Exception as e:
                    results.append(Result(inp, out, Status.FAILED, str(e)))
                    fail_count += 1

                    if silent:
                        tqdm.tqdm.write(f"Failed: {inp.name} - {e}")
                    else:
                        loguru.logger.error(f"Failed: {inp} - {e}")
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "Success": success_count,
                    "Skipped": skip_count,
                    "Failed": fail_count
                })

                if progress_callback:
                    progress_callback(index + 1, len(tasks), str(inp.name))
        
        self.print_summary(results)

        return results
    
    def print_summary(self, results: typing.List[Result]) -> None:
        success = sum(1 for r in results if r.status == Status.SUCCESS)
        skipped = sum(1 for r in results if r.status == Status.SKIPPED)
        failed = sum(1 for r in results if r.status == Status.FAILED)
        
        loguru.logger.success("=== Process completed ===")
        loguru.logger.info(f" - Total: {len(results)}")
        loguru.logger.success(f" - Success: {success}")
        loguru.logger.info(f" - Skipped: {skipped}")
        
        if failed > 0:
            loguru.logger.error(f" - Failed: {failed}")
            loguru.logger.error("Failed files:")

            for result in results:
                if result.status == Status.FAILED:
                    loguru.logger.error(f" - {result.input_path}: {result.error}")
