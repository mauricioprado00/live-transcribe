#!/usr/bin/env python3
"""Live transcription of the microphone and of what the speakers/headphones play.

Each source is recorded separately with parec (PulseAudio / PipeWire-Pulse),
cut into utterances at pauses, and transcribed with faster-whisper. Lines are
labelled "mic" (you) or "out" (whatever is playing), printed as they arrive and
appended to a transcript file.
"""

import argparse
import datetime as dt
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

RATE = 16000
FRAME = 480  # 30 ms at 16 kHz
FRAME_SEC = FRAME / RATE

COLOURS = {"mic": "\033[36m", "out": "\033[33m"}
RESET = "\033[0m"
DIM = "\033[2m"


class Segmenter:
    """Cuts a stream of 30 ms frames into utterances at pauses.

    A frame is speech when its RMS is well above the noise floor, taken as the
    quietest frame of the last few seconds, so steady background sound (music,
    fans) raises the floor instead of counting as one endless utterance. An
    utterance ends after `silence` seconds without speech; one that reaches
    `max_len` is cut at the quietest moment of its last 1.5 s so words aren't
    split, and the rest carries over into the next.
    """

    FLOOR_FRAMES = int(4 / FRAME_SEC)
    CUT_SEARCH_FRAMES = int(1.5 / FRAME_SEC)

    def __init__(self, silence, max_len, min_speech, threshold):
        self.silence_frames = int(silence / FRAME_SEC)
        self.max_frames = int(max_len / FRAME_SEC)
        self.min_speech_frames = int(min_speech / FRAME_SEC)
        self.preroll_frames = int(0.3 / FRAME_SEC)
        self.threshold = threshold
        self.recent_rms = []
        self.preroll = []
        self.reset()

    def reset(self):
        self.uid = getattr(self, "uid", 0) + 1  # identifies the utterance in progress
        self.frames = []  # (frame, rms, is_speech)
        self.quiet_run = 0
        self.started_at = None

    def emit(self, frames):
        if sum(s for _, _, s in frames) < self.min_speech_frames:
            return None
        audio = np.concatenate([f for f, _, _ in frames]).astype(np.float32) / 32768.0
        return (self.uid, self.started_at, audio)

    def flush(self):
        """Ends the utterance in progress, if any; for when capture stops."""
        result = self.emit(self.frames) if self.frames else None
        self.reset()
        return result

    def in_progress(self):
        """(uid, start_time, audio) of the utterance so far, once it holds enough speech."""
        if not self.frames:
            return None
        return self.emit(self.frames)

    def feed(self, frame, now):
        """Returns (uid, start_time, audio) when an utterance completes, else None."""
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))
        self.recent_rms.append(rms)
        del self.recent_rms[: -self.FLOOR_FRAMES]
        is_speech = rms > max(self.threshold, min(self.recent_rms) * 2.5)

        if not self.frames:
            if not is_speech:
                self.preroll.append((frame, rms, False))
                del self.preroll[: -self.preroll_frames]
                return None
            self.frames = self.preroll
            self.preroll = []
            self.started_at = now - len(self.frames) * FRAME_SEC

        self.frames.append((frame, rms, is_speech))
        self.quiet_run = 0 if is_speech else self.quiet_run + 1

        if self.quiet_run >= self.silence_frames:
            result = self.emit(self.frames)
            self.reset()
            return result

        if len(self.frames) >= self.max_frames:
            tail = self.frames[-self.CUT_SEARCH_FRAMES:]
            cut = len(self.frames) - len(tail) + int(np.argmin([r for _, r, _ in tail])) + 1
            result = self.emit(self.frames[:cut])
            rest = self.frames[cut:]
            self.reset()
            if rest:
                self.frames = rest
                self.started_at = now - len(rest) * FRAME_SEC
            return result
        return None


class Source:
    """One recorded stream: its segmenter, and the finished utterances awaiting transcription."""

    def __init__(self, label, device, args):
        self.label = label
        self.device = device
        self.seg = Segmenter(args.silence, args.max_len, args.min_speech, args.threshold)
        self.lock = threading.Lock()  # guards seg between the capture and worker threads
        self.finals = queue.Queue()
        self.captured = threading.Event()  # set when capture has ended

    def capture(self, stop):
        cmd = [
            "parec", "--raw", "--latency-msec=100", "-d", self.device,
            f"--rate={RATE}", "--channels=1", "--format=s16le",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        nbytes = FRAME * 2
        try:
            while not stop.is_set():
                data = proc.stdout.read(nbytes)
                if len(data) < nbytes:
                    if not stop.is_set():
                        print(f"[{self.label}] capture from {self.device} ended", file=sys.stderr)
                    break
                with self.lock:
                    out = self.seg.feed(np.frombuffer(data, dtype=np.int16), time.time())
                if out:
                    self.finals.put(out)
        finally:
            proc.terminate()
            with self.lock:
                out = self.seg.flush()
            if out:
                self.finals.put(out)
            self.captured.set()


class Display:
    """Final lines scroll up; below them, one draft line per source is redrawn in place."""

    def __init__(self, labels, out_file, show_partials, show_lag):
        self.labels = labels
        self.out_file = out_file
        self.show_partials = show_partials
        self.show_lag = show_lag
        self.partials = {}
        self.drawn = 0
        self.lock = threading.Lock()

    def _erase(self):
        if self.drawn:
            sys.stdout.write(f"\033[{self.drawn}F\033[J")
            self.drawn = 0

    def _draw(self):
        width = shutil.get_terminal_size().columns
        for label in self.labels:
            text = self.partials.get(label)
            if not text:
                continue
            room = width - len(label) - 4
            if len(text) > room:  # keep to one line, so it can be erased; show the newest words
                text = "…" + text[-(room - 1):]
            sys.stdout.write(f"{COLOURS[label]}{label}{RESET}  {DIM}{text}{RESET}\n")
            self.drawn += 1
        sys.stdout.flush()

    def partial(self, label, text):
        if not self.show_partials:
            return
        with self.lock:
            self._erase()
            self.partials[label] = text
            self._draw()

    def final(self, label, started, audio, text):
        with self.lock:
            self._erase()
            self.partials.pop(label, None)
            if text:
                stamp = dt.datetime.fromtimestamp(started).strftime("%H:%M:%S")
                lag = time.time() - started - len(audio) / RATE
                sys.stdout.write(f"{DIM}{stamp}{RESET} {COLOURS[label]}{label}{RESET}  {text}"
                                 + (f" {DIM}(+{lag:.0f}s){RESET}" if self.show_lag else "") + "\n")
                self.out_file.write(f"[{stamp}] {label}: {text}\n")
                self.out_file.flush()
            self._draw()


def is_hallucination(seg):
    # Whisper invents text ("Thank you.", subtitle credits) on near-silence,
    # and loops on a phrase when it loses track (high compression ratio).
    return (seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.8) or seg.compression_ratio > 2.4


def pick_language(model, args, audio):
    if len(args.languages) <= 1:
        return args.languages[0] if args.languages else None
    # Pick the likeliest of the allowed languages; free detection on a short
    # phrase often lands on an unrelated one.
    _, _, probs = model.detect_language(audio)
    probs = dict(probs)
    return max(args.languages, key=lambda l: probs.get(l, 0.0))


def transcribe(model, audio, language, beam_size, temperature):
    segments, _ = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=temperature,
    )
    return " ".join(s.text.strip() for s in segments if not is_hallucination(s)).strip()


def worker(model, draft_model, args, src, display):
    """Transcribes src's finished utterances as they come and, while none is
    waiting, redrafts the one in progress every --partial-interval seconds
    with the (smaller, faster) draft model."""
    drafted = (None, 0.0)  # (uid, seconds of audio) of the last draft
    draft_lang = (None, None)  # (uid, language) so drafts detect it only once
    while True:
        try:
            uid, started, audio = src.finals.get(timeout=0.05)
        except queue.Empty:
            if src.captured.is_set() and src.finals.empty():
                return  # capture queues its last utterance before setting captured
            if args.partial_interval <= 0:
                continue
            with src.lock:
                current = src.seg.in_progress()
            if not current:
                continue
            uid, started, audio = current
            secs = len(audio) / RATE
            if secs < 0.7 or (uid == drafted[0] and secs - drafted[1] < args.partial_interval):
                continue
            drafted = (uid, secs)
            if draft_lang[0] != uid:
                draft_lang = (uid, pick_language(draft_model, args, audio))
            # Greedy, no fallback: a draft has to be quick, not perfect.
            text = transcribe(draft_model, audio, draft_lang[1], 1, 0.0)
            if text and src.finals.empty():  # else the final is already waiting
                display.partial(src.label, text)
            continue
        text = transcribe(model, audio, pick_language(model, args, audio), args.beam_size,
                          # One retry at most: the default six-step fallback can
                          # stall a worker for many seconds on a hard utterance.
                          (0.0, 0.4))
        display.final(src.label, started, audio, text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", default="small",
                   help="whisper model: tiny, base, small, medium, large-v3, turbo, ... (default small)")
    p.add_argument("-l", "--language", default=None,
                   help="language code such as es, or a list such as es,en,el to choose from per "
                        "utterance; default is to detect any language per utterance")
    p.add_argument("--mic", default="@DEFAULT_SOURCE@", help="Pulse source for the microphone")
    p.add_argument("--out", default="@DEFAULT_MONITOR@", help="Pulse source for the speakers/headphones")
    p.add_argument("--no-mic", action="store_true", help="don't record the microphone")
    p.add_argument("--no-out", action="store_true", help="don't record the output")
    p.add_argument("-o", "--output", default=None,
                   help="transcript file (default transcripts/YYYY-MM-DD_HHMMSS.txt next to this script)")
    p.add_argument("--silence", type=float, default=0.8, help="pause in seconds that ends an utterance")
    p.add_argument("--max-len", type=float, default=20, help="longest utterance in seconds before it is cut")
    p.add_argument("--min-speech", type=float, default=0.3, help="ignore utterances with less speech than this")
    p.add_argument("--threshold", type=float, default=300, help="minimum RMS (int16) counted as speech")
    p.add_argument("--beam-size", type=int, default=1, help="1 is fastest; 5 is a little more accurate")
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--partial-interval", type=float, default=0.5,
                   help="seconds between drafts of the phrase being spoken; 0 shows only final lines")
    p.add_argument("--draft-model", default="tiny",
                   help="model for the drafts shown while a phrase is spoken (default tiny); "
                        "the final line always comes from --model")
    p.add_argument("--show-lag", action="store_true", help="show how far behind real time each line is")
    args = p.parse_args()
    args.languages = [l.strip() for l in args.language.split(",")] if args.language else []

    sources = []
    if not args.no_mic:
        sources.append(Source("mic", args.mic, args))
    if not args.no_out:
        sources.append(Source("out", args.out, args))
    if not sources:
        p.error("nothing to record")

    if args.output is None:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
        os.makedirs(d, exist_ok=True)
        args.output = os.path.join(d, dt.datetime.now().strftime("%Y-%m-%d_%H%M%S.txt"))

    from faster_whisper import WhisperModel

    print(f"{DIM}loading model {args.model}...{RESET}", file=sys.stderr)
    compute = "int8" if args.device == "cpu" else "float16"
    # One worker per source, so the mic and the output are transcribed in parallel.
    def load(name):
        return WhisperModel(name, device=args.device, compute_type=compute,
                            cpu_threads=max(1, args.threads // len(sources)), num_workers=len(sources))
    model = load(args.model)
    draft_model = model
    if args.partial_interval > 0 and args.draft_model != args.model:
        draft_model = load(args.draft_model)

    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    with open(args.output, "a", encoding="utf-8") as out_file:
        display = Display([s.label for s in sources], out_file,
                          show_partials=sys.stdout.isatty() and args.partial_interval > 0,
                          show_lag=args.show_lag)
        captures = [threading.Thread(target=s.capture, args=(stop,), daemon=True) for s in sources]
        workers = [threading.Thread(target=worker, args=(model, draft_model, args, s, display)) for s in sources]
        for t in captures + workers:
            t.start()
        print(f"{DIM}listening to {', '.join(s.label for s in sources)}; writing {args.output}; "
              f"Ctrl-C to stop{RESET}", file=sys.stderr)
        try:
            while any(w.is_alive() for w in workers):
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        stop.set()
        print(f"\n{DIM}finishing queued audio...{RESET}", file=sys.stderr)
        for w in workers:
            w.join()
    print(f"{DIM}transcript: {args.output}{RESET}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: CTranslate2 aborts if it is destroyed while the
    # capture threads are still blocked reading from parec.
    os._exit(0)


if __name__ == "__main__":
    main()
