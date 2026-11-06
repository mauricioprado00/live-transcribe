# live-transcribe

Transcribes, live and offline, both what your microphone hears and what your
speakers/headphones play, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
on the CPU. Lines are labelled `mic` (you) and `out` (the other side of a call,
a video, ...), printed as they come and appended to `transcripts/<date>_<time>.txt`.

While someone is still speaking, a dim draft of the phrase so far is shown under
the last line and redrawn about twice a second; when the phrase ends it is
replaced by the final text. Drafts come from the fast `tiny` model and are
only shown in the terminal; the transcript file gets the final lines only.

    ./transcribe                 # small model, language detected per utterance
    ./transcribe -l es           # fix the language: better accuracy on short phrases
    ./transcribe -l es,en,el     # choose among these per utterance, never anything else
    ./transcribe -m medium       # slower, more accurate; turbo / large-v3 also work
    ./transcribe --no-mic        # only what is playing
    ./transcribe --show-lag      # how far behind real time each line arrives
    ./transcribe --draft-model base     # better drafts, updated less often
    ./transcribe --partial-interval 0   # no drafts, final lines only

Ctrl-C stops; audio already captured is still transcribed before it exits.

## How it works

`parec` records `@DEFAULT_SOURCE@` (the microphone) and `@DEFAULT_MONITOR@`
(the monitor of the default output) as two 16 kHz mono streams. Each is cut
into utterances at pauses by an energy detector whose noise floor follows
steady background sound; utterances longer than `--max-len` are cut at their
quietest point. faster-whisper transcribes them with its Silero VAD on, and
segments Whisper likely invented on near-silence are dropped.

Use `pactl list short sources` to pick other devices with `--mic` / `--out`.
The monitor follows whatever the default sink is when the program starts.

## Notes

- Over open speakers the mic hears the output too, so the far side appears
  twice (`out` and `mic`). Use headphones, or `--no-mic`.
- On this laptop (no GPU) `small` runs a few seconds behind real time;
  `medium` and larger may fall behind in long, continuous speech.
- The first run downloads the model into `~/.cache/huggingface`.

Setup: `python3 -m venv .venv && .venv/bin/pip install faster-whisper`
