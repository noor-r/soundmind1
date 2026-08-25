# SoundMind

SoundMind analyzes an uploaded audio file or a YouTube video and produces a psychoacoustic report. YouTube input is downloaded as the lowest-quality MP4 available, converted to MP3 with FFmpeg, and then passed to the Python analyzer.

## Requirements

- Node.js 18 or newer
- Python 3.11
- .NET SDK 10 (the downloader repository pins `10.0.100`)
- An OpenAI API key

FFmpeg does not need a separate installation. The downloader fetches a local copy on the first YouTube download. You can also set `FFMPEG_PATH` to an existing executable.

## Setup

```powershell
npm install
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then validate and start the app:

```powershell
npm test
npm start
```

Open `http://localhost:3001`. Uploaded audio is limited to 50 MB.

The default report model is `gpt-5.4` for higher-quality writing. Set `OPENAI_MODEL=gpt-5.4-mini` in `.env` for faster, lower-cost reports. Reports use a strict JSON schema so the frontend always receives the expected fields.

Isolated vocal analysis is enabled by default. Demucs separates the vocal stem, and Praat measures pitch range, pitch variation, shimmer, and vocal HNR. Set `SOUNDMIND_ENABLE_DEMUCS=0` in `.env` to disable this slower stage.

The analyzer trims leading/trailing silence before feature extraction and now returns confidence diagnostics for tempo ambiguity and key ambiguity alongside the main metrics.

SoundMind stores uploads and downloaded working files in `.tmp` on the project drive and removes each job before returning its response. Set `SOUNDMIND_TEMP_DIR` to use a different drive or directory with at least 500 MB free. The server executes the downloader built by `npm test` / `postinstall` directly, avoiding a `dotnet run` build on every request.

## Deploy notes

The downloader binary is gitignored (`**/bin/`), so a fresh host must build it. Install **.NET SDK 10** on the server (same major as `downloader/YoutubeDownloader-prime/global.json`), then either:

- run `npm install` (runs `postinstall` → `build:downloader`), or
- run `npm run build:downloader` before `npm start`

If the binary is still missing at boot, `server.js` attempts one build automatically. You can also point `SOUNDMIND_DOWNLOADER_PATH` at a prebuilt `SoundMind.Downloader` / `.dll` / `.exe`.

This app needs a long-running Node host with Python and .NET — not a static-only or serverless-only host.

## API

- `POST /api/analyze`: multipart form data with an `audio` file and optional `song_name`.
- `GET /api/search?q=...`: returns up to five matching YouTube videos for song selection.
- `POST /api/download-and-analyze`: JSON with `url` and optional `song_name`.
- `GET /api/health`: reports OpenAI, Python, downloader, and FFmpeg availability.

Optional environment overrides are documented in `.env.example`.