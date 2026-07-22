const path    = require('path');
require('dotenv').config({ path: path.join(__dirname, '.env') });

const express  = require('express');
const cors     = require('cors');
const multer   = require('multer');
const fetch    = require('node-fetch');
const fs       = require('fs');
const { execFile } = require('child_process');

const app  = express();
const PORT = process.env.PORT || 3001;
const { OPENAI_API_KEY } = process.env;
const OPENAI_MODEL = process.env.OPENAI_MODEL || 'gpt-5.4';
const DOWNLOADER_ROOT = path.join(__dirname, 'downloader', 'YoutubeDownloader-prime');
const DOWNLOADER_BUILD_DIR = path.join(DOWNLOADER_ROOT, 'SoundMind.Downloader', 'bin', 'Debug', 'net10.0');
const DEFAULT_DOWNLOADER_TARGET = path.join(
  DOWNLOADER_BUILD_DIR,
  process.platform === 'win32' ? 'SoundMind.Downloader.exe' : 'SoundMind.Downloader'
);
const WORK_ROOT = path.resolve(process.env.SOUNDMIND_TEMP_DIR || path.join(__dirname, '.tmp'));
const CACHE_ROOT = path.join(__dirname, '.cache');

fs.mkdirSync(WORK_ROOT, { recursive: true });
fs.mkdirSync(CACHE_ROOT, { recursive: true });
const CHILD_PROCESS_ENV = {
  ...process.env,
  TEMP: WORK_ROOT,
  TMP: WORK_ROOT,
  TORCH_HOME: path.join(CACHE_ROOT, 'torch'),
  XDG_CACHE_HOME: CACHE_ROOT
};

function getDownloaderTargetPath() {
  const envPath = process.env.SOUNDMIND_DOWNLOADER_PATH || process.env.SOUND_MIND_YTD_PATH || process.env.YTD_PATH;
  return envPath ? path.resolve(envPath) : DEFAULT_DOWNLOADER_TARGET;
}

function getDownloaderInvocation(commandArgs) {
  const target = getDownloaderTargetPath();
  if (path.extname(target).toLowerCase() === '.csproj') {
    return { file: 'dotnet', args: ['run', '--project', target, '--', ...commandArgs], cwd: path.dirname(target) };
  }
  if (path.extname(target).toLowerCase() === '.dll') {
    return { file: 'dotnet', args: [target, ...commandArgs], cwd: path.dirname(target) };
  }
  return { file: target, args: commandArgs, cwd: path.dirname(target) };
}

function ensureWorkSpace(minimumBytes = 500 * 1024 * 1024) {
  if (!fs.statfsSync) return;
  const stats = fs.statfsSync(WORK_ROOT);
  const availableBytes = Number(stats.bavail) * Number(stats.bsize);
  if (availableBytes < minimumBytes) {
    throw new Error(`Not enough free space in ${WORK_ROOT}. At least 500 MB is required.`);
  }
}

function isYoutubeUrl(value) {
  try {
    const parsed = new URL(value);
    const hostname = parsed.hostname.toLowerCase();
    return (parsed.protocol === 'https:' || parsed.protocol === 'http:') && (
      hostname === 'youtu.be' ||
      hostname === 'youtube.com' ||
      hostname.endsWith('.youtube.com')
    );
  } catch {
    return false;
  }
}

app.use(cors());
app.use(express.json({ limit: '16kb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── File upload config ────────────────────────────────────
const upload = multer({
  dest: WORK_ROOT,
  limits: { fileSize: 50 * 1024 * 1024 }, // 50MB max
  fileFilter: (req, file, cb) => {
    const allowed = [
      'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave',
      'audio/ogg', 'audio/flac', 'audio/x-flac', 'audio/m4a',
      'audio/mp4', 'audio/aac', 'audio/x-m4a'
    ];
    if (allowed.includes(file.mimetype) || file.originalname.match(/\.(mp3|wav|ogg|flac|m4a|aac)$/i)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type. Please upload MP3, WAV, OGG, FLAC, or M4A.'));
    }
  }
});

// ── Scoring helpers ───────────────────────────────────────
const KEY_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const REPORT_SECTION_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    headline: { type: 'string' },
    text: { type: 'string' }
  },
  required: ['headline', 'text']
};
const REPORT_SCHEMA = {
  name: 'soundmind_psychoacoustic_report',
  strict: true,
  schema: {
    type: 'object',
    additionalProperties: false,
    properties: {
      summary: {
        type: 'object',
        additionalProperties: false,
        properties: {
          headline: { type: 'string' },
          text: { type: 'string' },
          emotional_signature: { type: 'string' },
          best_for: { type: 'string' }
        },
        required: ['headline', 'text', 'emotional_signature', 'best_for']
      },
      emotional_effect: REPORT_SECTION_SCHEMA,
      physiological_effect: REPORT_SECTION_SCHEMA,
      purpose_and_use: REPORT_SECTION_SCHEMA,
      what_you_may_not_have_noticed: REPORT_SECTION_SCHEMA,
      circumplex_zone_name: { type: 'string' },
      research_note: { type: 'string' }
    },
    required: [
      'summary',
      'emotional_effect',
      'physiological_effect',
      'purpose_and_use',
      'what_you_may_not_have_noticed',
      'circumplex_zone_name',
      'research_note'
    ]
  }
};

function modeAdjustedValence(valence, mode) {
  return Math.max(0, Math.min(1, valence + (mode === 1 ? 0.15 : -0.15)));
}
function computeArousal(tempo, energy) {
  const tempoNorm = Math.max(0, Math.min(1, (tempo - 60) / 120));
  return Math.max(0, Math.min(1, energy * 0.65 + tempoNorm * 0.35));
}
function circumplexPosition(valence, arousal) {
  const v = valence * 2 - 1;
  const a = arousal * 2 - 1;
  let angle = Math.atan2(a, v) * 180 / Math.PI;
  if (angle < 0) angle += 360;
  const radius = Math.sqrt(v * v + a * a) / Math.SQRT2;
  return {
    valence_cs:  Math.round(v    * 100) / 100,
    arousal_cs:  Math.round(a    * 100) / 100,
    angle:       Math.round(angle),
    radius:      Math.round(radius * 100) / 100
  };
}
function buildPayload(raw, songName) {
  const adjValence = modeAdjustedValence(raw.valence, raw.mode);
  const arousal    = computeArousal(raw.tempo, raw.energy);
  const pos        = circumplexPosition(adjValence, arousal);
  return {
    song_name: songName,
    track: {
      tempo:        raw.tempo,
      key:          raw.key_name  || KEY_NAMES[raw.key] || '?',
      mode:         raw.mode_name || (raw.mode === 1 ? 'major' : 'minor'),
      loudness:     raw.loudness,
      duration_s:   raw.duration_s,
      acousticness: raw.acousticness,
      danceability: raw.danceability,
      speechiness:  raw.speechiness,
    },
    texture: {
      brightness:              raw.brightness,
      spectral_centroid_hz:    raw.spectral_centroid_hz,
      spectral_bandwidth_hz:   raw.spectral_bandwidth_hz,
      spectral_rolloff_hz:     raw.spectral_rolloff_hz,
      spectral_flatness:       raw.spectral_flatness,
      hnr_db:                  raw.hnr_db,
      dynamic_range_db:        raw.dynamic_range_db,
      crest_factor:            raw.crest_factor,
      energy_variability:      raw.energy_variability,
      onset_rate_per_second:   raw.onset_rate_per_second,
      tempo_stability:         raw.tempo_stability,
    },
    vocals: raw.vocal_features || null,
    scores: {
      raw_valence:   raw.valence,
      raw_energy:    raw.energy,
      adj_valence:   Math.round(adjValence * 100) / 100,
      arousal:       Math.round(arousal    * 100) / 100,
      circumplex_angle_degrees: pos.angle,
      circumplex_radius:        pos.radius,
      circumplex_valence:       pos.valence_cs,
      circumplex_arousal:       pos.arousal_cs,
    },
    analysis: {
      confidence: raw.confidence || 'partial',
      key_confidence: raw.key_confidence,
      warnings: raw.warnings || [],
      sections: raw.sections || []
    },
    source:          'librosa',
    data_confidence: raw.confidence || 'partial'
  };
}

// ── POST /api/analyze ─────────────────────────────────────
app.post('/api/analyze', upload.single('audio'), async (req, res) => {
  const tmpPath = req.file?.path;

  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No audio file received.' });
    }
    if (!OPENAI_API_KEY) {
      return res.status(500).json({ error: 'Missing OPENAI_API_KEY in .env' });
    }

    const songName = req.body.song_name ||
      req.file.originalname.replace(/\.[^.]+$/, '') ||
      'Unknown track';

    // ── Step 1: Python audio analysis ──────────────────
    const raw = await runPythonAnalyzer(tmpPath);
    if (raw.error) {
      return res.status(422).json({ error: 'Audio analysis failed: ' + raw.error });
    }

    // ── Step 2: Build structured payload ───────────────
    const payload = buildPayload(raw, songName);

    // ── Step 3: OpenAI interpretation ──────────────────
    const report = await getReport(songName, payload);

    res.json({ payload, report });

  } catch (err) {
    console.error('[/api/analyze]', err.message);
    res.status(500).json({ error: err.message });
  } finally {
    if (tmpPath) fs.unlink(tmpPath, () => {});
  }
});

// ── GET /api/search ───────────────────────────────────────
app.get('/api/search', async (req, res) => {
  const query = String(req.query.q || '').trim();
  if (query.length < 2 || query.length > 100) {
    return res.status(400).json({ error: 'Search must be between 2 and 100 characters.' });
  }

  try {
    const results = await runYoutubeSearch(query);
    res.json({ results });
  } catch (err) {
    console.error('[/api/search]', err.message);
    res.status(502).json({ error: 'YouTube search is temporarily unavailable.' });
  }
});

// ── POST /api/download-and-analyze ─────────────────────────
app.post('/api/download-and-analyze', async (req, res) => {
  const { url, song_name } = req.body || {};
  if (!url) return res.status(400).json({ error: 'Missing "url" in request body.' });
  if (!isYoutubeUrl(url)) {
    return res.status(400).json({ error: 'Please provide a valid YouTube URL.' });
  }
  if (!OPENAI_API_KEY) {
    return res.status(500).json({ error: 'Missing OPENAI_API_KEY in .env' });
  }

  let jobDir;
  let videoPath;
  let audioPath;
  const cleanupJobDir = async () => {
    if (!jobDir) return;

    const directoryToRemove = jobDir;
    jobDir = null;
    try {
      await fs.promises.rm(directoryToRemove, { recursive: true, force: true });
    } catch (cleanupError) {
      console.error('Temporary file cleanup failed:', cleanupError.message);
    }
  };

  try {
    ensureWorkSpace();
    jobDir = await fs.promises.mkdtemp(path.join(WORK_ROOT, 'soundmind-'));
    videoPath = path.join(jobDir, 'source.mp4');
    audioPath = path.join(jobDir, 'analysis.mp3');

    await runDotnetDownloader(url, videoPath);
    await convertMp4ToMp3(videoPath, audioPath);
    const raw = await runPythonAnalyzer(audioPath);
    if (raw.error) {
      const analysisError = new Error('Audio analysis failed: ' + raw.error);
      analysisError.statusCode = 422;
      throw analysisError;
    }

    const songName = song_name || path.basename(audioPath, '.mp3');
    const payload = buildPayload(raw, songName);
    payload.source_media = {
      input: 'youtube',
      downloaded_format: 'mp4-lowest-quality',
      analyzed_format: 'mp3'
    };
    const report = await getReport(songName, payload);

    await cleanupJobDir();
    res.json({ payload, report });
  } catch (err) {
    console.error('[/api/download-and-analyze]', err.message);
    await cleanupJobDir();
    res.status(err.statusCode || 500).json({ error: 'Song analysis failed: ' + err.message });
  }
});

// ── Python analyzer ───────────────────────────────────────
function runPythonAnalyzer(filePath) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, "analyze.py");
    const python = getPythonPath();

    execFile(
      python,
      [scriptPath, filePath],
      { timeout: 15 * 60 * 1000, env: CHILD_PROCESS_ENV },
      (err, stdout, stderr) => {

        if (stderr) {
          console.error("========== PYTHON STDERR ==========");
          console.error(stderr);
          console.error("==================================");
        }

        if (err) {
          return reject(err);
        }

        try {
          resolve(JSON.parse(stdout));
        } catch (e) {
          console.error(stdout);
          reject(e);
        }
      }
    );
  });
}

// ── .NET downloader helper ─────────────────────────────────
function runDotnetDownloader(url, outPath) {
  return new Promise((resolve, reject) => {
    const target = getDownloaderTargetPath();

    if (!fs.existsSync(target)) {
      return reject(new Error(`YoutubeDownloader executable not found at ${target}. Run "npm run build:downloader" first or set SOUNDMIND_DOWNLOADER_PATH.`));
    }

    const invocation = getDownloaderInvocation([url, outPath]);

    execFile(invocation.file, invocation.args, { cwd: invocation.cwd, timeout: 15 * 60 * 1000, env: CHILD_PROCESS_ENV }, (err, stdout, stderr) => {
      if (stderr) console.error('downloader stderr:', stderr);
      if (err) return reject(err);
      if (!fs.existsSync(outPath)) {
        return reject(new Error(`Downloader finished but did not create ${outPath}. Output: ${stdout || '(empty)'}`));
      }
      resolve(stdout || '');
    });
  });
}

function runYoutubeSearch(query) {
  return new Promise((resolve, reject) => {
    const target = getDownloaderTargetPath();
    if (!fs.existsSync(target)) {
      return reject(new Error(`YoutubeDownloader executable not found at ${target}.`));
    }

    const invocation = getDownloaderInvocation(['search', query]);
    execFile(invocation.file, invocation.args, { cwd: invocation.cwd, timeout: 30000, env: CHILD_PROCESS_ENV }, (err, stdout, stderr) => {
      if (stderr) console.error('search stderr:', stderr);
      if (err) return reject(err);

      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error(`Downloader returned invalid search output: ${stdout || '(empty)'}`));
      }
    });
  });
}

function getFfmpegPath() {
  if (process.env.FFMPEG_PATH && fs.existsSync(process.env.FFMPEG_PATH)) {
    return process.env.FFMPEG_PATH;
  }

  const executable = process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg';
  const localCandidates = [
    path.join(DOWNLOADER_ROOT, 'SoundMind.Downloader', 'bin', 'Debug', 'net10.0', executable),
    path.join(DOWNLOADER_ROOT, 'SoundMind.Downloader', 'bin', 'Release', 'net10.0', executable),
    path.join(DOWNLOADER_ROOT, 'YoutubeDownloader', 'bin', 'Debug', 'net10.0', executable),
    path.join(DOWNLOADER_ROOT, 'YoutubeDownloader', 'bin', 'Release', 'net10.0', executable),
  ];

  for (const candidate of localCandidates) {
    if (fs.existsSync(candidate)) return candidate;
  }

  for (const dir of (process.env.PATH || '').split(path.delimiter)) {
    if (!dir) continue;
    const candidate = path.join(dir, executable);
    if (fs.existsSync(candidate)) return candidate;
  }

  return null;
}

function getPythonPath() {
  if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;

  const localPython = process.platform === 'win32'
    ? path.join(__dirname, '.venv', 'Scripts', 'python.exe')
    : path.join(__dirname, '.venv', 'bin', 'python');

  return fs.existsSync(localPython)
    ? localPython
    : (process.platform === 'win32' ? 'python' : 'python3');
}

function convertMp4ToMp3(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    const ffmpeg = getFfmpegPath();
    if (!ffmpeg) {
      return reject(new Error('FFmpeg was not found. Run a download once so YoutubeDownloader can fetch it, install ffmpeg, or set FFMPEG_PATH.'));
    }

    const args = ['-y', '-i', inputPath, '-vn', '-acodec', 'libmp3lame', '-q:a', '4', outputPath];
    execFile(ffmpeg, args, { timeout: 5 * 60 * 1000, env: CHILD_PROCESS_ENV }, (err, stdout, stderr) => {
      if (stderr) console.error('ffmpeg stderr:', stderr);
      if (err) return reject(err);
      if (!fs.existsSync(outputPath)) {
        return reject(new Error(`FFmpeg finished but did not create ${outputPath}. Output: ${stdout || '(empty)'}`));
      }
      resolve(outputPath);
    });
  });
}
// ── OpenAI report ─────────────────────────────────────────
async function getReport(songName, payload) {
  const prompt = `You are a rigorous music-psychology writer interpreting an automated acoustic analysis. Write with clarity, specificity, and restraint. Explain how the measured musical properties may shape a typical listener's experience without presenting associations as medical facts or claiming to know the listener's private emotions.

Use Russell's circumplex model (1980), Juslin and Sloboda's BRECVEMA framework, Koelsch's brain-network research (2014), Eerola and Vuoskoski on valence prediction, Bernardi et al. on cardiovascular entrainment (2006), and Huron's ITPRA theory (2006) only where they genuinely clarify the measurements. Do not invent studies, quotations, causal claims, or neurological precision.

Song name: "${songName}"

Computed acoustic features and analyzer caveats:
${JSON.stringify(payload, null, 2)}

Interpretation rules:
- circumplex_angle_degrees: 0°=pleasure, 90°=arousal, 180°=displeasure, 270°=sleepiness (Russell 1980)
- circumplex_radius: 0=emotionally neutral, 1=strongly saturated emotional signal
- valence and arousal are acoustic proxies, not direct measurements of emotion
- acousticness near 1.0 = acoustic/organic, near 0 = electronic/synthetic
- explicitly respect analysis.confidence and analysis.warnings; qualify uncertain key, mode, section, or vocal claims
- use section data to describe change over time when available
- when vocals is present, use its isolated pitch range, shimmer, and HNR to add a distinct vocal-delivery insight; do not infer lyrics or singer identity
- when vocals is null, do not make isolated-vocal claims
- connect claims to concrete values, but do not turn the prose into a list of statistics
- avoid repeating the same tempo, mode, or valence point across sections
- prefer active voice, varied sentence structure, and precise everyday language; avoid therapy language, hype, and generic filler
- make the summary useful on its own: one elegant 55-75 word paragraph, a precise 4-7 word headline, an emotional signature, and a concise best listening context
- lead the summary with the most defensible overall impression; mention no more than three acoustic drivers, avoid citations there, and use calibrated language such as "suggests" or "may" when moving from measurement to experience
- do not repeat the song title, raw coordinate values, or the detailed sections in the summary
- each detailed section should add a distinct insight in 1-2 compact paragraphs
- for "what you may not have noticed," identify one plausible low-awareness mechanism supported by the measurements and clearly qualify it

Return the structured report requested by the response schema. Do not use markdown.`;

  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: OPENAI_MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      response_format: {
        type: 'json_schema',
        json_schema: REPORT_SCHEMA
      }
    })
  });

  if (!r.ok) {
    const t = await r.text();
    throw new Error(`OpenAI error (${r.status}): ${t}`);
  }

  const d = await r.json();
  const raw = d.choices[0].message.content.replace(/```json|```/g, '').trim();
  return JSON.parse(raw);
}

// ── Health check ──────────────────────────────────────────
app.get('/api/health', (req, res) => {
  const python = getPythonPath();

  execFile(python, ['--version'], { env: CHILD_PROCESS_ENV }, (err) => {
    const pythonOk = !err;

    res.json({
      status: 'ok',
      openai: !!OPENAI_API_KEY,
      openai_model: OPENAI_MODEL,
      python: pythonOk,
      downloader: fs.existsSync(getDownloaderTargetPath()),
      ffmpeg: !!getFfmpegPath()
    });
  });
});

// ── Error handler ─────────────────────────────────────────
app.use((err, req, res, next) => {
  if (err.code === 'LIMIT_FILE_SIZE') {
    return res.status(413).json({ error: 'File too large. Maximum size is 50MB.' });
  }
  res.status(500).json({ error: err.message });
});

// Change this at the very bottom of your server.js file
app.listen(PORT, () => {
  console.log(`\n  SoundMind → http://localhost:${PORT}\n`);
  console.log(`  OpenAI:  ${OPENAI_API_KEY ? `✓ ${OPENAI_MODEL}` : '✗ add OPENAI_API_KEY to .env'}`);
  console.log(`  Upload:  POST /api/analyze (multipart, field: "audio")\n`);
  console.log(`  Search:  GET /api/search?q=<song name>`);
  console.log(`  YouTube: POST /api/download-and-analyze (json, field: "url")\n`);
});