require('dotenv').config();
const http = require('http');
const fs = require('fs');
const crypto = require('crypto');
const pg = require('pg');

// ---------- Network hardening (slow / unstable connections) ----------
try { require('dns').setDefaultResultOrder('ipv4first'); } catch {} // avoids IPv6 hangs on some ISPs
try {
  // npm i undici  -> longer connect timeout than Node's default 10s, optional proxy/VPN-style routing
  const { Agent, ProxyAgent, setGlobalDispatcher } = require('undici');
  const proxy = process.env.HTTPS_PROXY || process.env.PROXY_URL;
  setGlobalDispatcher(proxy ? new ProxyAgent(proxy) : new Agent({ connect: { timeout: 45000 } }));
} catch {
  console.log('Tip: run "npm install" so undici is available (longer connect timeout).');
}

// fetch with a fresh timeout per attempt and retries on network errors
async function fetchRetry(url, opts = {}, timeoutMs = 25000, tries = 3) {
  let lastErr;
  for (let i = 0; i < tries; i++) {
    try {
      const res = await fetch(url, { ...opts, signal: AbortSignal.timeout(timeoutMs) });
      if ((res.status === 429 || res.status >= 500) && i < tries - 1) {
        const ra = Math.min(Number(res.headers.get('retry-after')) || 0, 60) * 1000;
        console.log(`HTTP ${res.status} from ${new URL(url).host}, retrying (${i + 1}/${tries})`);
        await new Promise((r) => setTimeout(r, Math.max(ra, 1500 * (i + 1))));
        continue;
      }
      return res;
    } catch (e) {
      lastErr = e;
      console.log(`Network error (${(e.cause && e.cause.code) || e.message}) on attempt ${i + 1}/${tries}`);
      await new Promise((r) => setTimeout(r, 1500 * (i + 1)));
    }
  }
  throw lastErr;
}
const { Telegraf, Markup } = require('telegraf');

// =====================================================================
//  Football Predictions Bot
//  - Fixtures + match history : football-data.org   (free key)
//  - Odds                     : the-odds-api.com    (free key, 500 credits/month)
//  - AI predictions           : online hosted models (Groq / OpenRouter / Gemini / Hugging Face)
//  Prediction = log-linear blend of  market odds (62%)  +  opponent-adjusted Poisson/Dixon-Coles model (30%)  +  LLM (8%, clamped near the market)
//  (weights are tunable via W_MARKET / W_FORM / W_AI in .env)
// =====================================================================

// ---------- Config ----------
const BOT_TOKEN = process.env.BOT_TOKEN;
const FD_KEY = process.env.FOOTBALL_DATA_KEY || '';
const ODDS_KEY = process.env.ODDS_API_KEY || '';
// AI provider (OpenAI-compatible chat API) - ONLINE hosted models.
// AI_PROVIDER = groq | openrouter | gemini | hf | pollinations | custom | off   (ollama = local, optional)
//   groq         : GROQ_API_KEY        - fast, generous free tier (https://console.groq.com)
//   openrouter   : OPENROUTER_API_KEY  - many ":free" models (https://openrouter.ai)
//   gemini       : GEMINI_API_KEY      - Google AI Studio free tier (https://aistudio.google.com)
//   hf           : HF_TOKEN            - Hugging Face router (free monthly credits)
//   pollinations : no key, but unreliable (may answer 402/500)
//   custom       : AI_BASE_URL + AI_API_KEY + AI_MODELS (any OpenAI-compatible host)
// If AI_PROVIDER is not set, the first key found in .env is used (groq > openrouter > gemini > hf).
// Override the model list any time with AI_MODELS="modelA,modelB" (tried in order, first that works wins).
const AI_KEY = process.env.AI_API_KEY || '';
function detectProvider() {
  if (process.env.AI_PROVIDER) return process.env.AI_PROVIDER.toLowerCase();
  if (process.env.AI_BASE_URL) return 'custom';
  if (process.env.GROQ_API_KEY) return 'groq';
  if (process.env.OPENROUTER_API_KEY) return 'openrouter';
  if (process.env.GEMINI_API_KEY) return 'gemini';
  if (process.env.HF_TOKEN) return 'hf';
  return 'off';
}
const PROVIDER = detectProvider();
const PRESETS = {
  groq: { url: 'https://api.groq.com/openai/v1', models: 'llama-3.3-70b-versatile,llama-3.1-8b-instant', key: process.env.GROQ_API_KEY || AI_KEY, conc: 2 },
  openrouter: { url: 'https://openrouter.ai/api/v1', models: 'meta-llama/llama-3.3-70b-instruct:free,qwen/qwen-2.5-72b-instruct:free', key: process.env.OPENROUTER_API_KEY || AI_KEY, conc: 2 },
  gemini: { url: 'https://generativelanguage.googleapis.com/v1beta/openai', models: 'gemini-2.0-flash,gemini-1.5-flash', key: process.env.GEMINI_API_KEY || AI_KEY, conc: 2 },
  hf: { url: 'https://router.huggingface.co/v1', models: 'meta-llama/Llama-3.1-8B-Instruct,Qwen/Qwen2.5-7B-Instruct', key: process.env.HF_TOKEN || AI_KEY, conc: 3 },
  pollinations: { url: 'https://text.pollinations.ai/openai', models: 'openai,mistral', key: '', conc: 1 },
  ollama: { url: 'http://localhost:11434/v1', models: 'qwen2.5:3b', key: '', conc: 1 },
  custom: { url: process.env.AI_BASE_URL || '', models: '', key: AI_KEY, conc: 3 },
};
const PRESET = PRESETS[PROVIDER] || PRESETS.pollinations;
const KEYED = ['groq', 'openrouter', 'gemini', 'hf'];
const AI_BASE_URL = (process.env.AI_BASE_URL || PRESET.url).replace(/\/+$/, '');
const AI_USE_KEY = PRESET.key;
const HF_MODELS = (process.env.AI_MODELS || (PROVIDER === 'hf' ? process.env.HF_MODELS : '') || PRESET.models).split(',').map((x) => x.trim()).filter(Boolean);
const AI_ON = PROVIDER !== 'off' && HF_MODELS.length > 0 && !!AI_BASE_URL && (!KEYED.includes(PROVIDER) || !!AI_USE_KEY);
const AI_TIMEOUT_MS = Number(process.env.AI_TIMEOUT_SEC || (PROVIDER === 'ollama' ? 180 : 60)) * 1000;
const AI_CONCURRENCY = Number(process.env.AI_CONCURRENCY || PRESET.conc);
const TZ = process.env.TIMEZONE || 'Africa/Lagos';
const CACHE_MS = Number(process.env.CACHE_MINUTES || 180) * 60 * 1000;
const SEASON_CACHE_MS = 6 * 60 * 60 * 1000;
const MIN_PICK_ODDS = Number(process.env.MIN_PICK_ODDS || 1.15); // skip "pointless" 1.05 favourites in Top 5
const MAX_AI_MATCHES = Number(process.env.MAX_AI_MATCHES || 40); // how many top candidates get an LLM opinion
// ----- Model tuning -----
const W_MKT = Number(process.env.W_MARKET || 0.62);  // market odds are the strongest single signal
const W_FORM = Number(process.env.W_FORM || 0.30);   // team-rating Poisson model
const W_AI = Number(process.env.W_AI || 0.15);       // small LLMs are weak at probabilities -> low weight
const AI_MAX_DEV = Number(process.env.AI_MAX_DEV || 0.08); // LLM may move each outcome at most +-8 pts away from the market
const FORM_HALF_LIFE_DAYS = Number(process.env.FORM_HALF_LIFE_DAYS || 75); // older results count less
const DC_RHO = Number(process.env.DC_RHO || -0.1);   // Dixon-Coles low-score correction
const GOAL_CAP = Number(process.env.GOAL_CAP || 5);   // blow-outs (7-0) are capped so they do not distort team ratings
const PREV_SEASON_ON = process.env.PREV_SEASON !== 'off'; // use last season as a prior (huge help early in the season)
const PREV_WEIGHT = Number(process.env.PREV_WEIGHT || 0.4);
const DISAGREE_PENALTY = Number(process.env.DISAGREE_PENALTY || 0.5); // picks where model and market disagree rank lower
const ODDS_REGIONS = process.env.ODDS_REGIONS || 'eu'; // 'eu,uk' adds Betfair/Matchbook/Smarkets (sharpest prices) but costs 2x credits
const OUTLIER_DEV = Number(process.env.OUTLIER_DEV || 0.07); // soft-book lines this far from consensus are treated as stale and dropped
const PRED_LOG = process.env.PRED_LOG || './predictions.json'; // prediction log for /stats (use a persistent disk on Render)
// ----- Best 3 (under-the-radar picks) -----
const TOP_RANK_EXCLUDE = Number(process.env.TOP_RANK_EXCLUDE || 4); // skip teams sitting in the top N of their league table
const MIN_BEST3_GAMES = Number(process.env.MIN_BEST3_GAMES || 5);   // both teams need this many league games of data
const BEST3_MIN_CONF = Number(process.env.BEST3_MIN_CONF || 0.4);   // minimum blended probability for the pick
// "Popular" clubs that Best 3 never touches (matched loosely by name). Add more with EXTRA_BIG_CLUBS="Club A,Club B"
const BIG_CLUBS = [
  'Manchester City', 'Manchester United', 'Liverpool', 'Arsenal', 'Chelsea', 'Tottenham Hotspur', 'Newcastle United',
  'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Bayern', 'Dortmund', 'Leverkusen',
  'Juventus', 'Milan', 'Inter', 'Napoli', 'Roma', 'Paris Saint Germain', 'Marseille',
  'Ajax', 'PSV', 'Benfica', 'Porto', 'Sporting Portugal', 'Flamengo', 'Palmeiras',
].concat((process.env.EXTRA_BIG_CLUBS || '').split(',').map((x) => x.trim()).filter(Boolean));
const PAGE_SIZE = 6;

if (!BOT_TOKEN || !FD_KEY || !process.env.DATABASE_URL) {
  console.error('Missing BOT_TOKEN, FOOTBALL_DATA_KEY or DATABASE_URL in .env');
  process.exit(1);
}

// Health server + Paystack webhook (Render etc. need an open port)
let paystackWebhook = null; // assigned further down when PAYSTACK_SECRET_KEY is set
if (process.env.PORT || process.env.ENABLE_HEALTH_SERVER === 'true' || process.env.PAYSTACK_SECRET_KEY) {
  const PORT = process.env.PORT || 3000;
  http
    .createServer((req, res) => {
      const path = (req.url || '').split('?')[0];
      if (req.method === 'POST' && path === '/paystack/webhook' && paystackWebhook) return paystackWebhook(req, res);
      if (req.method === 'GET' && path === '/paystack/callback') {
        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        return res.end('<meta name="viewport" content="width=device-width,initial-scale=1"><body style="font-family:sans-serif;text-align:center;padding:3em"><h2>✅ Payment received</h2><p>Go back to Telegram — Premium unlocks within seconds. If not, tap “I have paid” in the bot.</p></body>');
      }
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.end('Football bot is running');
    })
    .listen(PORT, '0.0.0.0', () => console.log('Health server on port ' + PORT));
}

// football-data.org competition code -> The Odds API sport key
const COMP_TO_ODDS = {
  PL: 'soccer_epl',
  ELC: 'soccer_efl_champ',
  BL1: 'soccer_germany_bundesliga',
  SA: 'soccer_italy_serie_a',
  PD: 'soccer_spain_la_liga',
  FL1: 'soccer_france_ligue_one',
  DED: 'soccer_netherlands_eredivisie',
  PPL: 'soccer_portugal_primeira_liga',
  CL: 'soccer_uefa_champs_league',
  BSA: 'soccer_brazil_campeonato',
  EC: 'soccer_uefa_european_championship',
  WC: 'soccer_fifa_world_cup',
};

// ---------- Small helpers ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const pct = (x) => Math.round(x * 100);

const localDate = (d) =>
  new Intl.DateTimeFormat('en-CA', { timeZone: TZ, year: 'numeric', month: '2-digit', day: '2-digit' }).format(d);
const fmtTime = (ms) =>
  new Intl.DateTimeFormat('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(ms));
const fmtDate = (iso) =>
  new Intl.DateTimeFormat('en-GB', { timeZone: TZ, day: '2-digit', month: 'short' }).format(new Date(iso));

// In-memory cache with TTL + de-duplication of concurrent calls
const cache = new Map();
function cached(key, ttlMs, fn) {
  const c = cache.get(key);
  if (c && c.exp > Date.now()) return Promise.resolve(c.val);
  if (c && c.pending) return c.pending;
  const pending = fn()
    .then((val) => {
      cache.set(key, { val, exp: Date.now() + ttlMs });
      return val;
    })
    .catch((e) => {
      cache.delete(key);
      throw e;
    });
  cache.set(key, { pending, exp: 0 });
  return pending;
}

// drop expired cache entries so a long-running bot does not grow forever
setInterval(() => {
  const now = Date.now();
  for (const [k, v] of cache) if (!v.pending && v.exp <= now) cache.delete(k);
}, 30 * 60e3).unref();

async function pool(items, n, fn) {
  const out = new Array(items.length);
  let i = 0;
  await Promise.all(
    Array.from({ length: Math.min(n, items.length) }, async () => {
      while (i < items.length) {
        const idx = i++;
        out[idx] = await fn(items[idx], idx);
      }
    })
  );
  return out;
}

// ---------- football-data.org (rate limit: 10 req/min on the free plan) ----------
let fdChain = Promise.resolve();
function fdGet(path) {
  const run = async () => {
    const res = await fetchRetry('https://api.football-data.org/v4' + path, {
      headers: { 'X-Auth-Token': FD_KEY },
    });
    if (!res.ok) throw new Error(`football-data ${res.status} on ${path}`);
    return res.json();
  };
  const p = fdChain.then(run, run);
  fdChain = p.then(() => sleep(6500), () => sleep(6500)); // stay under 10 requests/min
  return p;
}

function getSeason(code) {
  if (String(code).startsWith('X:')) return Promise.resolve([]); // extra (non football-data) league
  return cached('season:' + code, SEASON_CACHE_MS, async () => {
    const data = await fdGet(`/competitions/${code}/matches?status=FINISHED`);
    return data.matches || [];
  });
}

// Last season = prior for team strength (fades out as the current season fills up). Fails soft on the free plan.
function seasonStartYear(code) {
  const d = new Date();
  if (code === 'BSA') return d.getUTCFullYear();
  return d.getUTCMonth() >= 6 ? d.getUTCFullYear() : d.getUTCFullYear() - 1;
}
const prevFail = new Map();
function getPrevSeason(code) {
  if (!PREV_SEASON_ON || !(DOMESTIC.has(code) || code === 'CL') || (prevFail.get(code) || 0) > Date.now()) return Promise.resolve([]);
  return cached('prev:' + code, 24 * 3600e3, async () => {
    const data = await fdGet(`/competitions/${code}/matches?season=${seasonStartYear(code) - 1}&status=FINISHED`);
    return data.matches || [];
  }).catch((e) => {
    prevFail.set(code, Date.now() + 3600e3);
    console.log('Previous season unavailable for', code, '-', e.message);
    return [];
  });
}

// ---------- Team-name matching (football-data <-> odds feed) ----------
const STOP = new Set(['fc', 'afc', 'cf', 'sc', 'ac', 'as', 'fk', 'sv', 'the', 'and', 'de', 'club', '1', '04', '05', '09']);
const ALIASES = [
  [/\bwolves\b/g, 'wolverhampton wanderers'],
  [/\bspurs\b/g, 'tottenham hotspur'],
  [/\bman city\b/g, 'manchester city'],
  [/\bman (utd|united)\b/g, 'manchester united'],
  [/\bpsg\b/g, 'paris saint germain'],
  [/\bnewcastle utd\b/g, 'newcastle united'],
];
function toks(s) {
  let t = s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().replace(/&/g, ' ');
  for (const [re, rep] of ALIASES) t = t.replace(re, rep);
  return t.replace(/[^a-z0-9 ]/g, ' ').split(/\s+/).filter((x) => x && !STOP.has(x));
}
function tokMatch(a, b) {
  if (a === b) return true;
  const [s, l] = a.length <= b.length ? [a, b] : [b, a];
  return s.length >= 4 && l.startsWith(s); // inter ~ internazionale, milan ~ milano
}
function nameScore(a, b) {
  const A = toks(a);
  const B = toks(b);
  if (!A.length || !B.length) return 0;
  let hit = 0;
  for (const x of A) if (B.some((y) => tokMatch(x, y))) hit++;
  return hit / Math.min(A.length, B.length);
}

// ---------- The Odds API ----------
// Margin removal with the POWER method: fixes the favourite-longshot bias that simple normalisation leaves in.
function devig(inv) {
  const sum = inv.H + inv.D + inv.A;
  if (sum <= 1.0001) return { H: inv.H / sum, D: inv.D / sum, A: inv.A / sum };
  let lo = 1, hi = 6;
  for (let i = 0; i < 40; i++) {
    const c = (lo + hi) / 2;
    if (Math.pow(inv.H, c) + Math.pow(inv.D, c) + Math.pow(inv.A, c) > 1) lo = c; else hi = c;
  }
  const c = (lo + hi) / 2;
  const f = { H: Math.pow(inv.H, c), D: Math.pow(inv.D, c), A: Math.pow(inv.A, c) };
  const t = f.H + f.D + f.A;
  return { H: f.H / t, D: f.D / t, A: f.A / t };
}
// sharp / exchange books price more accurately than soft books -> count them more
const SHARP_BOOKS = { pinnacle: 4, betfair_ex_eu: 3, betfair_ex_uk: 3, matchbook: 2, smarkets: 2 };

async function fetchOdds(sportKey) {
  if (!ODDS_KEY) return [];
  const url =
    `https://api.the-odds-api.com/v4/sports/${sportKey}/odds` +
    `?regions=${encodeURIComponent(ODDS_REGIONS)}&markets=h2h&oddsFormat=decimal&apiKey=${ODDS_KEY}`;
  const res = await fetchRetry(url);
  noteCredits(res);
  if (!res.ok) throw new Error('Odds API ' + res.status);
  const events = await res.json();
  const KS = ['H', 'D', 'A'];
  return events
    .map((ev) => {
      const books = [];
      for (const bk of ev.bookmakers || []) {
        const mk = (bk.markets || []).find((m) => m.key === 'h2h');
        if (!mk) continue;
        const o = {};
        for (const out of mk.outcomes) {
          if (out.name === ev.home_team) o.H = out.price;
          else if (out.name === ev.away_team) o.A = out.price;
          else if (out.name === 'Draw') o.D = out.price;
        }
        if (!(o.H && o.D && o.A)) continue;
        books.push({ o, fair: devig({ H: 1 / o.H, D: 1 / o.D, A: 1 / o.A }), w: SHARP_BOOKS[bk.key] || 1, title: bk.title || bk.key });
      }
      if (!books.length) return null;
      const consensus = (list) => {
        const sp = { H: 0, D: 0, A: 0 };
        let sw = 0;
        for (const b of list) { for (const k of KS) sp[k] += b.fair[k] * b.w; sw += b.w; }
        return { H: sp.H / sw, D: sp.D / sw, A: sp.A / sw };
      };
      let used = books;
      let probs = consensus(used);
      if (books.length >= 5) {
        // drop stale / outlier soft-book lines (sharp books are never dropped)
        const kept = books.filter((b) => b.w > 1 || KS.every((k) => Math.abs(b.fair[k] - probs[k]) <= OUTLIER_DEV));
        if (kept.length >= 3 && kept.length < books.length) { used = kept; probs = consensus(used); }
      }
      const n = used.length;
      const avg = (k) => used.reduce((s, b) => s + b.o[k], 0) / n;
      const mx = (k) => Math.max(...used.map((b) => b.o[k]));
      const mxBk = (k) => used.reduce((a, b) => (b.o[k] > a.o[k] ? b : a)).title;
      return {
        home: ev.home_team,
        away: ev.away_team,
        time: new Date(ev.commence_time).getTime(),
        n,
        odds: { H: avg('H'), D: avg('D'), A: avg('A') },
        best: { H: mx('H'), D: mx('D'), A: mx('A') },
        bestBk: { H: mxBk('H'), D: mxBk('D'), A: mxBk('A') },
        probs,
      };
    })
    .filter(Boolean);
}

function findOdds(fix, events) {
  let best = null;
  let bestScore = 0;
  for (const ev of events) {
    if (Math.abs(ev.time - fix.utc) > 6 * 3600e3) continue;
    const a = nameScore(fix.home, ev.home);
    const b = nameScore(fix.away, ev.away);
    if (a <= 0.5 || b <= 0.5) continue; // 0.5 = only one shared word (Man United vs Man City) → reject
    if (a + b > bestScore) {
      bestScore = a + b;
      best = ev;
    }
  }
  return best;
}


// ---------- EXTRA fixtures (leagues football-data.org free plan does NOT cover) ----------
// Source: ESPN public scoreboard (free, no key, no credits). Odds are added from The Odds API when a league maps to one.
// Env: EXTRA_FIXTURES=false to switch off | EXTRA_ODDS=false to never spend Odds API credits on them | EXTRA_ODDS_MAX=5 (leagues per refresh)
// EXTRA_LEAGUES="eng.3:League One,nga.1:NPFL" adds/overrides ESPN league slugs (slug:Name).
const EXTRA_ON = process.env.EXTRA_FIXTURES !== 'false';
const EXTRA_ODDS_ON = process.env.EXTRA_ODDS !== 'false';
const EXTRA_ODDS_MAX = Number(process.env.EXTRA_ODDS_MAX || 5);
const ESPN_LEAGUES = {
  'eng.3': 'England League One', 'eng.4': 'England League Two', 'eng.fa': 'FA Cup', 'eng.league_cup': 'EFL Cup',
  'esp.2': 'Spain Segunda', 'esp.copa_del_rey': 'Copa del Rey', 'ger.2': 'Germany 2. Bundesliga', 'ita.2': 'Italy Serie B',
  'fra.2': 'France Ligue 2', 'bel.1': 'Belgium Pro League', 'sco.1': 'Scottish Premiership', 'tur.1': 'Turkey Super Lig',
  'gre.1': 'Greece Super League', 'aut.1': 'Austria Bundesliga', 'sui.1': 'Swiss Super League', 'den.1': 'Denmark Superliga',
  'nor.1': 'Norway Eliteserien', 'swe.1': 'Sweden Allsvenskan', 'rus.1': 'Russian Premier League',
  'uefa.europa': 'Europa League', 'uefa.europa.conf': 'Conference League', 'uefa.nations': 'UEFA Nations League',
  'fifa.friendly': 'International Friendly', 'conmebol.libertadores': 'Copa Libertadores', 'conmebol.sudamericana': 'Copa Sudamericana',
  'arg.1': 'Argentina Liga', 'usa.1': 'MLS', 'mex.1': 'Liga MX', 'jpn.1': 'J1 League', 'chn.1': 'Chinese Super League',
  'aus.1': 'A-League', 'ksa.1': 'Saudi Pro League', 'caf.champions': 'CAF Champions League', 'caf.confed': 'CAF Confederation Cup',
  'concacaf.champions': 'CONCACAF Champions Cup', 'rsa.1': 'South Africa PSL', 'egy.1': 'Egypt Premier League',
  // already covered by football-data but harmless: duplicates are removed below, and they back it up when football-data is slow
  'eng.1': 'Premier League', 'eng.2': 'Championship', 'esp.1': 'La Liga', 'ger.1': 'Bundesliga', 'ita.1': 'Serie A',
  'fra.1': 'Ligue 1', 'ned.1': 'Eredivisie', 'por.1': 'Primeira Liga', 'bra.1': 'Brasileirao', 'uefa.champions': 'Champions League',
};
for (const part of (process.env.EXTRA_LEAGUES || '').split(',')) {
  const [slug, ...nm] = part.split(':'); if (slug && slug.trim()) ESPN_LEAGUES[slug.trim()] = nm.join(':').trim() || slug.trim();
}
// ESPN slug -> The Odds API sport key (only used to attach odds)
const ESPN_TO_ODDS = {
  'eng.3': 'soccer_england_league1', 'eng.4': 'soccer_england_league2', 'esp.2': 'soccer_spain_segunda_division',
  'ger.2': 'soccer_germany_bundesliga2', 'ita.2': 'soccer_italy_serie_b', 'fra.2': 'soccer_france_ligue_two',
  'bel.1': 'soccer_belgium_first_div', 'sco.1': 'soccer_spl', 'tur.1': 'soccer_turkey_super_league', 'gre.1': 'soccer_greece_super_league',
  'aut.1': 'soccer_austria_bundesliga', 'sui.1': 'soccer_switzerland_superleague', 'den.1': 'soccer_denmark_superliga',
  'nor.1': 'soccer_norway_eliteserien', 'swe.1': 'soccer_sweden_allsvenskan', 'uefa.europa': 'soccer_uefa_europa_league',
  'uefa.europa.conf': 'soccer_uefa_europa_conference_league', 'conmebol.libertadores': 'soccer_conmebol_copa_libertadores',
  'arg.1': 'soccer_argentina_primera_division', 'usa.1': 'soccer_usa_mls', 'mex.1': 'soccer_mexico_ligamx',
  'jpn.1': 'soccer_japan_j_league', 'chn.1': 'soccer_china_superleague', 'aus.1': 'soccer_australia_aleague',
  'eng.fa': 'soccer_fa_cup', 'eng.league_cup': 'soccer_efl_cup',
};
const ymd = (ms) => new Date(ms).toISOString().slice(0, 10).replace(/-/g, '');
const extraIdOf = (espnId) => { // numeric + far above football-data ids so buttons/DB (BIGINT, \d+) keep working
  const n = Number(espnId);
  if (Number.isFinite(n) && n > 0) return 9e12 + n;
  return 9e12 + parseInt(crypto.createHash('md5').update(String(espnId)).digest('hex').slice(0, 8), 16);
};
const extraDiag = { at: 0, espnLeagues: 0, espnOk: 0, espnFail: 0, espnMissing: 0, oxCount: 0, oxErr: '', firstErr: '', espnCount: 0, sdbCount: 0, sdbErr: '', kept: 0, withOdds: 0 };
const FEED_HEADERS = { 'User-Agent': 'Mozilla/5.0 (compatible; FootballBot/1.0)', Accept: 'application/json' };
async function espnJson(slug, dateStr) { // tries a few URL shapes; ESPN rejects some date formats with HTTP 400
  const base = `https://site.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard`;
  const variants = [`${base}?dates=${dateStr}`, base, `https://site.web.api.espn.com/apis/site/v2/sports/soccer/${slug}/scoreboard?dates=${dateStr}`];
  let last = '';
  for (const url of variants) {
    try {
      const res = await fetchRetry(url, { headers: FEED_HEADERS }, 15000, 2);
      if (res.ok) return await res.json();
      last = `HTTP ${res.status}`;
      if (res.status === 404) { const e = new Error(last); e.missing = true; throw e; }
    } catch (e) { if (e.missing) throw e; last = (e.cause && e.cause.code) || e.message; }
  }
  throw new Error(last);
}
async function espnLeague(slug, today) {
  const dates = [ymd(Date.now())];
  if (/^(usa|mex|arg|bra|conmebol|concacaf)/.test(slug)) dates.push(ymd(Date.now() - 864e5)); // late Americas kickoffs fall on the previous US date
  const out = [];
  let answered = false;
  for (const d of dates) {
    let data;
    try { data = await espnJson(slug, d); }
    catch (e) {
      if (e.missing) { extraDiag.espnMissing++; return out; }
      if (!extraDiag.firstErr) extraDiag.firstErr = `${slug}: ${e.message}`;
      continue;
    }
    answered = true;
    for (const ev of data.events || []) {
      const st = ev.status && ev.status.type && ev.status.type.state;
      if (st && st !== 'pre') continue; // only matches that have not started
      const utc = new Date(ev.date).getTime();
      if (!Number.isFinite(utc) || localDate(new Date(utc)) !== today) continue;
      const comp = (ev.competitions || [])[0];
      const h = comp && (comp.competitors || []).find((c) => c.homeAway === 'home');
      const a = comp && (comp.competitors || []).find((c) => c.homeAway === 'away');
      if (!h || !a || !h.team || !a.team) continue;
      const id = extraIdOf(ev.id);
      if (out.some((x) => x.id === id)) continue;
      out.push({
        id, comp: 'X:' + slug, compName: ESPN_LEAGUES[slug] || slug, slug, extra: true,
        home: h.team.shortDisplayName || h.team.displayName, away: a.team.shortDisplayName || a.team.displayName,
        homeFull: h.team.displayName || h.team.shortDisplayName, awayFull: a.team.displayName || a.team.shortDisplayName,
        homeId: null, awayId: null, espnHomeId: h.team.id, espnAwayId: a.team.id, utc, odds: null,
      });
    }
  }
  if (answered) extraDiag.espnOk++; else extraDiag.espnFail++;
  return out;
}
// Second, independent source: TheSportsDB (free public key '3') lists soccer events of a whole day in ONE call.
async function sportsDbDay(dateStr, today) {
  try {
    let res = null;
    for (const key of ['3', '123']) { // both are TheSportsDB's public test keys
      res = await fetchRetry(`https://www.thesportsdb.com/api/v1/json/${key}/eventsday.php?d=${dateStr}&s=Soccer`, { headers: FEED_HEADERS }, 20000, 2);
      if (res.ok) break;
    }
    if (!res.ok) { extraDiag.sdbErr = 'HTTP ' + res.status; return []; }
    const data = await res.json();
    const out = [];
    for (const ev of data.events || []) {
      if (!ev.strHomeTeam || !ev.strAwayTeam) continue;
      const stt = String(ev.strStatus || '').toLowerCase();
      if (ev.intHomeScore != null && ev.intHomeScore !== '') continue; // already has a score
      if (stt && !['not started', 'ns', 'scheduled', 'time to be defined', 'tbd'].includes(stt)) continue;
      const iso = ev.strTimestamp ? ev.strTimestamp : `${ev.dateEvent}T${ev.strTime || '12:00:00'}`;
      const utc = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z').getTime();
      if (!Number.isFinite(utc) || localDate(new Date(utc)) !== today) continue;
      out.push({
        id: 8e12 + (Number(ev.idEvent) || 0), comp: 'X:sdb:' + (ev.idLeague || '0'), compName: ev.strLeague || 'Other football', slug: null, extra: true,
        home: ev.strHomeTeam, away: ev.strAwayTeam, homeFull: ev.strHomeTeam, awayFull: ev.strAwayTeam, homeId: null, awayId: null, utc, odds: null,
      });
    }
    return out;
  } catch (e) { extraDiag.sdbErr = (e.cause && e.cause.code) || e.message; return []; }
}
// ---------- Previous matches for extra-league teams (ESPN team schedule) ----------
const espnTeamFail = new Map();
async function espnTeamGames(slug, teamId) {
  if (!slug || !teamId) return [];
  const key = 'espnteam:' + slug + ':' + teamId;
  if ((espnTeamFail.get(key) || 0) > Date.now()) return [];
  try {
    return await cached(key, 12 * 3600e3, async () => {
      const res = await fetchRetry(`https://site.api.espn.com/apis/site/v2/sports/soccer/${slug}/teams/${teamId}/schedule`, { headers: FEED_HEADERS }, 12000, 2);
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      const games = [];
      for (const ev of data.events || []) {
        const comp = (ev.competitions || [])[0];
        if (!comp) continue;
        const stt = ((comp.status || ev.status || {}).type) || {};
        if (!(stt.completed === true || stt.state === 'post')) continue;
        const me = (comp.competitors || []).find((c) => String((c.team && c.team.id) || c.id) === String(teamId));
        const other = (comp.competitors || []).find((c) => c !== me);
        if (!me || !other) continue;
        const sc = (c) => Number(c.score && typeof c.score === 'object' ? (c.score.value != null ? c.score.value : c.score.displayValue) : c.score);
        const gf = sc(me), ga = sc(other);
        if (!Number.isFinite(gf) || !Number.isFinite(ga)) continue;
        games.push({ date: ev.date || comp.date, home: me.homeAway === 'home', gf, ga, opp: (other.team && (other.team.shortDisplayName || other.team.displayName)) || 'opponent' });
      }
      return games.sort((a, b) => new Date(b.date) - new Date(a.date));
    });
  } catch (e) {
    espnTeamFail.set(key, Date.now() + 30 * 60e3);
    console.log('Team history unavailable', slug, teamId, '-', e.message);
    return [];
  }
}
// Same shape as statAnalysis, built from each team's previous matches (no football-data season needed)
async function extraAnalysis(m) {
  const [hg, ag] = await Promise.all([espnTeamGames(m.slug, m.espnHomeId), espnTeamGames(m.slug, m.espnAwayId)]);
  let form = null;
  if (hg.length >= 3 && ag.length >= 3) {
    const avg = (arr, k) => arr.reduce((x, g) => x + g[k], 0) / arr.length;
    const L = (arr) => arr.slice(0, 10);
    const clampG = (x) => Math.max(0.3, Math.min(3.5, x));
    const lh = clampG(1.08 * (avg(L(hg), 'gf') + avg(L(ag), 'ga')) / 2);
    const la = clampG(0.93 * (avg(L(ag), 'gf') + avg(L(hg), 'ga')) / 2);
    form = { ...poisson1x2(lh, la), lh, la, rel: 0.6 * Math.min(1, Math.min(hg.length, ag.length) / 8) };
  }
  const h2h = hg.filter((g) => nameScore(g.opp, m.awayFull) >= 0.99).slice(0, 3)
    .map((g) => (g.home ? `${m.home} ${g.gf}-${g.ga} ${g.opp}` : `${g.opp} ${g.ga}-${g.gf} ${m.home}`) + ` (${fmtDate(g.date)})`);
  const mkt = m.odds ? m.odds.probs : null;
  const disagree = mkt && form ? 0.5 * ['H', 'D', 'A'].reduce((x, k) => x + Math.abs(mkt[k] - form[k]), 0) : null;
  return { hs: summarise(hg), as: summarise(ag), histH: hg.slice(0, 8), histA: ag.slice(0, 8), form, h2h, mkt, disagree, final: blend(mkt, form, null), ai: null };
}

// Third source: The Odds API "events" lists (free, no credits used) - covers every in-season soccer competition it prices.
async function oddsApiEventsDay(today) {
  if (!ODDS_KEY) return [];
  try {
    const res = await fetchRetry(`https://api.the-odds-api.com/v4/sports/?apiKey=${ODDS_KEY}`, { headers: FEED_HEADERS }, 20000, 2);
    if (!res.ok) { extraDiag.oxErr = 'sports HTTP ' + res.status; return []; }
    const sports = (await res.json()).filter((x) => String(x.key).startsWith('soccer_') && x.active && !x.has_outrights);
    const lists = await pool(sports, 5, async (sp) => {
      try {
        const r = await fetchRetry(`https://api.the-odds-api.com/v4/sports/${sp.key}/events?apiKey=${ODDS_KEY}&dateFormat=iso`, { headers: FEED_HEADERS }, 15000, 2);
        if (!r.ok) return [];
        return (await r.json()).map((ev) => ({ ev, sp }));
      } catch (e) { return []; }
    });
    const out = [];
    for (const { ev, sp } of lists.flat()) {
      const utc = new Date(ev.commence_time).getTime();
      if (!Number.isFinite(utc) || utc <= Date.now() || localDate(new Date(utc)) !== today) continue;
      out.push({
        id: 7e12 + parseInt(crypto.createHash('md5').update(String(ev.id)).digest('hex').slice(0, 8), 16),
        comp: 'X:odds:' + sp.key, compName: sp.title || sp.key, slug: null, oddsKey: sp.key, extra: true,
        home: ev.home_team, away: ev.away_team, homeFull: ev.home_team, awayFull: ev.away_team, homeId: null, awayId: null, utc, odds: null,
      });
    }
    return out;
  } catch (e) { extraDiag.oxErr = (e.cause && e.cause.code) || e.message; return []; }
}
async function getExtraFixtures(today, existing) {
  if (!EXTRA_ON) return [];
  const slugs = Object.keys(ESPN_LEAGUES);
  Object.assign(extraDiag, { at: Date.now(), espnLeagues: slugs.length, espnOk: 0, espnFail: 0, espnMissing: 0, oxCount: 0, oxErr: '', firstErr: '', sdbErr: '' });
  const lists = await pool(slugs, 5, (slug) => espnLeague(slug, today));
  let extras = lists.flat();
  extraDiag.espnCount = extras.length;
  const sdb = (await Promise.all([-1, 0, 1].map((d) => sportsDbDay(new Date(Date.now() + d * 864e5).toISOString().slice(0, 10), today)))).flat();
  extraDiag.sdbCount = sdb.length;
  extras = extras.concat(sdb);
  extraDiag.oxErr = '';
  const ox = await oddsApiEventsDay(today);
  extraDiag.oxCount = ox.length;
  extras = extras.concat(ox);
  // drop anything football-data already gave us (same teams, kickoff within 3h)
  const same = (a, b) => Math.abs(a.utc - b.utc) < 3 * 3600e3 && nameScore(a.homeFull, b.homeFull) >= 0.99 && nameScore(a.awayFull, b.awayFull) >= 0.99;
  const seen = [];
  extras = extras.filter((x) => {
    if (existing.some((f) => same(x, f))) return false;
    const k = seen.find((f) => same(x, f));
    if (k) { if (!k.oddsKey && x.oddsKey) k.oddsKey = x.oddsKey; return false; } // keep the first copy, borrow the odds key
    seen.push(x); return true;
  });
  // odds (bonus): only leagues that really play today, capped to protect your Odds API credits
  if (ODDS_KEY && EXTRA_ODDS_ON) {
    const keyOf = (x) => x.oddsKey || ESPN_TO_ODDS[x.slug];
    const keysToday = [...new Set(extras.map(keyOf).filter(Boolean))].slice(0, EXTRA_ODDS_MAX);
    for (const ok of keysToday) {
      try {
        const events = await cached('odds:' + ok, CACHE_MS, () => fetchOdds(ok));
        for (const f of extras.filter((x) => keyOf(x) === ok)) f.odds = findOdds({ home: f.homeFull, away: f.awayFull, utc: f.utc }, events);
      } catch (e) { console.log('Extra odds failed for', ok, e.message); }
    }
  }
  extraDiag.kept = extras.length; extraDiag.withOdds = extras.filter((x) => x.odds).length;
  console.log(`Extra fixtures today: ${extras.length} (${extraDiag.withOdds} with odds) | ESPN ${extraDiag.espnCount} (${extraDiag.espnOk}/${extraDiag.espnLeagues} leagues ok${extraDiag.firstErr ? ', first error: ' + extraDiag.firstErr : ''}) | TheSportsDB ${extraDiag.sdbCount}${extraDiag.sdbErr ? ' (' + extraDiag.sdbErr + ')' : ''} | OddsAPI events ${extraDiag.oxCount}${extraDiag.oxErr ? ' (' + extraDiag.oxErr + ')' : ''}`);
  return extras;
}

// ---------- Independent score feed: ESPN + TheSportsDB (football-data only as a back-up) ----------
// One normalised game shape: { src, league, home, away, homeFull, awayFull, hs, as, hs90, as90, status: NS|LIVE|HT|FT, minute, utc, aet }
// Used for the Live screen AND to settle pending predictions, so a slow / rate-limited football-data no longer freezes results.
const FD_TO_ESPN = { PL: 'eng.1', ELC: 'eng.2', BL1: 'ger.1', SA: 'ita.1', PD: 'esp.1', FL1: 'fra.1', DED: 'ned.1', PPL: 'por.1', CL: 'uefa.champions', BSA: 'bra.1' };
const AMERICAS_RE = /^(usa|mex|arg|bra|conmebol|concacaf|col|chi|uru|per|ecu)/;
const LIVE_USE_FD = process.env.LIVE_USE_FD === 'always'; // default: football-data is only asked when ESPN returned nothing
const raceTimeout = (p, ms, label = 'timeout') => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label)), ms))]);
const scoreNum = (v) => { if (v && typeof v === 'object') v = v.value != null ? v.value : v.displayValue; if (v === undefined || v === null || v === '') return null; const n = Number(v); return Number.isFinite(n) ? n : null; };

function espnGame(ev, slug) {
  const comp = (ev.competitions || [])[0];
  if (!comp) return null;
  const h = (comp.competitors || []).find((c) => c.homeAway === 'home');
  const a = (comp.competitors || []).find((c) => c.homeAway === 'away');
  if (!h || !a || !h.team || !a.team) return null;
  const stt = (ev.status && ev.status.type) || (comp.status && comp.status.type) || {};
  const nm = String(stt.name || ''), state = stt.state || 'pre';
  let status = 'NS';
  if (state === 'in') status = /HALFTIME/i.test(nm) ? 'HT' : 'LIVE';
  else if (state === 'post') status = stt.completed === false || /POSTPONED|CANCEL|ABANDON|SUSPEND|FORFEIT|DELAY/i.test(nm) ? 'X' : 'FT';
  if (status === 'X') return null;
  const hs = scoreNum(h.score), as = scoreNum(a.score);
  const aet = /AET|PEN|SHOOT|EXTRA/i.test(nm);
  let hs90 = null, as90 = null;
  if (status === 'FT') {
    if (!aet) { hs90 = hs; as90 = as; }
    else { // extra time / penalties: the 90-minute score is the first two periods
      const l1 = (h.linescores || []).map(scoreNum), l2 = (a.linescores || []).map(scoreNum);
      if (l1.length >= 2 && l2.length >= 2 && l1[0] != null && l1[1] != null && l2[0] != null && l2[1] != null) { hs90 = l1[0] + l1[1]; as90 = l2[0] + l2[1]; }
    }
  }
  const clock = String((ev.status && ev.status.displayClock) || '').trim();
  return {
    src: 'espn', slug, league: ESPN_LEAGUES[slug] || slug,
    home: h.team.shortDisplayName || h.team.displayName, away: a.team.shortDisplayName || a.team.displayName,
    homeFull: h.team.displayName || h.team.shortDisplayName, awayFull: a.team.displayName || a.team.shortDisplayName,
    hs, as, hs90, as90, status, minute: status === 'LIVE' && clock && clock !== "0'" ? clock : '', utc: new Date(ev.date).getTime(), aet,
  };
}
async function espnFeed(dates, budgetMs = 22000) { // dates: [{ s: 'YYYYMMDD', onlyAm }]; returns whatever finished inside the time budget
  const jobs = [];
  for (const slug of Object.keys(ESPN_LEAGUES)) for (const d of dates) { if (d.onlyAm && !AMERICAS_RE.test(slug)) continue; jobs.push([slug, d.s]); }
  const out = [], st = { ok: 0, fail: 0, firstErr: '' };
  const work = pool(jobs, 8, async ([slug, d]) => {
    try {
      const data = await espnJson(slug, d);
      st.ok++;
      for (const ev of data.events || []) { const g = espnGame(ev, slug); if (g && Number.isFinite(g.utc)) out.push(g); }
    } catch (e) { if (!e.missing) { st.fail++; if (!st.firstErr) st.firstErr = `${slug}: ${e.message}`; } }
  });
  await Promise.race([work, sleep(budgetMs)]);
  return { games: out.slice(), ...st, jobs: jobs.length };
}
async function sdbFeed(dayStr) { // TheSportsDB: every soccer league for a day in one call
  let res = null;
  for (const key of ['3', '123']) {
    res = await fetchRetry(`https://www.thesportsdb.com/api/v1/json/${key}/eventsday.php?d=${dayStr}&s=Soccer`, { headers: FEED_HEADERS }, 15000, 2);
    if (res.ok) break;
  }
  if (!res || !res.ok) throw new Error('HTTP ' + (res && res.status));
  const data = await res.json();
  const out = [], now = Date.now();
  for (const ev of data.events || []) {
    if (!ev.strHomeTeam || !ev.strAwayTeam) continue;
    const s = String(ev.strStatus || '').trim().toLowerCase();
    if (/postp|cancel|abandon|susp|awarded/.test(s)) continue;
    const iso = ev.strTimestamp ? ev.strTimestamp : `${ev.dateEvent}T${ev.strTime || '12:00:00'}`;
    const utc = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? iso : iso + 'Z').getTime();
    if (!Number.isFinite(utc)) continue;
    const hs = scoreNum(ev.intHomeScore), as = scoreNum(ev.intAwayScore);
    let status = 'NS';
    if (/^(ft|match finished|finished|aet|aot|pen|after extra|after pen)/.test(s)) status = 'FT';
    else if (/^(ht|half ?time)/.test(s)) status = 'HT';
    else if (/^(1h|2h|et|live|in play|\d)/.test(s) || (hs != null && as != null && !/^(ns|not started)/.test(s))) status = now - utc > 3.5 * 3600e3 ? 'FT' : 'LIVE';
    const aet = /aet|aot|pen|extra/.test(s);
    out.push({
      src: 'sdb', slug: null, league: ev.strLeague || 'Other football', home: ev.strHomeTeam, away: ev.strAwayTeam, homeFull: ev.strHomeTeam, awayFull: ev.strAwayTeam,
      hs, as, hs90: status === 'FT' && !aet ? hs : null, as90: status === 'FT' && !aet ? as : null, status, minute: '', utc, aet,
    });
  }
  return out;
}
async function fdFeed() { // back-up only: shares the 10 requests/minute football-data budget
  const from = localDate(new Date(Date.now() - 864e5)), to = localDate(new Date(Date.now() + 864e5));
  const data = await fdGet(`/matches?dateFrom=${from}&dateTo=${to}`);
  return (data.matches || []).map((m) => {
    const sc = (m.score && m.score.fullTime) || {};
    const status = ['IN_PLAY', 'LIVE'].includes(m.status) ? 'LIVE' : m.status === 'PAUSED' ? 'HT' : m.status === 'FINISHED' ? 'FT' : 'NS';
    const reg = (m.score && m.score.regularTime) || sc;
    return {
      src: 'fd', slug: null, league: m.competition.name, home: m.homeTeam.shortName || m.homeTeam.name, away: m.awayTeam.shortName || m.awayTeam.name,
      homeFull: m.homeTeam.name, awayFull: m.awayTeam.name, hs: sc.home == null ? null : sc.home, as: sc.away == null ? null : sc.away,
      hs90: status === 'FT' && reg.home != null ? reg.home : null, as90: status === 'FT' && reg.away != null ? reg.away : null,
      status, minute: '', utc: new Date(m.utcDate).getTime(), aet: false,
    };
  });
}
const gameSame = (a, b) => Math.abs(a.utc - b.utc) < 3 * 3600e3 && nameScore(a.homeFull, b.homeFull) >= 0.99 && nameScore(a.awayFull, b.awayFull) >= 0.99;
function mergeGames(lists) { // lists in priority order; for a duplicate keep the most advanced status (NS < LIVE/HT < FT)
  const RANK = { NS: 0, LIVE: 1, HT: 1, FT: 2 };
  const out = [];
  for (const list of lists) for (const g of list) {
    const ex = out.find((x) => gameSame(x, g));
    if (!ex) { out.push({ ...g }); continue; }
    if (RANK[g.status] > RANK[ex.status]) Object.assign(ex, { hs: g.hs, as: g.as, hs90: g.hs90, as90: g.as90, status: g.status, minute: g.minute, aet: g.aet });
    else if (RANK[g.status] === RANK[ex.status] && ex.status !== 'NS' && (ex.hs == null || ex.as == null) && g.hs != null) Object.assign(ex, { hs: g.hs, as: g.as });
  }
  return out;
}
const feedDiag = { at: 0, espn: '-', sdb: '-', fd: '-' };
function getGames() {
  return cached('feed:games', LIVE_CACHE_MS, async () => {
    const hr = new Date().getUTCHours();
    const dates = [{ s: ymd(Date.now()), onlyAm: false }, { s: ymd(Date.now() - 864e5), onlyAm: hr >= 5 }]; // early UTC hours: yesterday's late games are still running everywhere
    const sdbDays = [0, -1].map((d) => new Date(Date.now() + d * 864e5).toISOString().slice(0, 10));
    const [e, s] = await Promise.allSettled([espnFeed(dates), Promise.all(sdbDays.map(sdbFeed)).then((a) => a.flat())]);
    const lists = [], ok = {};
    if (e.status === 'fulfilled') { lists.push(e.value.games); ok.espn = e.value.ok > 0; feedDiag.espn = `${e.value.games.length} games, ${e.value.ok}/${e.value.jobs} requests ok${e.value.firstErr ? ', first error ' + e.value.firstErr : ''}`; }
    else { ok.espn = false; feedDiag.espn = 'failed: ' + e.reason.message; }
    if (s.status === 'fulfilled') { lists.push(s.value); ok.sdb = true; feedDiag.sdb = `${s.value.length} games`; }
    else { ok.sdb = false; feedDiag.sdb = 'failed: ' + s.reason.message; }
    const espnN = e.status === 'fulfilled' ? e.value.games.length : 0;
    if (LIVE_USE_FD || !espnN) {
      try { const f = await raceTimeout(fdFeed(), 9000); lists.push(f); ok.fd = true; feedDiag.fd = `${f.length} games`; }
      catch (err) { ok.fd = false; feedDiag.fd = 'failed: ' + err.message; }
    } else { ok.fd = null; feedDiag.fd = 'not needed (ESPN answered)'; }
    feedDiag.at = Date.now();
    if (!lists.length) throw new Error('no score source reachable');
    return { at: Date.now(), games: mergeGames(lists), ok };
  });
}
async function resolveFromFeed(open) { // settle open predictions from ESPN / TheSportsDB finished games
  const now = Date.now();
  const dayOf = (ms) => new Date(ms).toISOString().slice(0, 10);
  const espnNeeds = new Map(), sdbDays = new Set();
  for (const r of open) {
    const slug = String(r.comp).startsWith('X:') ? (ESPN_LEAGUES[String(r.comp).slice(2)] ? String(r.comp).slice(2) : null) : FD_TO_ESPN[r.comp] || null;
    const days = [...new Set([dayOf(r.utc), dayOf(r.utc + 6 * 3600e3)])];
    for (const d of days) {
      if (slug) espnNeeds.set(slug + '|' + d, [slug, d]);
      if (!slug || now - r.utc > 8 * 3600e3) sdbDays.add(d); // second opinion for anything that is still open after 8 hours
    }
  }
  const games = [];
  await pool([...espnNeeds.values()], 6, async ([slug, d]) => {
    try {
      const list = await cached(`feedres:espn:${slug}:${d}`, 15 * 60e3, async () => (((await espnJson(slug, d.replace(/-/g, ''))).events || []).map((ev) => espnGame(ev, slug)).filter(Boolean)));
      games.push(...list);
    } catch (e) { if (!e.missing) console.log('Result feed (ESPN) failed', slug, d, e.message); }
  });
  await pool([...sdbDays], 3, async (d) => {
    try { games.push(...(await cached(`feedres:sdb:${d}`, 15 * 60e3, () => sdbFeed(d)))); }
    catch (e) { console.log('Result feed (TheSportsDB) failed', d, e.message); }
  });
  const sameTeams = (r, g) => (nameScore(r.home, g.home) >= 0.99 || nameScore(r.home, g.homeFull) >= 0.99) && (nameScore(r.away, g.away) >= 0.99 || nameScore(r.away, g.awayFull) >= 0.99);
  const changed = [];
  for (const r of open) {
    const hits = games.filter((g) => g.status === 'FT' && g.hs90 != null && g.as90 != null && Math.abs(g.utc - r.utc) < 8 * 3600e3 && sameTeams(r, g));
    if (!hits.length) continue;
    const g = hits.find((x) => x.src === 'espn') || hits[0];
    r.res = g.hs90 > g.as90 ? 'H' : g.hs90 < g.as90 ? 'A' : 'D';
    r.score = `${g.hs90}-${g.as90}`;
    predDirty.add(r.id);
    changed.push(r);
  }
  return changed;
}

// ---------- Today's matches ----------
let nextInfo = null; // next match day after today (shown when today is empty)
function getMatchesToday() {
  const key = 'today:' + localDate(new Date());
  return getMatchesTodayCached(key).then((list) => {
    if (!list.length) cache.delete(key); // never cache an empty day, so the next /today re-checks the API
    return list;
  });
}
function getMatchesTodayCached(key) {
  return cached(key, CACHE_MS, async () => {
    const day = 864e5;
    const from = localDate(new Date(Date.now() - day));
    const to = localDate(new Date(Date.now() + 2 * day));
    const data = await fdGet(`/matches?dateFrom=${from}&dateTo=${to}`);
    const today = localDate(new Date());
    const fixtures = (data.matches || [])
      .filter((m) => ['SCHEDULED', 'TIMED'].includes(m.status) && localDate(new Date(m.utcDate)) === today)
      .map((m) => ({
        id: m.id,
        comp: m.competition.code,
        compName: m.competition.name,
        home: m.homeTeam.shortName || m.homeTeam.name,
        away: m.awayTeam.shortName || m.awayTeam.name,
        homeFull: m.homeTeam.name,
        awayFull: m.awayTeam.name,
        homeId: m.homeTeam.id,
        awayId: m.awayTeam.id,
        utc: new Date(m.utcDate).getTime(),
        odds: null,
      }));

    const upcoming = (data.matches || [])
      .filter((m) => ['SCHEDULED', 'TIMED'].includes(m.status) && localDate(new Date(m.utcDate)) > today)
      .sort((a, b) => new Date(a.utcDate) - new Date(b.utcDate));
    if (upcoming.length) {
      const d0 = localDate(new Date(upcoming[0].utcDate));
      nextInfo = { date: fmtDate(upcoming[0].utcDate), n: upcoming.filter((m) => localDate(new Date(m.utcDate)) === d0).length };
    } else nextInfo = null;

    // odds: one call per competition that actually plays today
    const comps = [...new Set(fixtures.map((f) => f.comp))].filter((c) => COMP_TO_ODDS[c]);
    for (const c of comps) {
      try {
        const events = await cached('odds:' + c, CACHE_MS, () => fetchOdds(COMP_TO_ODDS[c]));
        for (const f of fixtures.filter((x) => x.comp === c)) {
          f.odds = findOdds({ home: f.homeFull, away: f.awayFull, utc: f.utc }, events);
        }
      } catch (e) {
        console.log('Odds fetch failed for', c, e.message);
      }
    }
    try { fixtures.push(...(await getExtraFixtures(today, fixtures))); } catch (e) { console.log('Extra fixtures failed', e.message); }
    fixtures.sort((a, b) => a.compName.localeCompare(b.compName) || a.utc - b.utc);
    return fixtures;
  });
}

// ---------- Form / history model ----------
function leagueModel(matches) {
  let hg = 0, ag = 0, n = 0;
  for (const m of matches) {
    const s = m.score && m.score.fullTime;
    if (!s || s.home == null) continue;
    hg += s.home;
    ag += s.away;
    n++;
  }
  return { n, avgH: n ? hg / n : 1.45, avgA: n ? ag / n : 1.15 };
}

function teamGames(matches, id) {
  const games = [];
  for (const m of matches) {
    const s = m.score && m.score.fullTime;
    if (!s || s.home == null) continue;
    if (m.homeTeam.id === id)
      games.push({ date: m.utcDate, home: true, gf: s.home, ga: s.away, opp: m.awayTeam.shortName || m.awayTeam.name });
    else if (m.awayTeam.id === id)
      games.push({ date: m.utcDate, home: false, gf: s.away, ga: s.home, opp: m.homeTeam.shortName || m.homeTeam.name });
  }
  return games.sort((a, b) => new Date(b.date) - new Date(a.date));
}

function summarise(games) {
  const last = games.slice(0, 5);
  let pts = 0;
  const form = last.map((g) => {
    if (g.gf > g.ga) { pts += 3; return 'W'; }
    if (g.gf === g.ga) { pts += 1; return 'D'; }
    return 'L';
  });
  const gf = last.reduce((s, g) => s + g.gf, 0);
  const ga = last.reduce((s, g) => s + g.ga, 0);
  return {
    games: games.length,
    form: form.join(' ') || 'n/a',
    ppg: last.length ? pts / last.length : null,
    gf, ga,
    recent: last.slice(0, 3).map((g) => `${g.home ? 'H' : 'A'} ${g.opp} ${g.gf}-${g.ga}`),
  };
}

// Opponent-adjusted attack/defence ratings fitted on the whole season, recent games weighted more.
// (A team that scored 3 against weak defences is no longer rated like one that scored 3 against strong ones.)
function fitRatings(season, prev = []) {
  const now = Date.now();
  const games = [];
  let sw = 0, shg = 0, sag = 0;
  const addGame = (m, scale, halfLife) => {
    const sc = m.score && m.score.fullTime;
    if (!sc || sc.home == null) return;
    const hg = Math.min(sc.home, GOAL_CAP), ag = Math.min(sc.away, GOAL_CAP);
    const w = scale * Math.pow(0.5, Math.max(0, now - new Date(m.utcDate).getTime()) / 864e5 / halfLife);
    games.push({ h: m.homeTeam.id, a: m.awayTeam.id, hg, ag, w });
    sw += w; shg += w * hg; sag += w * ag;
  };
  for (const m of season) addGame(m, 1, FORM_HALF_LIFE_DAYS);
  const curN = games.length;
  const prevScale = PREV_WEIGHT * Math.max(0.15, 1 - curN / 250); // last season matters most early in the season
  for (const m of prev) addGame(m, prevScale, FORM_HALF_LIFE_DAYS * 4);
  if (games.length < 10) return null;
  const avgH = shg / sw, avgA = sag / sw;
  const ids = new Set();
  games.forEach((g) => { ids.add(g.h); ids.add(g.a); });
  const atk = new Map([...ids].map((i) => [i, 1]));
  const def = new Map([...ids].map((i) => [i, 1]));
  const K = 4; // shrinkage: K "pseudo expected goals" at league-average strength
  for (let it = 0; it < 30; it++) {
    const nA = new Map(), dA = new Map(), nD = new Map(), dD = new Map();
    const add = (m, k, v) => m.set(k, (m.get(k) || 0) + v);
    for (const g of games) {
      add(nA, g.h, g.w * g.hg); add(dA, g.h, g.w * avgH * def.get(g.a));
      add(nA, g.a, g.w * g.ag); add(dA, g.a, g.w * avgA * def.get(g.h));
      add(nD, g.h, g.w * g.ag); add(dD, g.h, g.w * avgA * atk.get(g.a));
      add(nD, g.a, g.w * g.hg); add(dD, g.a, g.w * avgH * atk.get(g.h));
    }
    let ma = 0, md = 0;
    for (const i of ids) {
      atk.set(i, ((nA.get(i) || 0) + K) / ((dA.get(i) || 0) + K));
      def.set(i, ((nD.get(i) || 0) + K) / ((dD.get(i) || 0) + K));
      ma += atk.get(i); md += def.get(i);
    }
    ma /= ids.size; md /= ids.size; // keep the scale identifiable
    for (const i of ids) { atk.set(i, atk.get(i) / ma); def.set(i, def.get(i) / md); }
  }
  return { avgH, avgA, atk, def };
}

function expectedGoals(rt, homeId, awayId) {
  const g = (mp, id) => (mp.has(id) ? mp.get(id) : 1);
  return {
    lh: rt.avgH * g(rt.atk, homeId) * g(rt.def, awayId),
    la: rt.avgA * g(rt.atk, awayId) * g(rt.def, homeId),
  };
}

// Score-probability grid: Poisson with the Dixon-Coles correction for 0-0 / 1-0 / 0-1 / 1-1
function scoreGrid(lh, la, rho = DC_RHO) {
  const N = 10;
  const fact = [1];
  for (let i = 1; i <= N; i++) fact[i] = fact[i - 1] * i;
  const pm = (l, k) => (Math.exp(-l) * Math.pow(l, k)) / fact[k];
  const grid = [];
  let tot = 0;
  for (let i = 0; i <= N; i++) {
    grid[i] = [];
    for (let j = 0; j <= N; j++) {
      let p = pm(lh, i) * pm(la, j);
      if (i === 0 && j === 0) p *= 1 - lh * la * rho;
      else if (i === 0 && j === 1) p *= 1 + lh * rho;
      else if (i === 1 && j === 0) p *= 1 + la * rho;
      else if (i === 1 && j === 1) p *= 1 - rho;
      p = Math.max(p, 0);
      grid[i][j] = p;
      tot += p;
    }
  }
  for (const row of grid) for (let j = 0; j < row.length; j++) row[j] /= tot;
  return grid;
}

function poisson1x2(lh, la) {
  const grid = scoreGrid(lh, la);
  let H = 0, D = 0, A = 0;
  grid.forEach((row, i) => row.forEach((p, j) => { if (i > j) H += p; else if (i === j) D += p; else A += p; }));
  return { H, D, A };
}

// Keep the LLM on a leash: it may nudge the market, never overrule it.
function clampToMarket(ai, mkt) {
  const out = {};
  let s = 0;
  for (const k of ['H', 'D', 'A']) {
    out[k] = Math.max(mkt[k] - AI_MAX_DEV, Math.min(mkt[k] + AI_MAX_DEV, ai[k]));
    s += out[k];
  }
  for (const k of ['H', 'D', 'A']) out[k] /= s;
  return out;
}

// Log-linear (geometric) opinion pool: a source only moves the result where it is confident and the others are not,
// which beats a plain average when sources are correlated. Form weight shrinks when data is thin.
function blend(mkt, form, ai) {
  const parts = [];
  if (mkt) parts.push([mkt, W_MKT]);
  if (form) parts.push([form, W_FORM * (form.rel == null ? 1 : form.rel)]);
  if (ai) parts.push([mkt ? clampToMarket(ai, mkt) : ai, W_AI]);
  if (!parts.length) return null;
  const w = parts.reduce((s, p) => s + p[1], 0);
  const out = { H: 0, D: 0, A: 0 };
  for (const [p, wt] of parts) for (const k of ['H', 'D', 'A']) out[k] += (wt / w) * Math.log(Math.max(p[k], 1e-6));
  let t = 0;
  for (const k of ['H', 'D', 'A']) { out[k] = Math.exp(out[k]); t += out[k]; }
  for (const k of ['H', 'D', 'A']) out[k] /= t;
  return out;
}

// ---------- AI model (online, OpenAI-compatible) ----------
const deadModels = new Map(); // model -> time until we stop trying it
let aiPausedUntil = 0; // set on HTTP 402 (credits used up) so we stop hammering the API

// ---- Automatic model discovery: asks the provider which models exist and picks the best chat models ----
const NOT_CHAT = /whisper|guard|tts|embed|moderat|image|imagen|veo|audio|speech|transcri|orpheus|playai|safeguard|aqa|rerank|dall|vision-preview|live|robotics|gpt-oss|qwen3|deepseek-r1|-r1|reason|think/i; // last group: reasoning models burn the token budget / pollute the JSON
function modelScore(id) {
  const x = id.toLowerCase();
  const size = x.match(/(\d+(?:\.\d+)?)b\b/);
  let sc = size ? Math.min(Number(size[1]), 120) : 30;               // bigger = smarter (capped so huge models do not hog the free tier)
  if (/gemini/.test(x)) sc = /flash/.test(x) ? 60 : 20;               // Gemini: prefer the flash models
  if (/lite|mini|nano|tiny/.test(x)) sc -= 10;
  if (/instruct|versatile|chat|flash|turbo/.test(x)) sc += 5;
  if (/preview|exp|beta|deprecated/.test(x)) sc -= 15;               // unstable models last
  return sc;
}
let lastDiscover = 0;
async function discoverModels() {
  if (!AI_ON || Date.now() - lastDiscover < 10 * 60e3) return false;
  lastDiscover = Date.now();
  try {
    const res = await fetch(`${AI_BASE_URL}/models`, {
      headers: AI_USE_KEY ? { Authorization: `Bearer ${AI_USE_KEY}` } : {},
      signal: AbortSignal.timeout(20000),
    });
    if (!res.ok) {
      console.log(`Model discovery failed: HTTP ${res.status} from ${AI_BASE_URL}/models` + (res.status === 404 ? ' (is AI_BASE_URL / AI_PROVIDER right?)' : res.status === 401 ? ' (API key rejected)' : ''));
      return false;
    }
    const j = await res.json();
    let ids = (Array.isArray(j) ? j : j.data || j.models || [])
      .map((x) => String(x.id || x.name || '').replace(/^models\//, ''))
      .filter((id) => id && !NOT_CHAT.test(id));
    if (PROVIDER === 'openrouter') ids = ids.filter((id) => id.endsWith(':free'));
    ids.sort((a, b) => modelScore(b) - modelScore(a));
    const picked = ids.slice(0, 3);
    if (!picked.length) { console.log('Model discovery: provider returned no usable chat models.'); return false; }
    HF_MODELS.splice(0, HF_MODELS.length, ...picked);
    console.log(`AI models auto-selected from ${AI_BASE_URL}: ${picked.join(', ')}`);
    return true;
  } catch (e) {
    console.log('Model discovery failed:', (e.cause && e.cause.code) || e.message);
    return false;
  }
}
let discoveryDone = Promise.resolve();

async function hfPredict(m, st) {
  if (!AI_ON || Date.now() < aiPausedUntil) return null;
  await discoveryDone;
  const line = (name, s) =>
    `${name}: last5 ${s.form} (${s.ppg == null ? 'n/a' : s.ppg.toFixed(1)} pts/game, ${s.gf} scored, ${s.ga} conceded); recent: ${s.recent.join(', ') || 'n/a'}`;
  const gTxt = (g) => `${g.home ? 'home' : 'away'} vs ${g.opp} ${g.gf}-${g.ga} (${fmtDate(g.date)})`;
  const hist = (name, arr, s, isHome) => {
    if (!arr || !arr.length) return line(name, s);
    const w = arr.filter((g) => g.gf > g.ga).length, d = arr.filter((g) => g.gf === g.ga).length, l = arr.length - w - d;
    const gf = arr.reduce((x, g) => x + g.gf, 0) / arr.length, ga = arr.reduce((x, g) => x + g.ga, 0) / arr.length;
    const v = arr.filter((g) => g.home === isHome);
    const vt = v.length ? ` ${isHome ? 'At home' : 'Away'}: ${v.filter((g) => g.gf > g.ga).length}W ${v.filter((g) => g.gf === g.ga).length}D ${v.filter((g) => g.gf < g.ga).length}L in ${v.length}.` : '';
    return `${name} previous ${arr.length} matches (newest first): ${arr.map(gTxt).join('; ')}. Record ${w}W ${d}D ${l}L, average ${gf.toFixed(1)} scored / ${ga.toFixed(1)} conceded.${vt}`;
  };
  const prompt = [
    `Match: ${m.homeFull} (home) vs ${m.awayFull} (away) — ${m.compName}`,
    m.odds
      ? `Bookmaker average odds: home ${m.odds.odds.H.toFixed(2)}, draw ${m.odds.odds.D.toFixed(2)}, away ${m.odds.odds.A.toFixed(2)}. Margin-free market chances: home ${pct(m.odds.probs.H)}%, draw ${pct(m.odds.probs.D)}%, away ${pct(m.odds.probs.A)}%.`
      : 'No odds available.',
    hist(m.homeFull, st.histH, st.hs, true),
    hist(m.awayFull, st.histA, st.as, false),
    st.form ? `Statistical goal model expects ${st.form.lh.toFixed(2)}-${st.form.la.toFixed(2)} goals.` : '',
    st.h2h.length ? `Head-to-head this season: ${st.h2h.join('; ')}` : '',
    st.extra || '',
    'Estimate the probabilities of home win, draw and away win. Respond with ONLY JSON like {"home":45,"draw":28,"away":27,"reason":"max 25 words"}. Percentages must sum to 100.',
  ].filter(Boolean).join('\n');

  for (const model of HF_MODELS) {
    if ((deadModels.get(model) || 0) > Date.now()) continue;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const res = await fetch(`${AI_BASE_URL}/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(AI_USE_KEY ? { Authorization: `Bearer ${AI_USE_KEY}` } : {}) },
          body: JSON.stringify({
            model,
            max_tokens: 250,
            temperature: 0.2,
            messages: [
              { role: 'system', content: 'You are a careful football analyst. Base your estimate mainly on the previous matches of both teams given below (recent results, goals scored and conceded, home/away record, opponents faced) and compare it with the bookmaker prices. Use only the data provided. Never invent injuries, lineups or news. Reply with JSON only.' },
              { role: 'user', content: prompt },
            ],
          }),
          signal: AbortSignal.timeout(AI_TIMEOUT_MS),
        });
        if (!res.ok) {
          const err = new Error('HTTP ' + res.status);
          err.status = res.status;
          throw err;
        }
        const j = await res.json();
        const txt = (j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || '';
        const o = JSON.parse((txt.match(/\{[\s\S]*\}/) || [])[0]);
        const h = Number(o.home), d = Number(o.draw), a = Number(o.away);
        if (![h, d, a].every((x) => Number.isFinite(x) && x >= 0) || h + d + a <= 0) throw new Error('bad numbers');
        const sum = h + d + a;
        return { probs: { H: h / sum, D: d / sum, A: a / sum }, reason: String(o.reason || '').slice(0, 200), model };
      } catch (e) {
        const code = (e.cause && e.cause.code) || '';
        if (['EAI_AGAIN', 'ENOTFOUND', 'ECONNREFUSED', 'ETIMEDOUT', 'UND_ERR_CONNECT_TIMEOUT'].includes(code)) {
          aiPausedUntil = Date.now() + 5 * 60e3;
          console.log(
            code === 'ECONNREFUSED'
              ? `AI server not running at ${AI_BASE_URL} (is the AI host/Ollama up?). AI paused 5 min; using odds + form model.`
              : `Cannot reach AI host ${AI_BASE_URL} (${code} = DNS/network problem on this PC). AI paused 5 min; using odds + form model.`
          );
          return null;
        }
        if (e.status === 402) {
          const alreadyPaused = Date.now() < aiPausedUntil;
          aiPausedUntil = Date.now() + 6 * 3600e3;
          if (!alreadyPaused) console.log(`AI provider "${PROVIDER}" has no credits left (HTTP 402). AI paused for 6h; continuing with odds + form model. Add another key (GROQ_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY) or set AI_PROVIDER to switch.`);
          return null;
        }
        if (e.status === 429 && attempt < 2) {
          await sleep(16000); // free anonymous tiers allow roughly 1 request / 15s
          continue;
        }
        if ([400, 401, 403, 404].includes(e.status)) {
          deadModels.set(model, Date.now() + 24 * 3600e3);
          console.log(`AI model ${model} rejected (HTTP ${e.status}) - disabled for 24h.`);
          if ([400, 404].includes(e.status)) discoverModels().catch(() => {}); // model renamed/retired -> pick current ones automatically
        } else if (e.status === 429) {
          deadModels.set(model, Date.now() + 10 * 60e3);
        } else {
          console.log(`AI model ${model} failed:`, (e.cause && e.cause.code) || e.message);
        }
        break; // next model
      }
    }
  }
  return null;
}

// ---------- Analysis pipeline ----------
function statAnalysis(m) {
  return cached('stat:' + m.id, CACHE_MS, async () => {
    if (m.extra) return extraAnalysis(m);
    let season = [];
    try {
      season = await getSeason(m.comp);
    } catch (e) {
      console.log('Season fetch failed', m.comp, e.message);
    }
    const hg = teamGames(season, m.homeId);
    const ag = teamGames(season, m.awayId);
    let form = null;
    const prev = await getPrevSeason(m.comp);
    const rt = fitRatings(season, prev);
    if (rt && rt.atk.has(m.homeId) && rt.atk.has(m.awayId) && (prev.length || (hg.length >= 3 && ag.length >= 3))) {
      const { lh, la } = expectedGoals(rt, m.homeId, m.awayId);
      // rel: the fewer games we have on either side, the less the goal model is trusted in the blend
      const rel = Math.min(1, (Math.min(hg.length, ag.length) + (prev.length ? 6 : 0)) / 8);
      form = { ...poisson1x2(lh, la), lh, la, rel };
    }
    const h2h = season
      .filter((x) => (x.homeTeam.id === m.homeId && x.awayTeam.id === m.awayId) || (x.homeTeam.id === m.awayId && x.awayTeam.id === m.homeId))
      .map((x) => `${x.homeTeam.shortName || x.homeTeam.name} ${x.score.fullTime.home}-${x.score.fullTime.away} ${x.awayTeam.shortName || x.awayTeam.name} (${fmtDate(x.utcDate)})`)
      .slice(0, 3);
    const mkt = m.odds ? m.odds.probs : null;
    const disagree = mkt && form ? 0.5 * ['H', 'D', 'A'].reduce((s, k) => s + Math.abs(mkt[k] - form[k]), 0) : null; // total-variation distance model vs market
    return { hs: summarise(hg), as: summarise(ag), histH: hg.slice(0, 8), histA: ag.slice(0, 8), form, h2h, mkt, disagree, final: blend(mkt, form, null), ai: null };
  });
}

// One AI call per match, shared by /top5, /best3 and the detail view; concurrent requests are de-duplicated.
const aiInflight = new Map();
function aiRefine(m, st) {
  if (m.extra && !m.odds && !(st.hs && st.hs.games >= 3)) return Promise.resolve(st); // nothing real to base an AI guess on
  const key = 'ai:' + m.id;
  const c = cache.get(key);
  if (c && c.exp > Date.now()) return Promise.resolve(c.val);
  if (aiInflight.has(key)) return aiInflight.get(key);
  const p = hfPredict(m, st)
    .then((ai) => {
      const val = { ...st, ai, final: blend(st.mkt, st.form, ai && ai.probs) };
      if (ai) cache.set(key, { val, exp: Date.now() + CACHE_MS }); // only cache real AI answers
      return val;
    })
    .finally(() => aiInflight.delete(key));
  aiInflight.set(key, p);
  return p;
}

function topPick(m, final) {
  if (!final) return null;
  const [k, v] = Object.entries(final).sort((a, b) => b[1] - a[1])[0];
  const label = { H: `${m.home} win`, D: 'Draw', A: `${m.away} win` }[k];
  const sign = { H: '1', D: 'X', A: '2' }[k];
  const odds = m.odds ? m.odds.odds[k] : null;
  return { k, p: v, label, sign, odds };
}

function reasonOf(m, st) {
  if (st.ai && st.ai.reason) return st.ai.reason;
  if (st.form) return `Goal model expects ${st.form.lh.toFixed(1)}–${st.form.la.toFixed(1)}; blended with market odds.`;
  if (st.mkt) return 'Based on bookmaker market consensus.';
  return '';
}

// Ranked list shared by Top 5 (premium) and Best 10 (free): 5 premium picks + the next 10.
async function computeRanked() {
  const key = 'top5:' + localDate(new Date());
  const c = cache.get(key);
  if (c && c.exp > Date.now()) return c.val;
  if (c && c.pending) return c.pending;
  const pending = computeTop5Raw();
  cache.set(key, { pending, exp: 0 });
  try {
    const val = await pending;
    const aiOk = !AI_ON || val.some((x) => x.st.ai);
    cache.set(key, { val, exp: Date.now() + (aiOk ? CACHE_MS : 10 * 60e3) });
    return val;
  } catch (e) {
    cache.delete(key);
    throw e;
  }
}

async function computeTop5Raw() {
  const now = Date.now();
  const matches = (await getMatchesToday()).filter((m) => m.odds && m.utc > now);
  const withStats = [];
  for (const m of matches) withStats.push({ m, st: await deepAnalysis(m) });
  const rank = (arr) =>
    arr.map((x) => ({ ...x, sc: pickScore(x.m, x.st) })).filter((x) => x.sc).sort((a, b) => b.sc.score - a.sc.score);
  const cands = rank(withStats).slice(0, MAX_AI_MATCHES);
  const refined = await pool(cands, AI_CONCURRENCY, async (x) => ({ m: x.m, st: await aiRefine(x.m, x.st).catch(() => x.st) }));
  withStats.forEach((x) => logPrediction(x.m, x.st));
  refined.forEach((x) => logPrediction(x.m, x.st));
  const top = rank(refined).slice(0, 15);
  top.forEach((x, i) => logPrediction(x.m, x.st, i < 5 ? 'top5' : 'best10'));
  return top;
}
const computeTop5 = async () => (await computeRanked()).slice(0, 5);
const computeBest10 = async () => (await computeRanked()).slice(BEST10_INCLUDES_TOP5 ? 0 : 5, BEST10_INCLUDES_TOP5 ? 10 : 15);

// ---------- Best 3 : under-the-radar picks (never top-table or "popular" clubs) ----------
// Uses the season data already downloaded for the other features, so it costs NO extra football-data requests.
const DOMESTIC = new Set(['PL', 'ELC', 'BL1', 'SA', 'PD', 'FL1', 'DED', 'PPL', 'BSA']); // leagues where a table rank is meaningful
const f1 = (x, d = 1) => (x == null ? 'n/a' : x.toFixed(d));
const pctOrNA = (x) => (x == null ? 'n/a' : pct(x) + '%');
const argmax = (o) => ['H', 'D', 'A'].reduce((a, b) => (o[b] > o[a] ? b : a)); // only H/D/A (the form object also carries lh/la)

function isBigClub(...names) {
  // a name only counts if it carries at least as many words as the big-club entry (stops "Paris FC" matching "Paris Saint Germain")
  return names.filter(Boolean).some((n) => BIG_CLUBS.some((b) => nameScore(n, b) >= 1 && (toks(b).length === 1 || toks(n).length >= toks(b).length)));
}

function buildTable(season) {
  const t = new Map();
  const row = (id) => {
    if (!t.has(id)) t.set(id, { id, p: 0, pts: 0, gf: 0, ga: 0 });
    return t.get(id);
  };
  for (const m of season) {
    const s = m.score && m.score.fullTime;
    if (!s || s.home == null) continue;
    const h = row(m.homeTeam.id), a = row(m.awayTeam.id);
    h.p++; a.p++;
    h.gf += s.home; h.ga += s.away; a.gf += s.away; a.ga += s.home;
    if (s.home > s.away) h.pts += 3;
    else if (s.home < s.away) a.pts += 3;
    else { h.pts++; a.pts++; }
  }
  const rows = [...t.values()].sort((x, y) => y.pts - x.pts || (y.gf - y.ga) - (x.gf - x.ga) || y.gf - x.gf);
  rows.forEach((r, i) => { r.rank = i + 1; r.gd = r.gf - r.ga; });
  return { rows, byId: new Map(rows.map((r) => [r.id, r])) };
}

function venueProfile(games, home) {
  const own = games.filter((g) => g.home === home);
  const n = own.length;
  const avg = (f) => (n ? own.reduce((s, g) => s + f(g), 0) / n : null);
  return {
    n,
    ppg: avg((g) => (g.gf > g.ga ? 3 : g.gf === g.ga ? 1 : 0)),
    gf: avg((g) => g.gf),
    ga: avg((g) => g.ga),
  };
}

function teamProfile(games, tbl, id, kickoff, home) {
  const n = games.length;
  const rate = (f) => (n ? games.filter(f).length / n : null);
  // recency-weighted points per game over the last 6 matches (newest counts most)
  let w = 0, ws = 0;
  games.slice(0, 6).forEach((g, i) => {
    const wt = 6 - i;
    w += wt * (g.gf > g.ga ? 3 : g.gf === g.ga ? 1 : 0);
    ws += wt;
  });
  const row = tbl.byId.get(id);
  return {
    n,
    rank: row ? row.rank : null,
    pts: row ? row.pts : null,
    gd: row ? row.gd : null,
    teams: tbl.rows.length,
    venue: venueProfile(games, home),           // home record for the home side, away record for the away side
    cleanSheets: rate((g) => g.ga === 0),
    failedToScore: rate((g) => g.gf === 0),
    btts: rate((g) => g.gf > 0 && g.ga > 0),
    over25: rate((g) => g.gf + g.ga > 2),
    wForm: ws ? w / ws : null,
    restDays: games.length ? (kickoff - new Date(games[0].date).getTime()) / 864e5 : null, // league games only
  };
}

function profileLine(name, p, where) {
  const v = p.venue;
  return (
    `${name}: ${p.rank ? `${p.rank}/${p.teams} in table (${p.pts} pts, GD ${p.gd >= 0 ? '+' : ''}${p.gd})` : 'table n/a'}; ` +
    `${where} only: ${f1(v.ppg)} pts/game, ${f1(v.gf)} scored, ${f1(v.ga)} conceded (${v.n} games); ` +
    `recency-weighted form ${f1(p.wForm)} pts/game; clean sheets ${pctOrNA(p.cleanSheets)}, failed to score ${pctOrNA(p.failedToScore)}, ` +
    `both-teams-scored ${pctOrNA(p.btts)}, over 2.5 goals ${pctOrNA(p.over25)}; rest ${f1(p.restDays, 0)} days since last league game.`
  );
}

function poissonMarkets(lh, la) {
  const grid = scoreGrid(lh, la);
  let over25 = 0, btts = 0, best = { p: 0, i: 0, j: 0 };
  grid.forEach((row, i) => row.forEach((p, j) => {
    if (i + j > 2) over25 += p;
    if (i > 0 && j > 0) btts += p;
    if (p > best.p) best = { p, i, j };
  }));
  return { over25, btts, score: `${best.i}-${best.j}` };
}

// Everything the model knows about a match: base analysis + table position, venue splits, scoring habits, rest, goal markets
function deepAnalysis(m) {
  return cached('deep:' + m.id, CACHE_MS, async () => {
    const base = await statAnalysis(m);
    let season = [];
    try { season = await getSeason(m.comp); } catch (e) { console.log('Season fetch failed', m.comp, e.message); }
    const tbl = buildTable(season);
    const hp = teamProfile(teamGames(season, m.homeId), tbl, m.homeId, m.utc, true);
    const ap = teamProfile(teamGames(season, m.awayId), tbl, m.awayId, m.utc, false);
    const mk = base.form ? poissonMarkets(base.form.lh, base.form.la) : null;
    const extra = [
      profileLine(m.homeFull, hp, 'at home'),
      profileLine(m.awayFull, ap, 'away'),
      mk ? `Goal model: over 2.5 goals ${pct(mk.over25)}%, both teams score ${pct(mk.btts)}%, likeliest score ${mk.score}.` : '',
    ].filter(Boolean).join('\n');
    return { ...base, hp, ap, mk, extra, table: DOMESTIC.has(m.comp) && tbl.rows.length >= 8 };
  });
}

function whyPopular(short, full, prof, tableOk) {
  if (isBigClub(short, full)) return 'big club';
  if (tableOk && prof.rank && prof.rank <= TOP_RANK_EXCLUDE) return `top ${TOP_RANK_EXCLUDE} (#${prof.rank})`;
  return null;
}

// Confidence + agreement between odds / form model / AI + small "value" edge vs the market - fatigue
function pickScore(m, st, minConf = 0) {
  const pick = topPick(m, st.final);
  if (!pick || !m.odds || !pick.odds || pick.odds < MIN_PICK_ODDS || pick.p < minConf) return null;
  const srcs = [m.odds.probs, st.form, st.ai && st.ai.probs].filter(Boolean);
  const agreeN = srcs.filter((x) => argmax(x) === pick.k).length;
  const agree = agreeN / srcs.length;
  const edge = Math.max(-0.04, Math.min(0.04, pick.p - m.odds.probs[pick.k]));
  const side = pick.k === 'H' ? st.hp : pick.k === 'A' ? st.ap : null;
  const tired = side && side.restDays != null && side.restDays < 3 ? 0.02 : 0;
  // penalties: model leans a different way than the pick, or model and market are far apart (usually = missing info like injuries)
  const split = (st.form && argmax(st.form) !== pick.k ? 0.05 : 0) + (st.ai && argmax(st.ai.probs) !== pick.k ? 0.02 : 0);
  const dis = st.disagree != null ? DISAGREE_PENALTY * Math.max(0, st.disagree - 0.05) : 0;
  return { pick, agreeN, total: srcs.length, score: pick.p + 0.06 * agree + edge - tired - split - dis };
}
const best3Score = (m, st) => pickScore(m, st, BEST3_MIN_CONF);

async function computeBest3Raw() {
  const now = Date.now();
  const matches = (await getMatchesToday()).filter((m) => m.odds && m.utc > now);
  const pool0 = [];
  let skipped = 0;
  for (const m of matches) {
    const d = await deepAnalysis(m);
    const big = whyPopular(m.home, m.homeFull, d.hp, d.table) || whyPopular(m.away, m.awayFull, d.ap, d.table);
    if (big) { skipped++; continue; }                                       // top-table / popular club involved → never used
    pool0.push({ m, st: d, thin: Math.max(d.hp.n, d.hs.games || 0) < MIN_BEST3_GAMES || Math.max(d.ap.n, d.as.games || 0) < MIN_BEST3_GAMES });
  }
  // Tiers keep Best 3 filled every day: 0 = enough form data + confidence >= minimum, 1 = thin data (judged mostly on bookmaker odds),
  // 2 = confidence below the minimum. Better tiers always come first; the list only dips into a lower tier when it has to.
  const rank = (arr) =>
    arr.map((x) => {
      const sc = pickScore(x.m, x.st, 0);
      if (!sc) return null;
      return { ...x, sc, tier: sc.pick.p < BEST3_MIN_CONF ? 2 : x.thin ? 1 : 0 };
    }).filter(Boolean).sort((a, b) => a.tier - b.tier || b.sc.score - a.sc.score);

  const cands = rank(pool0).slice(0, MAX_AI_MATCHES);
  const refined = await pool(cands, AI_CONCURRENCY, async (x) => ({ m: x.m, thin: x.thin, st: await aiRefine(x.m, x.st).catch(() => x.st) }));
  pool0.forEach((x) => logPrediction(x.m, x.st));
  refined.forEach((x) => logPrediction(x.m, x.st));
  const list = rank(refined).slice(0, 3);
  list.forEach((x) => logPrediction(x.m, x.st, 'best3'));
  return { list, skipped, considered: matches.length };
}

async function computeBest3() {
  const key = 'best3:' + localDate(new Date());
  const c = cache.get(key);
  if (c && c.exp > Date.now()) return c.val;
  if (c && c.pending) return c.pending;
  const pending = computeBest3Raw();
  cache.set(key, { pending, exp: 0 });
  try {
    const val = await pending;
    const aiOk = !AI_ON || val.list.some((x) => x.st.ai);
    cache.set(key, { val, exp: Date.now() + (aiOk ? CACHE_MS : 10 * 60e3) });
    return val;
  } catch (e) {
    cache.delete(key);
    throw e;
  }
}

// ---------- Prediction log + self-evaluation (/stats) ----------
// Every analysed match is stored before kickoff; results are matched later to score each source (Brier / log-loss / accuracy).
let predLog = [];
const predById = new Map();
const predDirty = new Set();
let dbReady = false;
let saveTimer = null;
function savePredLog() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => flushPredLog().catch((e) => console.log('Prediction save failed:', e.message)), 1500);
}
async function flushPredLog() {
  if (!dbReady || !predDirty.size) return;
  const ids = [...predDirty];
  predDirty.clear();
  for (const id of ids) {
    const r = predById.get(id);
    if (!r) continue;
    try {
      await q('INSERT INTO predictions (id, data) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data', [id, JSON.stringify(r)]);
    } catch (e) {
      predDirty.add(id);
      console.log('Could not save prediction', id, e.message);
    }
  }
  if (predLog.length > 3000) {
    predLog = predLog.slice(-3000);
    predById.clear();
    predLog.forEach((r) => predById.set(r.id, r));
  }
}
async function loadPredLog() {
  const { rows } = await q('SELECT data FROM predictions ORDER BY id DESC LIMIT 3000');
  predLog = rows.map((r) => r.data).sort((a, b) => a.utc - b.utc);
  if (!predLog.length) {
    // one-time import of the old predictions.json (if you ran the file-based version before)
    try {
      const old = JSON.parse(fs.readFileSync(PRED_LOG, 'utf8'));
      if (Array.isArray(old)) {
        predLog = old.filter((r) => r && r.id);
        predLog.forEach((r) => predDirty.add(r.id));
      }
    } catch {}
  }
  predLog.forEach((r) => predById.set(r.id, r));
  if (predDirty.size) savePredLog();
}

const hda = (o) => (o ? { H: o.H, D: o.D, A: o.A } : null);
function logPrediction(m, st, tag) {
  if (!m.odds || !st || !st.final || m.utc <= Date.now()) return; // frozen once the match has started
  let r = predById.get(m.id);
  if (!r) {
    r = { id: m.id, comp: m.comp, home: m.home, away: m.away, utc: m.utc, tags: [] };
    predLog.push(r);
    predById.set(m.id, r);
  }
  if (r.res) return;
  r.mkt = hda(m.odds.probs); r.form = hda(st.form); r.ai = st.ai ? hda(st.ai.probs) : null; r.final = hda(st.final);
  r.odds = hda(m.odds.odds); r.best = hda(m.odds.best); // prices at the time of the pick (for ROI tracking)
  if (tag && !r.tags.includes(tag)) {
    r.tags.push(tag);
    if (!r.pubPick) { r.pubPick = argmax(st.final); r.pubOdds = hda(m.odds.odds); } // odds + side at the moment we first published it
  }
  predDirty.add(r.id);
  savePredLog();
}

// Settles finished matches. Re-downloads a competition's season at most every 25 min (results lag up to 6h otherwise).
const seasonFreshAt = new Map();
async function resolveLog() {
  const now = Date.now();
  const open = predLog.filter((r) => !r.res && r.utc < now - 3 * 3600e3 && r.utc > now - 5 * 864e5);
  const changed = [];
  for (const c of [...new Set(open.map((r) => r.comp))]) {
    if (now - (seasonFreshAt.get(c) || 0) > 25 * 60e3) { cache.delete('season:' + c); seasonFreshAt.set(c, now); }
    let season = [];
    try { season = await getSeason(c); } catch { continue; }
    const byId = new Map(season.map((x) => [x.id, x]));
    for (const r of open.filter((x) => x.comp === c)) {
      const x = byId.get(r.id);
      const s = x && x.score && (x.score.regularTime || x.score.fullTime); // 90-minute result for knockout games
      if (s && s.home != null) {
        r.res = s.home > s.away ? 'H' : s.home < s.away ? 'A' : 'D';
        r.score = `${s.home}-${s.away}`;
        predDirty.add(r.id);
        changed.push(r);
      }
    }
  }
  const still = open.filter((r) => !r.res);
  if (still.length) { // football-data lagging, rate-limited or not covering the league: settle from ESPN / TheSportsDB instead
    try { changed.push(...(await resolveFromFeed(still))); } catch (e) { console.log('Feed settle failed:', e.message); }
  }
  if (changed.length) savePredLog();
  return changed;
}

const brier = (p, o) => ['H', 'D', 'A'].reduce((s, k) => s + (p[k] - (k === o ? 1 : 0)) ** 2, 0);
const logloss = (p, o) => -Math.log(Math.max(p[o], 1e-6));
function pool2(a, b, wa, wb) {
  const o = {};
  let t = 0;
  for (const k of ['H', 'D', 'A']) { o[k] = Math.exp(wa * Math.log(Math.max(a[k], 1e-6)) + wb * Math.log(Math.max(b[k], 1e-6))); t += o[k]; }
  for (const k of ['H', 'D', 'A']) o[k] /= t;
  return o;
}
// Finds the market/form mix that would have scored best on your own logged results (needs 80+ matches)
function tuneWeights(rows) {
  const r = rows.filter((x) => x.form && x.mkt);
  if (r.length < 80) return null;
  const ll = (wf) => r.reduce((s, x) => s + logloss(pool2(x.mkt, x.form, 1 - wf, wf), x.res), 0) / r.length;
  let best = null;
  for (let wf = 0; wf <= 0.7001; wf += 0.05) { const v = ll(wf); if (!best || v < best.v) best = { wf, v }; }
  const cur = W_FORM / (W_MKT + W_FORM);
  return { ...best, cur, curV: ll(cur), n: r.length };
}

// Flat 1-unit stake on the model's pick at the average bookmaker price logged at pick time
function roiOf(rows) {
  let n = 0, hit = 0, profit = 0;
  for (const r of rows) {
    if (!r.odds || !r.final || !r.res) continue;
    const k = argmax(r.final);
    n++;
    if (k === r.res) { hit++; profit += r.odds[k] - 1; } else profit -= 1;
  }
  return { n, hit, profit, roi: n ? profit / n : null };
}
const signed = (x, d = 1) => (x >= 0 ? '+' : '') + x.toFixed(d);
const LEAGUE_NAMES = {
  PL: 'Premier League', ELC: 'Championship', BL1: 'Bundesliga', SA: 'Serie A', PD: 'La Liga', FL1: 'Ligue 1',
  DED: 'Eredivisie', PPL: 'Primeira Liga', CL: 'Champions League', BSA: 'Brasileirão', EC: 'Euros', WC: 'World Cup',
};

async function renderStats() {
  await resolveLog();
  const done = predLog.filter((r) => r.res && r.mkt && r.final);
  if (done.length < 10)
    return `📉 Only ${done.length} finished predictions logged so far (need 10+). The bot records every match it analyses and checks results itself — try again in a few days.\n\n` + DISCLAIMER;
  const both = done.filter((r) => r.form);
  const base = both.length >= 10 ? both : done;
  const row = (name, key) => {
    const r = base.filter((x) => x[key]);
    if (r.length < 5) return '';
    const acc = r.filter((x) => argmax(x[key]) === x.res).length / r.length;
    const br = r.reduce((s, x) => s + brier(x[key], x.res), 0) / r.length;
    const ll = r.reduce((s, x) => s + logloss(x[key], x.res), 0) / r.length;
    return `• ${name}: ${pct(acc)}% correct · Brier ${br.toFixed(3)} · log-loss ${ll.toFixed(3)} (${r.length})\n`;
  };
  let t = `📊 <b>Model scorecard</b> — ${done.length} finished matches\n<i>Lower Brier / log-loss = better calibrated.</i>\n\n`;
  t += row('Market odds', 'mkt') + row('Form model', 'form') + row('AI', 'ai') + row('Final blend', 'final');

  // published picks: accuracy + flat-stake ROI
  const picks = done.filter((x) => x.tags && x.tags.length);
  for (const [tag, label] of [['top5', 'Top 5'], ['best10', 'Best 10'], ['best3', 'Best 3']]) {
    const r = done.filter((x) => x.tags && x.tags.includes(tag));
    if (!r.length) continue;
    const hit = r.filter((x) => argmax(x.final) === x.res).length;
    const ro = roiOf(r);
    t += `\n🎯 ${label} picks: ${hit}/${r.length} correct (${pct(hit / r.length)}%)` + (ro.n >= 5 ? ` · ROI ${signed(ro.roi * 100)}% <i>(1u flat, ${ro.n} priced)</i>` : '');
  }
  const last30 = picks.filter((x) => x.utc > Date.now() - 30 * 864e5);
  if (last30.length >= 3) {
    const hit = last30.filter((x) => argmax(x.final) === x.res).length;
    const ro = roiOf(last30);
    t += `\n📅 Last 30 days: ${hit}/${last30.length} correct (${pct(hit / last30.length)}%)` + (ro.n >= 5 ? ` · ROI ${signed(ro.roi * 100)}%` : '');
  }
  const byComp = new Map();
  for (const r of picks) { if (!byComp.has(r.comp)) byComp.set(r.comp, []); byComp.get(r.comp).push(r); }
  const lg = [...byComp.entries()].filter(([, v]) => v.length >= 3).sort((a, b) => b[1].length - a[1].length).slice(0, 8);
  if (lg.length) {
    t += '\n\n🌍 <b>By league</b> (published picks)\n';
    for (const [c, v] of lg) {
      const hit = v.filter((x) => argmax(x.final) === x.res).length;
      t += `• ${esc(LEAGUE_NAMES[c] || c)}: ${hit}/${v.length} (${pct(hit / v.length)}%)\n`;
    }
  }
  const tw = tuneWeights(done);
  if (tw) {
    t += tw.v < tw.curV - 0.002
      ? `\n🔧 On ${tw.n} matches the best form share is ${pct(tw.wf)}% (now ${pct(tw.cur)}%) → try W_MARKET=${(1 - tw.wf).toFixed(2)} W_FORM=${tw.wf.toFixed(2)}`
      : `\n🔧 Current market/form weights are close to optimal on ${tw.n} matches.`;
  }
  return t + '\n\n' + DISCLAIMER;
}

function renderHistory() {
  const rows = predLog.filter((r) => r.tags && r.tags.length && r.final).sort((a, b) => b.utc - a.utc).slice(0, 15);
  if (!rows.length) return '📜 No published picks logged yet. Check back after the next match day.';
  let t = '📜 <b>Recent published picks</b>\n<i>✅ won · ❌ lost · ⏳ pending (1X2, 90 minutes)</i>\n\n';
  for (const r of rows) {
    const k = argmax(r.final);
    const label = k === 'H' ? `${r.home} win` : k === 'A' ? `${r.away} win` : 'Draw';
    const sign = { H: '1', D: 'X', A: '2' };
    const icon = !r.res ? '⏳' : r.res === k ? '✅' : '❌';
    const tags = (r.tags.includes('top5') ? '🏆' : '') + (r.tags.includes('best10') ? '🔟' : '') + (r.tags.includes('best3') ? '💎' : '');
    t += `${icon} ${fmtDate(r.utc)} ${tags} <b>${esc(r.home)} vs ${esc(r.away)}</b> — ${esc(label)} (${sign[k]})` +
      (r.odds ? ` @ ${r.odds[k].toFixed(2)}` : '') + (r.res ? ` · result ${sign[r.res]}${r.score ? ' (' + r.score + ')' : ''}` : '') + '\n';
  }
  return t + '\n' + DISCLAIMER;
}

async function renderCalibration() {
  await resolveLog();
  const done = predLog.filter((r) => r.res && r.final);
  if (done.length < 30) return `📐 Need 30+ finished predictions for a calibration table (have ${done.length}).`;
  const bins = [[0, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 1.01]];
  let t = '📐 <b>Calibration</b>\n<i>When the model says X%, how often does that outcome actually happen?</i>\n\n';
  for (const [lo, hi] of bins) {
    const r = done.filter((x) => { const p = Math.max(x.final.H, x.final.D, x.final.A); return p >= lo && p < hi; });
    if (r.length < 5) continue;
    const avg = r.reduce((s, x) => s + Math.max(x.final.H, x.final.D, x.final.A), 0) / r.length;
    const hit = r.filter((x) => argmax(x.final) === x.res).length / r.length;
    const gap = hit - avg;
    t += `• Said ~${pct(avg)}% → happened ${pct(hit)}% (${r.length} matches) ${Math.abs(gap) <= 0.05 ? '✅' : gap > 0 ? '⬆️ under-confident' : '⬇️ over-confident'}\n`;
  }
  return t + '\n' + DISCLAIMER;
}

// ---------- Message builders ----------
const DISCLAIMER =
  '⚠️ <i>Estimates only, not guaranteed. Odds shown are an average across European bookmakers and may differ from your bookie. 18+ · bet responsibly.</i>';

function oddsLine(m) {
  if (!m.odds) return '   <i>odds n/a</i>';
  const o = m.odds.odds;
  return `   1 <b>${o.H.toFixed(2)}</b> · X <b>${o.D.toFixed(2)}</b> · 2 <b>${o.A.toFixed(2)}</b>`;
}

// =====================================================================
//  PREMIUM LAYER — PostgreSQL · Paystack · Telegram Stars · extra features
// =====================================================================
const { Pool, types: pgTypes } = pg;
pgTypes.setTypeParser(20, (v) => Number(v));       // BIGINT  -> number
pgTypes.setTypeParser(1700, (v) => parseFloat(v)); // NUMERIC -> number
pgTypes.setTypeParser(1082, (v) => v);             // DATE    -> 'YYYY-MM-DD'

// ---------- Config ----------
const ADMIN_IDS = (process.env.ADMIN_IDS || '').split(',').map((x) => Number(x.trim())).filter((x) => Number.isFinite(x) && x > 0);
const isAdminId = (id) => ADMIN_IDS.includes(Number(id));
const PAYSTACK_KEY = process.env.PAYSTACK_SECRET_KEY || '';
const PUBLIC_URL = (process.env.PUBLIC_URL || process.env.RENDER_EXTERNAL_URL || '').replace(/\/+$/, '');
const PAYSTACK_EMAIL_DOMAIN = process.env.PAYSTACK_EMAIL_DOMAIN || 'footballprediction.com'; // Telegram gives no email; Paystack needs one
const STARS_ON = process.env.STARS !== 'off';
const PLANS = {
  day: { label: 'Day pass', days: 1, ngn: Number(process.env.PRICE_DAY_NGN || 500), stars: Number(process.env.STARS_DAY || 30) },
  week: { label: 'Weekly', days: 7, ngn: Number(process.env.PRICE_WEEK_NGN || 2000), stars: Number(process.env.STARS_WEEK || 100) },
  month: { label: 'Monthly', days: 30, ngn: Number(process.env.PRICE_MONTH_NGN || 6000), stars: Number(process.env.STARS_MONTH || 350) },
};
const FREE_DAILY_LIMIT = Number(process.env.FREE_DAILY_LIMIT || 3);   // free users: match analyses / predictions per day
const FREE_PICKS = Number(process.env.FREE_PICKS || 1);               // (legacy) no longer used: Top 5 / Best 3 / Best 1 are Premium-only
const BEST10_INCLUDES_TOP5 = process.env.BEST10_INCLUDES_TOP5 === 'on'; // default off: free Best 10 = next 10 picks AFTER the premium Top 5, so Premium picks are not given away
const REF_BONUS_DAYS = Number(process.env.REF_BONUS_DAYS || 2);       // premium days the referrer earns per new user
const REF_NEW_DAYS = Number(process.env.REF_NEW_DAYS || 1);           // premium days a referred new user gets
const REF_MAX = Number(process.env.REF_MAX || 10);                    // max rewarded referrals per user
const REF_PAID_DAYS = Number(process.env.REF_PAID_DAYS || 3);         // extra days when a friend you invited PAYS for the first time
const REF_PAID_MAX = Number(process.env.REF_PAID_MAX || 20);          // max rewarded paying referrals per user
const REF_MILESTONE_N = Number(process.env.REF_MILESTONE_N || 3);     // every N paying friends = milestone bonus
const REF_MILESTONE_DAYS = Number(process.env.REF_MILESTONE_DAYS || 7);
const TEASER_HOUR = Number(process.env.TEASER_HOUR || 9);             // free daily tips are sent from this local hour (user's timezone)
const STALE_DRIFT = Number(process.env.STALE_DRIFT || 0.1);           // price-move alert when a published pick's odds move this much (10%)
const SAFE_TARGET_ODDS = Number(process.env.SAFE_TARGET_ODDS || 2.0); // /safe builds the safest slip reaching these combined odds
const MAX_FOLLOW_DAY = Number(process.env.MAX_FOLLOW_DAY || 30);      // picks a user can follow per day
const LB_MIN_PICKS = Number(process.env.LB_MIN_PICKS || 5);           // settled followed picks needed to enter the leaderboard
const VALUE_MIN_EDGE = Number(process.env.VALUE_MIN_EDGE || 0.03);    // model prob must beat the market by this much
const VALUE_MIN_EV = Number(process.env.VALUE_MIN_EV || 0.05);        // and expected value at the best price must be at least +5%
const VALUE_MAX_ODDS = Number(process.env.VALUE_MAX_ODDS || 6);       // ignore long shots
const KELLY_CAP = Number(process.env.KELLY_CAP || 0.05);              // never suggest more than 5% of bankroll
const MAX_WATCH = Number(process.env.MAX_WATCH || 20);
const ODDS_LOW_CREDITS = Number(process.env.ODDS_LOW_CREDITS || 40);
const LEAGUES = Object.keys(COMP_TO_ODDS);

// ---------- PostgreSQL ----------
const dbUrl = process.env.DATABASE_URL;
const dbSsl = !/localhost|127\.0\.0\.1/.test(dbUrl) && process.env.PGSSL !== 'off';
const db = new Pool({ connectionString: dbUrl, ssl: dbSsl ? { rejectUnauthorized: false } : false, max: 8 });
db.on('error', (e) => console.log('Postgres pool error:', e.message));
const q = (text, params) => db.query(text, params);

async function initDb() {
  await q(`CREATE TABLE IF NOT EXISTS users (
    id BIGINT PRIMARY KEY, username TEXT, first_name TEXT, lang TEXT DEFAULT 'en', tz TEXT,
    joined_at TIMESTAMPTZ DEFAULT now(), last_seen TIMESTAMPTZ DEFAULT now(), referred_by BIGINT,
    plan_until TIMESTAMPTZ, warned_exp BOOLEAN DEFAULT false,
    sub_daily BOOLEAN DEFAULT false, push_hour INT DEFAULT 9, last_push DATE,
    leagues TEXT[] DEFAULT '{}', min_odds NUMERIC DEFAULT 0, bankroll NUMERIC, kelly NUMERIC DEFAULT 0.25,
    banned BOOLEAN DEFAULT false, blocked BOOLEAN DEFAULT false)`);
  await q(`CREATE TABLE IF NOT EXISTS payments (
    id SERIAL PRIMARY KEY, user_id BIGINT, provider TEXT, ref TEXT UNIQUE, plan TEXT,
    amount NUMERIC, currency TEXT, status TEXT, created_at TIMESTAMPTZ DEFAULT now())`);
  await q(`CREATE TABLE IF NOT EXISTS usage (user_id BIGINT, day DATE, n INT DEFAULT 0, PRIMARY KEY (user_id, day))`);
  await q(`CREATE TABLE IF NOT EXISTS watch (
    user_id BIGINT, match_id BIGINT, comp TEXT, home TEXT, away TEXT, utc BIGINT, lean TEXT,
    ko_sent BOOLEAN DEFAULT false, res_sent BOOLEAN DEFAULT false, PRIMARY KEY (user_id, match_id))`);
  await q(`CREATE TABLE IF NOT EXISTS predictions (id BIGINT PRIMARY KEY, data JSONB NOT NULL)`);
  await q(`CREATE INDEX IF NOT EXISTS users_sub_idx ON users (sub_daily) WHERE sub_daily`);
  await q(`CREATE INDEX IF NOT EXISTS watch_open_idx ON watch (utc) WHERE NOT ko_sent OR NOT res_sent`);
  // --- feature pack: self-exclusion, free tips, follows, promo codes ---
  await q(`ALTER TABLE users ADD COLUMN IF NOT EXISTS excluded_until TIMESTAMPTZ`);
  await q(`ALTER TABLE users ADD COLUMN IF NOT EXISTS teaser_daily BOOLEAN DEFAULT true`);
  await q(`ALTER TABLE users ADD COLUMN IF NOT EXISTS last_teaser DATE`);
  await q(`ALTER TABLE users ADD COLUMN IF NOT EXISTS expired_notified BOOLEAN DEFAULT false`);
  await q(`CREATE TABLE IF NOT EXISTS followed (
    user_id BIGINT, match_id BIGINT, home TEXT, away TEXT, utc BIGINT, pick TEXT, label TEXT, odds NUMERIC,
    res TEXT, won BOOLEAN, profit NUMERIC, created_at TIMESTAMPTZ DEFAULT now(), PRIMARY KEY (user_id, match_id))`);
  await q(`CREATE INDEX IF NOT EXISTS followed_open_idx ON followed (utc) WHERE won IS NULL`);
  await q(`CREATE TABLE IF NOT EXISTS promo_codes (
    code TEXT PRIMARY KEY, days INT NOT NULL, max_uses INT NOT NULL DEFAULT 1, uses INT NOT NULL DEFAULT 0,
    expires_at TIMESTAMPTZ, created_at TIMESTAMPTZ DEFAULT now())`);
  await q(`CREATE TABLE IF NOT EXISTS promo_redemptions (
    code TEXT, user_id BIGINT, redeemed_at TIMESTAMPTZ DEFAULT now(), PRIMARY KEY (code, user_id))`);
  dbReady = true;
}

// ---------- Small utils ----------
const money = (n) => Number(n).toLocaleString('en-NG', { maximumFractionDigits: 0 });
const peek = (key) => { const c = cache.get(key); return c && c.exp > Date.now() ? c.val : null; };
function refineAll() { // AI opinion for EVERY upcoming match of the day; cached and shared by Top 5 / Best 3 / Value / Acca / Goals / DC
  return cached('refineall:' + localDate(new Date()), CACHE_MS, async () => {
    if (!AI_ON) return true;
    const ms = (await getMatchesToday()).filter((m) => m.utc > Date.now());
    let ok = 0;
    await pool(ms, AI_CONCURRENCY, async (m) => { try { const r = await aiRefine(m, await deepAnalysis(m)); if (r && r.ai) ok++; } catch (e) {} });
    if (ms.length && !ok) throw new Error('AI unavailable'); // not cached -> retried on the next request
    return true;
  });
}
const bestKnown = (m, d) => peek('ai:' + m.id) || d; // use the AI-refined analysis when it already exists
const argText = (ctx) => ((ctx.message && ctx.message.text) || '').replace(/^\/\S+\s*/, '').trim();
function splitMsg(text, max = 3900) {
  if (text.length <= max) return [text];
  const out = [];
  let cur = '';
  for (const line of text.split('\n')) {
    if ((cur + '\n' + line).length > max) { out.push(cur); cur = line; } else cur = cur ? cur + '\n' + line : line;
  }
  if (cur) out.push(cur);
  return out;
}
function localDateTz(d, tz) {
  try { return new Intl.DateTimeFormat('en-CA', { timeZone: tz || TZ, year: 'numeric', month: '2-digit', day: '2-digit' }).format(d); }
  catch { return localDate(d); }
}
function hourTz(d, tz) {
  try { return Number(new Intl.DateTimeFormat('en-GB', { timeZone: tz || TZ, hour: '2-digit', hour12: false }).format(d)) % 24; }
  catch { return Number(fmtTime(d.getTime()).slice(0, 2)) % 24; }
}
const validTz = (tz) => { try { new Intl.DateTimeFormat('en', { timeZone: tz }); return true; } catch { return false; } };

// ---------- Languages: English · Pidgin · Français ----------
const CMD_LIST =
  "/menu – main menu\n/best10 – 10 free picks · /picks – all pick lists\n/top5 · /best3 · /best1 – top picks (Premium)\n/best10pro · /potw – Pro list &amp; pick of the week (Premium)\n" +
  '/today · /live · /table · /predict\n/value · /goals · /dc · /scores – markets (Premium)\n/result · /htft · /combos · /handicaps · /market – full market boards (Premium)\n/acca · /safe · /weekend – accumulators (Premium)\n' +
  '/myrecord · /leaders – track your picks\n/watchlist · /subscribe · /stake\n/premium · /redeem · /refer · /account · /settings · /lang\n' +
  '/stats · /history · /calibration – model track record\n/responsible – stay in control · /help';
const TX = {
  en: {
    start: (n) => `👋 <b>Welcome${n ? ', ' + esc(n) : ''}!</b>\n\n⚽ <b>Football Predictions Bot</b>\nBookmaker odds + a goal model + AI, blended into daily picks.\n\n🆓 <b>Free:</b> 🔟 Best 10 every day.\n💎 <b>Premium:</b> Top 5, Best 3, Best 1, value bets, accas and more.\n\nTap a section below to begin 👇`,
    wait_top5: '🔎 Analysing today\'s matches… the first run can take a minute or two.',
    wait_best3: '💎 Hunting for under-the-radar picks… the first run can take a minute or two.',
    wait_generic: '⏳ Crunching the numbers… this can take a minute on the first run.',
    fail: '⚠️ That failed. Please try again shortly.',
    top5_head: 'Top 5 picks today',
    top5_sub: 'Ranked by confidence, source agreement and value vs the market',
    best3_head: 'Best 3 — under-the-radar picks',
    today_head: (n, p, pages) => `⚽ <b>Today's matches</b> (${n}) — page ${p}/${pages}`,
    none_qual: '😴 No qualifying matches with odds today. Try again later.',
    none_prefs: '😴 Nothing qualifies inside your league / min-odds filters today. Adjust them in /settings.',
    locked_more: (n) => `${n} more pick${n > 1 ? 's' : ''} unlocked with Premium`,
    premium_only: '🔒 <b>Premium feature.</b> Upgrade to unlock it — pay with Telegram Stars or Paystack.',
    limit_reached: (n) => `🔒 You've used your ${n} free analyses for today. Premium gives unlimited analyses, Top 5 / Best 3 / Best 1, value bets, accas, daily push and kickoff alerts.`,
    upgrade: '⭐ Upgrade to Premium',
    plan_free: 'Free',
    plan_active: (d) => `Premium until ${d}`,
    sub_on: (h, tz) => `🔔 Daily picks ON — you'll get them around ${String(h).padStart(2, '0')}:00 (${tz}). Change with /pushtime and /timezone.`,
    sub_off: '🔕 Daily picks turned off.',
    paid_ok: (plan, d) => `🎉 <b>Payment received — ${plan} Premium active until ${d}.</b>\nEnjoy! Try /value, /acca or /subscribe.`,
    lang_set: '✅ Language set to English.',
    daily_head: '📬 Your daily picks',
    kickoff: (home, away, mins, lean) => `⏰ <b>${esc(home)} vs ${esc(away)}</b> kicks off in ~${mins} min.${lean ? `\nOur lean: <b>${esc(lean)}</b>` : ''}`,
    exp_warn: (d) => `⏳ Your Premium ends ${d}. Renew with /premium to keep daily picks and alerts.`,
    disc: DISCLAIMER_EN(),
  },
  pcm: {
    start: (n) => `👋 <b>Welcome${n ? ', ' + esc(n) : ''}!</b>\n\n⚽ <b>Football Predictions Bot</b>\nWe mix bookmaker odds, goal model and AI to give you correct picks every day.\n\n🆓 <b>Free:</b> Best 10 every day.\n💎 <b>Premium:</b> Top 5, Best 3, Best 1, value bets, accas and more.\n\nTap one section below to start 👇`,
    wait_top5: '🔎 I dey check today matches… first time fit take small minute.',
    wait_best3: '💎 I dey find the hidden gems… first time fit take small minute.',
    wait_generic: '⏳ I dey do the maths… e fit take small minute.',
    fail: '⚠️ E no work. Try again small time.',
    top5_head: 'Top 5 picks for today',
    top5_sub: 'We rank am by confidence, how sources agree and value against market',
    best3_head: 'Best 3 — hidden gems',
    today_head: (n, p, pages) => `⚽ <b>Today matches</b> (${n}) — page ${p}/${pages}`,
    none_qual: '😴 No match qualify with odds today. Check back later.',
    none_prefs: '😴 Nothing fit your league / min-odds filter today. Change am for /settings.',
    locked_more: (n) => `${n} more pick${n > 1 ? 's' : ''} dey wait for Premium`,
    premium_only: '🔒 <b>Na Premium feature.</b> Upgrade to open am — pay with Telegram Stars or Paystack.',
    limit_reached: (n) => `🔒 You don use your ${n} free analyses for today. Premium go give you unlimited analyses, full Top 5 / Best 3, value bets, accas, daily push and kickoff alerts.`,
    upgrade: '⭐ Go Premium',
    plan_free: 'Free',
    plan_active: (d) => `Premium till ${d}`,
    sub_on: (h, tz) => `🔔 Daily picks don ON — you go get am around ${String(h).padStart(2, '0')}:00 (${tz}). Use /pushtime and /timezone to change.`,
    sub_off: '🔕 Daily picks don off.',
    paid_ok: (plan, d) => `🎉 <b>Payment don land — ${plan} Premium dey active till ${d}.</b>\nEnjoy! Try /value, /acca or /subscribe.`,
    lang_set: '✅ Language don change to Pidgin.',
    daily_head: '📬 Your daily picks',
    kickoff: (home, away, mins, lean) => `⏰ <b>${esc(home)} vs ${esc(away)}</b> go start in ~${mins} min.${lean ? `\nWetin we dey lean: <b>${esc(lean)}</b>` : ''}`,
    exp_warn: (d) => `⏳ Your Premium go end ${d}. Renew with /premium make you no lose daily picks.`,
    disc: '⚠️ <i>Na estimate be this, no be guarantee. Odds fit differ for your bookie. 18+ · bet wisely.</i>',
  },
  fr: {
    start: (n) => `👋 <b>Bienvenue${n ? ', ' + esc(n) : ''} !</b>\n\n⚽ <b>Bot de pronostics football</b>\nCotes des bookmakers + modèle de buts + IA, combinés en pronostics quotidiens.\n\n🆓 <b>Gratuit :</b> Best 10 chaque jour.\n💎 <b>Premium :</b> Top 5, Best 3, Best 1, value bets, combinés et plus.\n\nChoisissez une section ci-dessous 👇`,
    wait_top5: '🔎 Analyse des matchs du jour… la première fois peut prendre une ou deux minutes.',
    wait_best3: '💎 Recherche de pronostics sous le radar… la première fois peut prendre une ou deux minutes.',
    wait_generic: '⏳ Calculs en cours… cela peut prendre une minute la première fois.',
    fail: '⚠️ Échec. Réessayez dans un instant.',
    top5_head: 'Top 5 des pronostics du jour',
    top5_sub: 'Classés par confiance, accord des sources et valeur face au marché',
    best3_head: 'Best 3 — pronostics sous le radar',
    today_head: (n, p, pages) => `⚽ <b>Matchs du jour</b> (${n}) — page ${p}/${pages}`,
    none_qual: '😴 Aucun match éligible avec des cotes aujourd\'hui. Réessayez plus tard.',
    none_prefs: '😴 Rien ne correspond à vos filtres (ligues / cote min.) aujourd\'hui. Modifiez-les dans /settings.',
    locked_more: (n) => `${n} pronostic${n > 1 ? 's' : ''} de plus avec Premium`,
    premium_only: '🔒 <b>Fonction Premium.</b> Passez à Premium — paiement par Telegram Stars ou Paystack.',
    limit_reached: (n) => `🔒 Vous avez utilisé vos ${n} analyses gratuites du jour. Premium : analyses illimitées, Top 5 / Best 3 complets, value bets, combinés, envoi quotidien et alertes de coup d'envoi.`,
    upgrade: '⭐ Passer à Premium',
    plan_free: 'Gratuit',
    plan_active: (d) => `Premium jusqu'au ${d}`,
    sub_on: (h, tz) => `🔔 Pronostics quotidiens activés — vers ${String(h).padStart(2, '0')}:00 (${tz}). Modifiez avec /pushtime et /timezone.`,
    sub_off: '🔕 Pronostics quotidiens désactivés.',
    paid_ok: (plan, d) => `🎉 <b>Paiement reçu — Premium ${plan} actif jusqu'au ${d}.</b>\nEssayez /value, /acca ou /subscribe.`,
    lang_set: '✅ Langue réglée sur Français.',
    daily_head: '📬 Vos pronostics du jour',
    kickoff: (home, away, mins, lean) => `⏰ <b>${esc(home)} vs ${esc(away)}</b> commence dans ~${mins} min.${lean ? `\nNotre tendance : <b>${esc(lean)}</b>` : ''}`,
    exp_warn: (d) => `⏳ Votre Premium se termine le ${d}. Renouvelez avec /premium.`,
    disc: '⚠️ <i>Estimations uniquement, sans garantie. Les cotes peuvent différer chez votre bookmaker. 18+ · jouez responsable.</i>',
  },
};
function DISCLAIMER_EN() {
  return '⚠️ <i>Estimates only, not guaranteed. Odds shown are an average across European bookmakers and may differ from your bookie. 18+ · bet responsibly.</i>';
}
function t(u, key, ...args) {
  const lang = (u && TX[u.lang] ? u.lang : 'en');
  const v = TX[lang][key] !== undefined ? TX[lang][key] : TX.en[key];
  return typeof v === 'function' ? v(...args) : v;
}

// ---------- Users, plans, quotas ----------
const hasPremium = (u) => isAdminId(u.id) || (u.plan_until && new Date(u.plan_until).getTime() > Date.now());
const fmtUntil = (d) => new Intl.DateTimeFormat('en-GB', { timeZone: TZ, day: '2-digit', month: 'short', year: 'numeric' }).format(new Date(d));

async function updUser(id, fields) {
  const keys = Object.keys(fields); // keys are internal constants (never user input)
  if (!keys.length) return null;
  const sets = keys.map((k, i) => `${k} = $${i + 2}`).join(', ');
  const { rows } = await q(`UPDATE users SET ${sets} WHERE id = $1 RETURNING *`, [id, ...keys.map((k) => fields[k])]);
  return rows[0] || null;
}
async function grantDays(userId, days) {
  const { rows } = await q(
    `UPDATE users SET plan_until = GREATEST(COALESCE(plan_until, now()), now()) + make_interval(days => $2::int), warned_exp = false, expired_notified = false
     WHERE id = $1 RETURNING *`, [userId, Math.round(days)]);
  return rows[0] || null;
}
async function useQuota(ctx) { // free users get FREE_DAILY_LIMIT match analyses per day; premium + admin are unlimited
  if (ctx.premium || ctx.isAdmin) return true;
  const { rows } = await q(
    `INSERT INTO usage (user_id, day, n) VALUES ($1, $2, 1) ON CONFLICT (user_id, day) DO UPDATE SET n = usage.n + 1 RETURNING n`,
    [ctx.u.id, localDate(new Date())]);
  return rows[0].n <= FREE_DAILY_LIMIT;
}
async function quotaUsed(userId) {
  const { rows } = await q('SELECT n FROM usage WHERE user_id = $1 AND day = $2', [userId, localDate(new Date())]);
  return rows.length ? rows[0].n : 0;
}

async function sendTo(id, text, extra = {}) {
  try {
    await bot.telegram.sendMessage(id, text, { parse_mode: 'HTML', link_preview_options: { is_disabled: true }, ...extra });
    return true;
  } catch (e) {
    const code = e && e.response && e.response.error_code;
    if (code === 403) q('UPDATE users SET blocked = true WHERE id = $1', [id]).catch(() => {});
    else if (code === 429) await sleep((((e.response || {}).parameters || {}).retry_after || 3) * 1000);
    else console.log('sendTo failed for', id, '-', e.message);
    return false;
  }
}
const notifyAdmins = (text) => Promise.all(ADMIN_IDS.map((id) => sendTo(id, text)));

// ---------- Payments: activation (shared by Stars + Paystack) ----------
async function activatePlan(userId, planKey, provider, amount, currency) {
  const plan = PLANS[planKey];
  if (!plan) return null;
  const u = await grantDays(userId, plan.days);
  if (!u) return null;
  await sendTo(userId, t(u, 'paid_ok', plan.label, fmtUntil(u.plan_until)), { reply_markup: undefined });
  creditReferrer(u).catch((e) => console.log('Referral credit failed:', e.message));
  notifyAdmins(`💰 <b>New payment</b>: ${esc(u.first_name || '')} (<code>${userId}</code>) · ${plan.label} · ${provider} ${currency === 'XTR' ? amount + ' ⭐' : '₦' + money(amount)}`).catch(() => {});
  return u;
}

// Referrer earns bonus days when an invited friend pays for the first time (+ a milestone bonus every N paying friends)
async function creditReferrer(u) {
  if (!u.referred_by) return;
  const mine = (await q(`SELECT count(*)::int AS n FROM payments WHERE user_id = $1 AND status = 'success'`, [u.id])).rows[0].n;
  if (mine !== 1) return; // only the friend's FIRST payment counts
  const paying = (await q(`SELECT count(DISTINCT p.user_id)::int AS n FROM payments p JOIN users x ON x.id = p.user_id WHERE x.referred_by = $1 AND p.status = 'success'`, [u.referred_by])).rows[0].n;
  if (paying > REF_PAID_MAX) return;
  let days = REF_PAID_DAYS;
  let msg = `💎 A friend you invited just went Premium — <b>+${REF_PAID_DAYS} days</b> for you!`;
  if (REF_MILESTONE_N > 0 && paying % REF_MILESTONE_N === 0) {
    days += REF_MILESTONE_DAYS;
    msg += `\n🏅 Milestone: <b>${paying} paying friends</b> — bonus <b>+${REF_MILESTONE_DAYS} days</b>!`;
  }
  if (days <= 0) return;
  const ru = await grantDays(u.referred_by, days);
  if (ru) await sendTo(u.referred_by, msg + `\nYour Premium now runs until ${fmtUntil(ru.plan_until)}.`);
}

// --- Telegram Stars (currency XTR, no provider token) ---
async function sendStarsInvoice(ctx, planKey) {
  const plan = PLANS[planKey];
  await ctx.telegram.sendInvoice(ctx.chat.id, {
    title: `Premium ${plan.label}`,
    description: `${plan.days} days of Premium: full picks, value bets, accas, daily push, kickoff alerts.`,
    payload: `plan:${planKey}`,
    provider_token: '',
    currency: 'XTR',
    prices: [{ label: `Premium ${plan.label}`, amount: plan.stars }],
  });
}
async function onStarsPaid(ctx) {
  const sp = ctx.message.successful_payment;
  const planKey = String(sp.invoice_payload || '').replace(/^plan:/, '');
  if (!PLANS[planKey]) return notifyAdmins(`⚠️ Stars payment with unknown payload "${esc(sp.invoice_payload)}" from <code>${ctx.from.id}</code> (charge ${esc(sp.telegram_payment_charge_id)})`);
  const ref = 'xtr_' + sp.telegram_payment_charge_id;
  const ins = await q(
    `INSERT INTO payments (user_id, provider, ref, plan, amount, currency, status) VALUES ($1, 'stars', $2, $3, $4, 'XTR', 'success') ON CONFLICT (ref) DO NOTHING RETURNING id`,
    [ctx.from.id, ref, planKey, sp.total_amount]);
  if (!ins.rows.length) return; // already processed
  await activatePlan(ctx.from.id, planKey, 'Stars', sp.total_amount, 'XTR');
}

// --- Paystack (NGN) ---
async function paystackInit(u, planKey) {
  const plan = PLANS[planKey];
  const ref = `fb_${u.id}_${Date.now().toString(36)}${crypto.randomBytes(3).toString('hex')}`;
  const res = await fetchRetry('https://api.paystack.co/transaction/initialize', {
    method: 'POST',
    headers: { Authorization: `Bearer ${PAYSTACK_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      email: `tg${u.id}@${PAYSTACK_EMAIL_DOMAIN}`,
      amount: Math.round(plan.ngn * 100),
      currency: 'NGN',
      reference: ref,
      callback_url: PUBLIC_URL ? `${PUBLIC_URL}/paystack/callback` : undefined,
      metadata: { user_id: u.id, plan: planKey },
    }),
  }, 30000, 2);
  const j = await res.json().catch(() => ({}));
  if (!j.status || !j.data) throw new Error('Paystack init failed: ' + (j.message || res.status));
  await q(`INSERT INTO payments (user_id, provider, ref, plan, amount, currency, status) VALUES ($1, 'paystack', $2, $3, $4, 'NGN', 'pending')`, [u.id, ref, planKey, plan.ngn]);
  return { url: j.data.authorization_url, ref };
}
async function paystackVerify(ref) {
  const res = await fetchRetry(`https://api.paystack.co/transaction/verify/${encodeURIComponent(ref)}`, { headers: { Authorization: `Bearer ${PAYSTACK_KEY}` } }, 25000, 2);
  const j = await res.json().catch(() => null);
  return j && j.status && j.data && j.data.status === 'success' ? j.data : null;
}
async function settlePaystack(ref) {
  const { rows } = await q(`SELECT * FROM payments WHERE ref = $1 AND provider = 'paystack'`, [ref]);
  if (!rows.length) return { ok: false, reason: 'unknown' };
  const p = rows[0];
  if (p.status === 'success') return { ok: true, already: true };
  const d = await paystackVerify(ref);
  if (!d) return { ok: false, reason: 'pending' };
  if (d.amount < Math.round(p.amount * 100)) return { ok: false, reason: 'amount' };
  const upd = await q(`UPDATE payments SET status = 'success' WHERE ref = $1 AND status <> 'success' RETURNING id`, [ref]);
  if (!upd.rows.length) return { ok: true, already: true }; // webhook + button raced; only one activates
  await activatePlan(p.user_id, p.plan, 'Paystack', p.amount, 'NGN');
  return { ok: true };
}
if (PAYSTACK_KEY) {
  paystackWebhook = (req, res) => {
    const chunks = [];
    let size = 0;
    req.on('data', (c) => { size += c.length; if (size > 1e6) req.destroy(); else chunks.push(c); });
    req.on('end', async () => {
      const raw = Buffer.concat(chunks);
      const sig = String(req.headers['x-paystack-signature'] || '');
      const mac = crypto.createHmac('sha512', PAYSTACK_KEY).update(raw).digest('hex');
      let valid = false;
      try { valid = sig.length === mac.length && crypto.timingSafeEqual(Buffer.from(sig), Buffer.from(mac)); } catch {}
      if (!valid) { res.writeHead(401); return res.end(); }
      res.writeHead(200); res.end('ok');
      try {
        const ev = JSON.parse(raw.toString('utf8'));
        if (ev.event === 'charge.success' && ev.data && ev.data.reference) await settlePaystack(ev.data.reference);
      } catch (e) { console.log('Paystack webhook error:', e.message); }
    });
  };
}
// safety net for missed webhooks: re-check recent pending Paystack payments
async function recheckPendingPaystack() {
  if (!PAYSTACK_KEY) return;
  const { rows } = await q(`SELECT ref FROM payments WHERE provider = 'paystack' AND status = 'pending' AND created_at > now() - interval '3 hours' LIMIT 20`);
  for (const r of rows) { try { await settlePaystack(r.ref); } catch (e) { console.log('Paystack recheck failed:', e.message); } }
}

// ---------- Odds API credit tracking (admin /credits) ----------
const oddsCredits = { remaining: null, used: null, at: 0, alertedDay: '' };
function noteCredits(res) {
  const rem = Number(res.headers.get('x-requests-remaining'));
  const used = Number(res.headers.get('x-requests-used'));
  if (Number.isFinite(rem)) {
    oddsCredits.remaining = rem; oddsCredits.used = Number.isFinite(used) ? used : null; oddsCredits.at = Date.now();
    const day = localDate(new Date());
    if (rem <= ODDS_LOW_CREDITS && oddsCredits.alertedDay !== day) {
      oddsCredits.alertedDay = day;
      notifyAdmins(`⚠️ The Odds API credits are low: <b>${rem}</b> left this month.`).catch(() => {});
    }
  }
}

// ---------- Rendering (per-user prefs + paywall) ----------
const FORM_EMOJI = { W: '✅', D: '🟨', L: '❌' };
const formStrip = (s) => (s && s.form && s.form !== 'n/a' ? s.form.split(' ').map((x) => FORM_EMOJI[x] || x).join('') : 'n/a');
const userLeagues = (u) => (u && Array.isArray(u.leagues) ? u.leagues : []);
const byLeague = (arr, u) => { const lg = userLeagues(u); return lg.length ? arr.filter((x) => lg.includes(x.m.comp) || x.m.extra) : arr; };
function applyPrefs(arr, u) { // league filter + minimum pick odds for Top 5 / Best 3
  let out = byLeague(arr, u);
  const mo = u && Number(u.min_odds) > 0 ? Number(u.min_odds) : 0;
  if (mo) out = out.filter((x) => { const p = x.sc ? x.sc.pick : topPick(x.m, x.st.final); return p && p.odds >= mo; });
  return out;
}
const upgradeKb = (u) => Markup.inlineKeyboard([[Markup.button.callback(t(u, 'upgrade'), 'up')]]);
const bestPriceTxt = (m, k) => (m.odds && m.odds.best && m.odds.best[k] ? ` · best ${m.odds.best[k].toFixed(2)}${m.odds.bestBk && m.odds.bestBk[k] ? ' @ ' + esc(m.odds.bestBk[k]) : ''}` : '');

// ----- Stake maths -----
function kellyStake(p, o, bankroll, frac = 0.25) {
  const b = o - 1;
  const f = b > 0 ? (p * o - 1) / b : 0;
  if (!(f > 0)) return { f: 0, adj: 0, stake: 0 };
  const adj = Math.min(f * frac, KELLY_CAP);
  return { f, adj, stake: bankroll ? bankroll * adj : null };
}
function stakeTxt(u, p, o) {
  if (!u || !u.bankroll) return '';
  const k = kellyStake(p, o, Number(u.bankroll), Number(u.kelly) || 0.25);
  return k.adj > 0 ? `\n   💵 Stake idea: <b>${money(k.stake)}</b> (${(k.adj * 100).toFixed(1)}% of bankroll, ${Math.round((Number(u.kelly) || 0.25) * 100)}% Kelly)` : '';
}

// ----- /today -----
const dayName = (off) => (off === 0 ? 'Today' : off === 1 ? 'Tomorrow' : new Intl.DateTimeFormat('en-GB', { timeZone: TZ, weekday: 'short', day: '2-digit', month: 'short' }).format(new Date(Date.now() + off * 864e5)));
const dayTabs = (day) => [0, 1, 2].map((d) => Markup.button.callback((d === day ? '• ' : '') + dayName(d), `pd:${d}:0`));
async function listForDay(day) {
  if (day === 0) return getMatchesToday();
  const target = localDate(new Date(Date.now() + day * 864e5));
  return (await getUpcoming()).filter((m) => localDate(new Date(m.utc)) === target).sort((a, b) => a.compName.localeCompare(b.compName) || a.utc - b.utc);
}
async function renderPage(page, u, day = 0) {
  const all = await listForDay(day);
  const lg = userLeagues(u);
  const list = lg.length ? all.filter((m) => lg.includes(m.comp) || m.extra) : all;
  const tabs = dayTabs(day);
  if (!list.length) {
    const when = day === 0 ? 'today' : day === 1 ? 'tomorrow' : 'on ' + dayName(day);
    let text = all.length ? `😴 No matches ${when} in your selected leagues. Change them in /settings.` : `😴 No matches ${when} in the covered leagues.`;
    if (day === 0 && !all.length) text += nextInfo
      ? `\n📅 Next up: ${nextInfo.n} match${nextInfo.n > 1 ? 'es' : ''} on ${nextInfo.date}.`
      : '\n📅 Nothing in the next 2 days either (likely an international break).';
    return { text, kb: Markup.inlineKeyboard([tabs, [Markup.button.callback('🔄 Refresh', `rf:${day}:0`), Markup.button.callback('⚙️ Leagues', 'set:lg'), MENU_BTN]]) };
  }
  const pages = Math.ceil(list.length / PAGE_SIZE);
  page = Math.max(0, Math.min(page, pages - 1));
  const slice = list.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  let text = day === 0 ? t(u, 'today_head', list.length, page + 1, pages) + '\n' : `📅 <b>${dayName(day)}'s matches</b> (${list.length}) — page ${page + 1}/${pages}\n`;
  let lastComp = '';
  slice.forEach((m, i) => {
    if (m.compName !== lastComp) { text += `\n🏆 <b>${esc(m.compName)}</b>\n`; lastComp = m.compName; }
    text += `${i + 1}. 🕒 ${fmtTime(m.utc)} ${esc(m.home)} vs ${esc(m.away)}\n${oddsLine(m)}\n`;
  });
  text += '\n👇 Tap a number for history + AI prediction.';
  const nums = slice.map((m, i) => Markup.button.callback(String(i + 1), `m:${m.id}:${day * 100 + page}`));
  const rows = [tabs, nums];
  if (pages > 1) {
    const nav = [];
    nav.push(page > 0 ? Markup.button.callback('◀️ Prev', `pd:${day}:${page - 1}`) : Markup.button.callback('·', 'noop'));
    nav.push(Markup.button.callback(`${page + 1} / ${pages}`, 'noop'));
    nav.push(page < pages - 1 ? Markup.button.callback('Next ▶️', `pd:${day}:${page + 1}`) : Markup.button.callback('·', 'noop'));
    rows.push(nav);
  }
  rows.push([Markup.button.callback('🔟 Best 10', 'best10'), Markup.button.callback('🏆 Top 5', 'top5'), Markup.button.callback('💎 Best 3', 'best3')]);
  rows.push([Markup.button.callback('📈 Value', 'value'), Markup.button.callback('🎯 Acca', 'acca:balanced:3'), Markup.button.callback('🔴 Live', 'live')]);
  rows.push([Markup.button.callback('🔄 Refresh', `rf:${day}:${page}`), MENU_BTN]);
  return { text, kb: Markup.inlineKeyboard(rows) };
}

// ----- match lookup (today's list or the 7-day list used by /predict) -----
function getUpcoming() {
  return cached('upcoming:' + localDate(new Date()), CACHE_MS, async () => {
    const from = localDate(new Date());
    const to = localDate(new Date(Date.now() + 7 * 864e5));
    const data = await fdGet(`/matches?dateFrom=${from}&dateTo=${to}`);
    const fixtures = (data.matches || [])
      .filter((m) => ['SCHEDULED', 'TIMED'].includes(m.status))
      .map((m) => ({
        id: m.id, comp: m.competition.code, compName: m.competition.name,
        home: m.homeTeam.shortName || m.homeTeam.name, away: m.awayTeam.shortName || m.awayTeam.name,
        homeFull: m.homeTeam.name, awayFull: m.awayTeam.name, homeId: m.homeTeam.id, awayId: m.awayTeam.id,
        utc: new Date(m.utcDate).getTime(), odds: null,
      }))
      .sort((a, b) => a.utc - b.utc);
    if (ODDS_KEY) {
      for (const c of [...new Set(fixtures.map((f) => f.comp))].filter((x) => COMP_TO_ODDS[x])) {
        try {
          const events = await cached('odds:' + c, CACHE_MS, () => fetchOdds(COMP_TO_ODDS[c]));
          for (const f of fixtures.filter((x) => x.comp === c)) f.odds = findOdds({ home: f.homeFull, away: f.awayFull, utc: f.utc }, events);
        } catch (e) { console.log('Odds fetch failed for', c, e.message); }
      }
    }
    return fixtures;
  });
}
async function findMatchAny(id) {
  const today = peek('today:' + localDate(new Date())) || (await getMatchesToday().catch(() => []));
  let m = today.find((x) => x.id === id);
  if (m) return m;
  const up = peek('upcoming:' + localDate(new Date())) || [];
  return up.find((x) => x.id === id) || null;
}
function teamHit(qs, f) {
  return Math.max(nameScore(qs, f.homeFull), nameScore(qs, f.awayFull), nameScore(qs, f.home), nameScore(qs, f.away));
}
async function findFixture(query) {
  const up = (await getUpcoming()).filter((f) => f.utc > Date.now() - 2 * 3600e3);
  const words = query.split(/\s+/).filter(Boolean);
  if (!words.length) return null;
  const sep = query.split(/\s+(?:vs?\.?|-|–)\s+/i);
  if (sep.length === 2) {
    const [a, b] = sep;
    return up.find((f) => (nameScore(a, f.homeFull) >= 0.99 && nameScore(b, f.awayFull) >= 0.99) || (nameScore(b, f.homeFull) >= 0.99 && nameScore(a, f.awayFull) >= 0.99)) || null;
  }
  const single = up.find((f) => teamHit(query, f) >= 0.99);
  if (single) return single;
  for (let i = 1; i < words.length; i++) { // "man city arsenal" -> try every split point
    const a = words.slice(0, i).join(' '), b = words.slice(i).join(' ');
    const f = up.find((x) => (nameScore(a, x.homeFull) >= 0.99 && nameScore(b, x.awayFull) >= 0.99) || (nameScore(b, x.homeFull) >= 0.99 && nameScore(a, x.awayFull) >= 0.99));
    if (f) return f;
  }
  return null;
}

// ----- match detail -----
async function renderDetailFor(m, backData, u) {
  const st = await aiRefine(m, await deepAnalysis(m));
  let tx = `<b>${esc(m.home)} vs ${esc(m.away)}</b>\n🏆 ${esc(m.compName)} · 🕒 ${fmtDate(m.utc)} ${fmtTime(m.utc)}\n\n`;
  if (m.odds) {
    const o = m.odds.odds, p = m.odds.probs;
    tx += `💰 <b>Odds</b> (avg of ${m.odds.n} bookmakers)\n1 <b>${o.H.toFixed(2)}</b> · X <b>${o.D.toFixed(2)}</b> · 2 <b>${o.A.toFixed(2)}</b>\n`;
    if (m.odds.best) {
      const bk = m.odds.bestBk || {};
      tx += `🏷 Best prices: 1 <b>${m.odds.best.H.toFixed(2)}</b>${bk.H ? ' (' + esc(bk.H) + ')' : ''} · X <b>${m.odds.best.D.toFixed(2)}</b>${bk.D ? ' (' + esc(bk.D) + ')' : ''} · 2 <b>${m.odds.best.A.toFixed(2)}</b>${bk.A ? ' (' + esc(bk.A) + ')' : ''}\n`;
    }
    tx += `📊 Market chance: ${pct(p.H)}% / ${pct(p.D)}% / ${pct(p.A)}%\n\n`;
  } else tx += '💰 Odds not available for this match.\n\n';

  const teamBlock = (name, s) =>
    `📈 <b>${esc(name)}</b> ${formStrip(s)}` +
    (s.ppg == null ? '' : ` (${s.ppg.toFixed(1)} pts/game)`) +
    (s.recent.length ? `\n   ${s.recent.map(esc).join(' · ')}` : '') + '\n';
  tx += teamBlock(m.home, st.hs) + teamBlock(m.away, st.as);
  if (st.h2h.length) tx += `🔁 H2H this season: ${st.h2h.map(esc).join('; ')}\n`;

  const pick = topPick(m, st.final);
  tx += '\n🔮 <b>Prediction</b>\n';
  if (pick) {
    const f = st.final;
    tx += `Home ${pct(f.H)}% · Draw ${pct(f.D)}% · Away ${pct(f.A)}%\n`;
    tx += `👉 Lean: <b>${esc(pick.label)}</b> (${pick.sign})${pick.odds ? ` @ ${pick.odds.toFixed(2)}` : ''}\n`;
    const r = reasonOf(m, st);
    if (r) tx += `💬 <i>${esc(r)}</i>\n`;
    if (st.mk) tx += `⚽ Expected ${st.form.lh.toFixed(1)}–${st.form.la.toFixed(1)} · likeliest ${st.mk.score} · O2.5 ${pct(st.mk.over25)}% · BTTS ${pct(st.mk.btts)}%\n`;
    tx += st.ai ? `🤖 AI: ${esc(st.ai.model)}\n` : AI_ON ? '🤖 AI unavailable right now — stats-only prediction.\n' : '';
    const sk = pick.odds ? stakeTxt(u, pick.p, m.odds.best ? m.odds.best[pick.k] : pick.odds) : '';
    if (sk) tx += sk.replace(/^\n   /, '') + '\n';
  } else tx += 'Not enough data to predict this one.\n';
  tx += '\n' + t(u, 'disc');
  const rows = [[Markup.button.callback('⭐ Watch', `w:${m.id}`), Markup.button.callback('🔁 H2H history', `h2h:${m.id}`)]];
  rows.push([Markup.button.callback((hasPremium(u) ? '' : '🔒 ') + '📋 All markets (GG, O/U, HT/FT, handicaps…)', `am:${m.id}`)]);
  if (backData) rows.push([Markup.button.callback('⬅️ Back to list', backData)]);
  rows.push([Markup.button.callback('🔟 Best 10', 'best10'), Markup.button.callback('🏆 Top 5', 'top5'), Markup.button.callback('💎 Best 3', 'best3'), MENU_BTN]);
  return { text: tx, kb: Markup.inlineKeyboard(rows) };
}

// ----- Top 5 / Best 3 -----
async function renderTop5(u, { full = true } = {}) {
  const all = (await computeTop5()).filter((x) => x.m.utc > Date.now());
  const top = applyPrefs(all, u);
  if (!top.length) return { text: all.length ? t(u, 'none_prefs') : t(u, 'none_qual'), kb: null, empty: true };
  const shown = full ? top : top.slice(0, FREE_PICKS);
  let tx = `🏆 <b>${t(u, 'top5_head')}</b>\n<i>${t(u, 'top5_sub')}</i>\n\n${FOLLOW_HINT}`;
  shown.forEach(({ m, st, sc }, i) => {
    const p = sc.pick;
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b>\n   🕒 ${fmtTime(m.utc)} · ${esc(m.compName)}\n`;
    tx += `   👉 <b>${esc(p.label)}</b> (${p.sign}) @ ${p.odds.toFixed(2)}${bestPriceTxt(m, p.k)} · ${dotOf(p.p)} ${pct(p.p)}% confidence\n`;
    const r = reasonOf(m, st);
    if (r) tx += `   <i>${esc(r)}</i>\n`;
    tx += stakeTxt(u, p.p, m.odds.best ? m.odds.best[p.k] : p.odds) + '\n';
  });
  let kb = null;
  if (shown.length < top.length) { tx += `🔒 ${t(u, 'locked_more', top.length - shown.length)}\n\n`; kb = upgradeKb(u); }
  return { text: tx + t(u, 'disc'), kb: followKb(shown, kb), empty: false };
}

async function renderBest10(u) {
  const all = (await computeBest10()).filter((x) => x.m.utc > Date.now());
  const list = applyPrefs(all, u);
  if (!list.length) return { text: all.length ? t(u, 'none_prefs') : t(u, 'none_qual'), kb: null, empty: true };
  let tx = `🔟 <b>Best 10 picks today</b>\n<i>Free for everyone. Ranked by confidence, source agreement and value vs the market. The sharpest picks are in Premium (/top5 · /best3 · /best1).</i>\n\n${FOLLOW_HINT}`;
  list.forEach(({ m, st, sc }, i) => {
    const p = sc.pick;
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b>\n   🕒 ${fmtTime(m.utc)} · ${esc(m.compName)}\n`;
    tx += `   👉 <b>${esc(p.label)}</b> (${p.sign}) @ ${p.odds.toFixed(2)}${bestPriceTxt(m, p.k)} · ${dotOf(p.p)} ${pct(p.p)}% confidence\n`;
    const r = reasonOf(m, st);
    if (r) tx += `   <i>${esc(r)}</i>\n`;
    tx += stakeTxt(u, p.p, m.odds.best ? m.odds.best[p.k] : p.odds) + '\n';
  });
  return { text: tx + t(u, 'disc'), kb: followKb(list, u && hasPremium(u) ? null : quickBuyKb()), empty: false };
}

async function renderBest3(u, { full = true, only1 = false } = {}) {
  const b3 = await computeBest3();
  const { skipped, considered } = b3;
  const all = b3.list.filter((x) => x.m.utc > Date.now());
  const list = applyPrefs(all, u);
  const rule = `no top-${TOP_RANK_EXCLUDE} table teams, no big-name clubs`;
  if (!list.length)
    return { text: all.length ? t(u, 'none_prefs') : `😴 No under-the-radar match with usable odds right now (${considered} with odds, ${skipped} skipped because a top/popular club plays). Odds for later kickoffs usually appear closer to match time, so try again a bit later, or use /top5.\n\n` + t(u, 'disc'), kb: null, empty: true };
  const shown = only1 ? list.slice(0, 1) : full ? list : list.slice(0, FREE_PICKS);
  let tx = only1
    ? `🥇 <b>Best pick of the day</b>\n<i>The most likely result among under-the-radar matches (${rule}).</i>\n\n`
    : `💎 <b>${t(u, 'best3_head')}</b>\n<i>${rule}. Ranked by confidence, source agreement and value vs the market. 🥇 = best pick of the day.</i>\n`;
  if (!only1) {
    if (list.length < 3) tx += `<i>Only ${list.length} match${list.length > 1 ? 'es' : ''} had usable odds today.</i>\n`;
    tx += '\n';
  }
  tx += FOLLOW_HINT;
  const medal = ['🥇', '🥈', '🥉'];
  shown.forEach(({ m, st, sc, tier }, i) => {
    const p = sc.pick;
    const k = p.k;
    tx += `${medal[i] || i + 1 + '.'} <b>${esc(m.home)} vs ${esc(m.away)}</b>\n   🕒 ${fmtTime(m.utc)} · ${esc(m.compName)}\n`;
    tx += `   👉 <b>${esc(p.label)}</b> (${p.sign}) @ ${p.odds.toFixed(2)}${bestPriceTxt(m, k)} · ${dotOf(p.p)} ${pct(p.p)}% confidence · ${sc.agreeN}/${sc.total} sources agree\n`;
    tx += `   📊 Market ${pct(m.odds.probs[k])}%` + (st.form ? ` · Form model ${pct(st.form[k])}%` : '') + (st.ai ? ` · AI ${pct(st.ai.probs[k])}%` : '') + '\n';
    const tl = (name, pr) =>
      `${esc(name)} #${pr.rank || '?'}` + (pr.venue.n ? ` (${f1(pr.venue.ppg)} pts/g ${pr === st.hp ? 'home' : 'away'}, ${f1(pr.venue.gf)}-${f1(pr.venue.ga)} goals/g)` : '');
    if (st.hp.rank || st.ap.rank || st.hp.venue.n || st.ap.venue.n) tx += `   📈 ${tl(m.home, st.hp)} | ${tl(m.away, st.ap)}\n`;
    if (st.mk) tx += `   ⚽ Expected ${st.form.lh.toFixed(1)}–${st.form.la.toFixed(1)} · likeliest ${st.mk.score} · O2.5 ${pct(st.mk.over25)}% · BTTS ${pct(st.mk.btts)}%\n`;
    if (tier === 1) tx += `   ⚠️ <i>Little form data for this league, so this leans mostly on bookmaker odds.</i>\n`;
    if (tier === 2) tx += `   ⚠️ <i>Lower-confidence pick: few safe options today.</i>\n`;
    const r = reasonOf(m, st);
    if (r) tx += `   <i>${esc(r)}</i>\n`;
    tx += stakeTxt(u, p.p, m.odds.best ? m.odds.best[k] : p.odds) + '\n';
  });
  let kb = null;
  if (!only1 && shown.length < list.length) { tx += `🔒 ${t(u, 'locked_more', list.length - shown.length)}\n\n`; kb = upgradeKb(u); }
  return { text: tx + t(u, 'disc'), kb: followKb(shown, kb), empty: false };
}

// ----- Value bets: model probability vs the BEST bookmaker price -----
function computeValue() {
  return cached('value:' + localDate(new Date()), CACHE_MS, async () => {
    await refineAll().catch(() => {});
    const now = Date.now();
    const matches = (await getMatchesToday()).filter((m) => m.odds && m.odds.best && m.utc > now);
    const out = [];
    for (const m of matches) {
      const st = bestKnown(m, await deepAnalysis(m));
      if (!st.final) continue;
      let best = null;
      for (const k of ['H', 'D', 'A']) {
        const o = m.odds.best[k], p = st.final[k];
        if (!o || o > VALUE_MAX_ODDS || o < 1.2) continue;
        const ev = p * o - 1, edge = p - m.odds.probs[k];
        if (ev < VALUE_MIN_EV || edge < VALUE_MIN_EDGE) continue;
        if (!best || ev > best.ev) best = { k, p, o, ev, edge };
      }
      if (best) out.push({ m, st, ...best });
    }
    return out.sort((a, b) => b.ev - a.ev).slice(0, 8);
  });
}
async function renderValue(u) {
  const list = byLeague(await computeValue(), u).filter((x) => x.m.utc > Date.now());
  if (!list.length) return `📈 <b>Value bets</b>\n\nNo clear value today — the market and our model mostly agree (edge ≥ ${pct(VALUE_MIN_EDGE)} pts and EV ≥ +${pct(VALUE_MIN_EV)}% needed).\n\n` + t(u, 'disc');
  let tx = '📈 <b>Value bets</b>\n<i>Model probability vs the best bookmaker price. Edge = our % − market %. EV = expected profit per 1 staked if the model is right.</i>\n\n';
  list.forEach(({ m, k, p, o, ev, edge }, i) => {
    const label = { H: `${m.home} win`, D: 'Draw', A: `${m.away} win` }[k];
    const bk = m.odds.bestBk && m.odds.bestBk[k];
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b>\n   🕒 ${fmtTime(m.utc)} · ${esc(m.compName)}\n`;
    tx += `   👉 <b>${esc(label)}</b> @ <b>${o.toFixed(2)}</b>${bk ? ' (' + esc(bk) + ')' : ''}\n`;
    tx += `   📊 Model ${pct(p)}% vs market ${pct(m.odds.probs[k])}% → edge ${signed(edge * 100, 1)} pts · EV ${signed(ev * 100, 0)}%` + stakeTxt(u, p, o) + '\n\n';
  });
  return tx + '<i>Edges are model-vs-market estimates. The market is usually sharper than any model — treat these as small-stake ideas.</i>\n\n' + t(u, 'disc');
}

// ----- Accumulator builder -----
async function accaPool() {
  await refineAll().catch(() => {});
  const now = Date.now();
  const matches = (await getMatchesToday()).filter((m) => m.odds && m.utc > now);
  const out = [];
  for (const m of matches) {
    const st = bestKnown(m, await deepAnalysis(m));
    const sc = st.final ? pickScore(m, st) : null;
    if (sc) out.push({ m, st, sc });
  }
  return out;
}
async function renderAcca(u, mode = 'balanced', legsN = 3) {
  mode = ['safe', 'balanced', 'risky'].includes(mode) ? mode : 'balanced';
  legsN = Math.max(2, Math.min(6, legsN || 3));
  let pool = byLeague(await accaPool(), u);
  if (mode === 'safe') pool = pool.filter((x) => x.sc.pick.p >= 0.55).sort((a, b) => b.sc.pick.p - a.sc.pick.p);
  else if (mode === 'risky') pool = pool.filter((x) => x.sc.pick.p >= 0.35 && x.sc.pick.odds >= 1.8).sort((a, b) => b.sc.pick.p * b.sc.pick.odds - a.sc.pick.p * a.sc.pick.odds);
  else pool = pool.sort((a, b) => b.sc.score - a.sc.score);
  const legs = pool.slice(0, legsN);
  const kb = Markup.inlineKeyboard([
    ['safe', 'balanced', 'risky'].map((md) => Markup.button.callback((md === mode ? '• ' : '') + md[0].toUpperCase() + md.slice(1), `acca:${md}:${legsN}`)),
    [2, 3, 4, 5].map((n) => Markup.button.callback((n === legsN ? '• ' : '') + n + ' legs', `acca:${mode}:${n}`)),
    [Markup.button.callback('🔟 Best 10', 'best10'), Markup.button.callback('🏆 Top 5', 'top5'), Markup.button.callback('⚽ Today', 'today'), MENU_BTN],
  ]);
  if (legs.length < 2) return { text: `🎯 <b>Accumulator (${mode})</b>\n\nNot enough qualifying matches today for a ${mode} acca. Try another style or check back later.\n\n` + t(u, 'disc'), kb };
  let odds = 1, bestOdds = 1, prob = 1;
  let tx = `🎯 <b>${legs.length}-leg accumulator — ${mode}</b>\n\n`;
  legs.forEach(({ m, sc }, i) => {
    const p = sc.pick;
    odds *= p.odds; prob *= p.p; bestOdds *= (m.odds.best ? m.odds.best[p.k] : p.odds);
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)}\n   👉 ${esc(p.label)} (${p.sign}) @ ${p.odds.toFixed(2)} · ${pct(p.p)}%\n`;
  });
  const ev = prob * odds - 1;
  tx += `\n📦 Combined odds (avg prices): <b>${odds.toFixed(2)}</b> · at best prices: <b>${bestOdds.toFixed(2)}</b>\n`;
  tx += `🎲 Chance all ${legs.length} land: <b>${(prob * 100).toFixed(1)}%</b> · EV ${signed(ev * 100, 0)}%` + stakeTxt(u, prob, odds) + '\n\n';
  tx += '<i>Legs are treated as independent. Accas multiply the bookmaker margin, so even "safe" ones lose more often than a single bet — keep stakes small.</i>\n\n' + t(u, 'disc');
  return { text: tx, kb };
}

// ----- Goals markets & safer markets (from the goal model) -----
async function analysedToday() {
  await refineAll().catch(() => {});
  const now = Date.now();
  const out = [];
  for (const m of (await getMatchesToday()).filter((x) => x.utc > now)) {
    const st = bestKnown(m, await deepAnalysis(m));
    if (st.final) out.push({ m, st });
  }
  return out;
}
const fair = (p) => (p > 0 ? (1 / p).toFixed(2) : 'n/a');
async function renderGoals(u) {
  const rows = [];
  for (const { m, st } of byLeague(await analysedToday(), u)) {
    if (!st.mk || !st.form) continue;
    const opts = [
      ['Over 2.5 goals', st.mk.over25], ['Under 2.5 goals', 1 - st.mk.over25],
      ['Both teams score', st.mk.btts], ['Both teams NOT to score', 1 - st.mk.btts],
    ].sort((a, b) => b[1] - a[1]);
    if (opts[0][1] >= 0.58) rows.push({ m, st, label: opts[0][0], p: opts[0][1] });
  }
  rows.sort((a, b) => b.p - a.p);
  if (!rows.length) return '⚽ <b>Goals markets</b>\n\nNo goal-market pick clears the 58% bar today.\n\n' + t(u, 'disc');
  let tx = '⚽ <b>Goals markets</b>\n<i>From the Dixon-Coles goal model. "Fair odds" = 1 ÷ probability — only bet if your bookie pays MORE than that.</i>\n\n';
  rows.slice(0, 8).forEach(({ m, st, label, p }, i) => {
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)} · ${esc(m.compName)}\n   👉 <b>${label}</b> — ${pct(p)}% (fair odds ${fair(p)})\n   Expected goals ${st.form.lh.toFixed(1)}–${st.form.la.toFixed(1)} · likeliest ${st.mk.score}\n\n`;
  });
  return tx + t(u, 'disc');
}
async function renderDC(u) {
  const rows = [];
  for (const { m, st } of byLeague(await analysedToday(), u)) {
    const f = st.final;
    const dcs = [['1X', `${m.home} or Draw`, f.H + f.D], ['X2', `${m.away} or Draw`, f.D + f.A], ['12', `${m.home} or ${m.away} (no draw)`, f.H + f.A]].sort((a, b) => b[2] - a[2]);
    const top = Math.max(f.H, f.D, f.A);
    if (dcs[0][2] >= 0.74 && top < 0.55) {
      const side = f.H >= f.A ? 'H' : 'A';
      const dnb = f[side] / (f.H + f.A);
      rows.push({ m, dc: dcs[0], side, dnb, top });
    }
  }
  rows.sort((a, b) => b.dc[2] - a.dc[2]);
  if (!rows.length) return '🛡 <b>Safer markets</b>\n\nNo match today where a double chance clearly beats a risky 1X2 pick.\n\n' + t(u, 'disc');
  let tx = '🛡 <b>Safer markets — Double Chance & Draw No Bet</b>\n<i>For matches where the straight 1X2 pick is under 55%. Fair odds = 1 ÷ probability.</i>\n\n';
  rows.slice(0, 8).forEach(({ m, dc, side, dnb }, i) => {
    const team = side === 'H' ? m.home : m.away;
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)} · ${esc(m.compName)}\n   🛡 Double chance <b>${dc[0]}</b> (${esc(dc[1])}) — ${pct(dc[2])}% · fair ${fair(dc[2])}\n   ↩️ Draw No Bet <b>${esc(team)}</b> — wins ${pct(dnb)}% of non-draws · fair ${fair(dnb)} (stake back on a draw)\n\n`;
  });
  return tx + t(u, 'disc');
}


// ===================== FULL MARKET ENGINE =====================
// Every market below comes from ONE score grid (Dixon-Coles), re-weighted so its 1X2 matches the blended final probabilities.
// Halves use independent Poisson grids. Only 1X2 has real bookmaker odds, so everything else shows model "fair odds" (1 ÷ probability).
const HT_SHARE = Number(process.env.HT_SHARE || 0.45);        // share of a match's goals scored before half-time
const CORNERS_AVG = Number(process.env.CORNERS_AVG || 10);    // league-average total corners per match (no live corner feed)
const CARDS_AVG = Number(process.env.CARDS_AVG || 4.2);       // league-average bookings per match (yellow = 1, red = 1)
const MK_MIN_P = Number(process.env.MK_MIN_P || 0.55);        // default band shown on the daily market boards
const MK_MAX_P = Number(process.env.MK_MAX_P || 0.82);        // above this the price is too short to matter

function tiltGrid(grid, f) { // re-weight score cells so P(home win / draw / away win) equals f
  const G = { H: 0, D: 0, A: 0 };
  grid.forEach((row, i) => row.forEach((p, j) => { G[i > j ? 'H' : i === j ? 'D' : 'A'] += p; }));
  let tot = 0;
  const out = grid.map((row, i) => row.map((p, j) => {
    const k = i > j ? 'H' : i === j ? 'D' : 'A';
    const v = G[k] > 0 ? p * (f[k] / G[k]) : p;
    tot += v;
    return v;
  }));
  return out.map((row) => row.map((p) => p / tot));
}
const gridSum = (grid, fn) => { let s = 0; grid.forEach((row, i) => row.forEach((p, j) => { if (fn(i, j)) s += p; })); return s; };
const transpose = (grid) => grid[0].map((_, j) => grid.map((row) => row[j]));
function diffDist(grid) { // distribution of (home - away) goals
  const d = new Map();
  grid.forEach((row, i) => row.forEach((p, j) => d.set(i - j, (d.get(i - j) || 0) + p)));
  return d;
}
function ahProb(grid, line) { // Asian handicap applied to the HOME side. Returns win / push / lose (quarter lines = half stake on each neighbour)
  const one = (h) => {
    let w = 0, pu = 0, l = 0;
    grid.forEach((row, i) => row.forEach((p, j) => { const d = i - j + h; if (d > 1e-9) w += p; else if (d < -1e-9) l += p; else pu += p; }));
    return { w, pu, l };
  };
  const q = Math.round(line * 4) / 4;
  if (Math.abs(q * 2 - Math.round(q * 2)) < 1e-9) return one(q);
  const a = one(q - 0.25), b = one(q + 0.25);
  return { w: (a.w + b.w) / 2, pu: (a.pu + b.pu) / 2, l: (a.l + b.l) / 2 };
}
function nbOver(mu, disp, line) { // P(total > line) for corners / cards: negative binomial (over-dispersed vs Poisson)
  const r = mu / (disp - 1);
  let pk = Math.pow(r / (r + mu), r), cum = pk;
  for (let k = 0; k < Math.floor(line); k++) { pk = (pk * (k + r) / (k + 1)) * (mu / (r + mu)); cum += pk; }
  return 1 - cum;
}
const signOf = (d) => (d > 0 ? '1' : d < 0 ? '2' : 'X');
const fmtLine = (x) => (x > 0 ? '+' : '') + x;

// GROUP_BAND: [min, max] probability a pick must sit in to appear on a daily board (low-probability groups get a lower floor)
const GROUP_BAND = {
  x12: [MK_MIN_P, MK_MAX_P], dc: [0.68, MK_MAX_P], dnb: [MK_MIN_P, MK_MAX_P],
  ou: [MK_MIN_P, MK_MAX_P], bts: [MK_MIN_P, MK_MAX_P], team: [MK_MIN_P, MK_MAX_P], multi: [0.5, MK_MAX_P],
  htft: [0.2, 0.9], half: [0.45, MK_MAX_P], weh: [MK_MIN_P, MK_MAX_P], sbh: [0.35, MK_MAX_P],
  cgg: [0.35, 0.8], cov: [0.35, 0.8],
  eu: [0.45, MK_MAX_P], ah: [MK_MIN_P, MK_MAX_P],
};
const GROUP_TITLE = { x12: '1X2', dc: 'Double chance', dnb: 'Draw no bet', ou: 'Over / Under', bts: 'Both teams to score', team: 'Team goals', multi: 'Multi-goals', htft: 'HT/FT', half: 'Half result', weh: 'Win either half', sbh: 'Score both halves', cgg: 'Result & BTTS', cov: 'Result & goals', eu: 'Handicap (3-way)', ah: 'Asian handicap' };
const MK_CATS = {
  res: { short: 'Result', icon: '🏁', title: 'Result markets', sub: '1X2 · Double Chance · Draw No Bet', groups: ['x12', 'dc', 'dnb'] },
  goals: { short: 'Goals', icon: '⚽', title: 'Goals markets', sub: 'GG/NG · Over/Under 0.5–3.5 · team goals · multi-goals', groups: ['ou', 'bts', 'team', 'multi'] },
  htft: { short: 'HT/FT', icon: '⏱', title: 'Half-time / Full-time', sub: 'HT/FT · HT · 2HT · win either half · score both halves', groups: ['htft', 'half', 'weh', 'sbh'] },
  combo: { short: 'Combos', icon: '🧩', title: 'Combo markets', sub: 'Result & GG · Result & Over/Under', groups: ['cgg', 'cov'] },
  hcp: { short: 'Handicaps', icon: '📐', title: 'Handicaps', sub: '1 (-1) · 2 (+1) · Asian handicap', groups: ['eu', 'ah'] },
};

const lamCache = new Map();
function fitLambdas(f) { // goal rates that reproduce a 1X2 split (used when a match has odds but no team-stats data)
  const key = [f.H, f.D, f.A].map((x) => x.toFixed(3)).join('|');
  if (lamCache.has(key)) return lamCache.get(key);
  let best = null;
  for (let lh = 0.2; lh <= 3.6; lh += 0.1) for (let la = 0.2; la <= 3.0; la += 0.1) {
    const p = poisson1x2(lh, la);
    const e = (p.H - f.H) ** 2 + (p.D - f.D) ** 2 + (p.A - f.A) ** 2 + 0.002 * (lh + la - 2.7) ** 2;
    if (!best || e < best.e) best = { e, lh, la };
  }
  const out = { lh: best.lh, la: best.la };
  if (lamCache.size > 500) lamCache.clear();
  lamCache.set(key, out);
  return out;
}
function buildMarkets(m, st) {
  if (!st) return null;
  const approx = !st.form;
  const rates = st.form || (st.final ? fitLambdas(st.final) : null);
  if (!rates) return null;
  const { lh, la } = rates;
  let grid = scoreGrid(lh, la);
  if (st.final) grid = tiltGrid(grid, st.final);
  const H = m.home, A = m.away;
  const mk = [];
  const add = (cat, group, code, desc, p, extra = {}) => mk.push({ cat, group, code, desc, p, ...extra });
  const P = (fn) => gridSum(grid, fn);
  const tg = (i, j) => i + j;

  // 1) result, double chance, DNB
  const pH = P((i, j) => i > j), pD = P((i, j) => i === j), pA = P((i, j) => i < j);
  add('res', 'x12', '1', `${H} win`, pH); add('res', 'x12', 'X', 'Draw', pD); add('res', 'x12', '2', `${A} win`, pA);
  add('res', 'dc', '1X', `${H} or Draw`, pH + pD); add('res', 'dc', 'X2', `${A} or Draw`, pD + pA); add('res', 'dc', '12', 'Either team wins (no draw)', pH + pA);
  add('res', 'dnb', 'DNB 1', `${H} (stake back on a draw)`, pH / (pH + pA), { push: pD });
  add('res', 'dnb', 'DNB 2', `${A} (stake back on a draw)`, pA / (pH + pA), { push: pD });

  // 2) goals
  const gg = P((i, j) => i > 0 && j > 0);
  add('goals', 'bts', 'GG', 'Both teams score', gg); add('goals', 'bts', 'NG', 'Both teams do not score', 1 - gg);
  for (const L of [0.5, 1.5, 2.5, 3.5]) {
    const ov = P((i, j) => tg(i, j) > L);
    add('goals', 'ou', `OV ${L}`, `Over ${L} goals (${Math.floor(L) + 1}+)`, ov, { line: L });
    add('goals', 'ou', `UN ${L}`, `Under ${L} goals (${Math.floor(L)} or fewer)`, 1 - ov, { line: L });
  }
  for (const L of [0.5, 1.5]) {
    add('goals', 'team', `Home OV ${L}`, `${H} score over ${L}`, P((i) => i > L));
    add('goals', 'team', `Away OV ${L}`, `${A} score over ${L}`, P((i, j) => j > L));
  }
  for (const [a, b] of [[1, 3], [2, 4], [2, 5]]) add('goals', 'multi', `${a}–${b} goals`, `Total goals between ${a} and ${b}`, P((i, j) => tg(i, j) >= a && tg(i, j) <= b));

  // 3) halves
  const g1 = scoreGrid(lh * HT_SHARE, la * HT_SHARE), g2 = scoreGrid(lh * (1 - HT_SHARE), la * (1 - HT_SHARE));
  const d1 = diffDist(g1), d2 = diffDist(g2);
  const htft = {};
  for (const [a, pa] of d1) for (const [b, pb] of d2) { const k = signOf(a) + '/' + signOf(a + b); htft[k] = (htft[k] || 0) + pa * pb; }
  for (const k of ['1/1', 'X/1', '2/1', '1/X', 'X/X', '2/X', '1/2', 'X/2', '2/2']) {
    const [a, b] = k.split('/');
    const name = (s) => (s === '1' ? H : s === '2' ? A : 'draw');
    add('htft', 'htft', k, `HT ${name(a)} → FT ${name(b)}`, htft[k] || 0);
  }
  const half = (g, tag, lab) => {
    const h = gridSum(g, (i, j) => i > j), d = gridSum(g, (i, j) => i === j), a = gridSum(g, (i, j) => i < j);
    add('htft', 'half', `${tag} 1`, `${lab}: ${H} ahead`, h); add('htft', 'half', `${tag} X`, `${lab}: level`, d); add('htft', 'half', `${tag} 2`, `${lab}: ${A} ahead`, a);
  };
  half(g1, 'HT', 'First half'); half(g2, '2HT', 'Second half');
  const le = (g, f) => gridSum(g, f);
  add('htft', 'weh', 'HWEH', `${H} win either half`, 1 - le(g1, (i, j) => i <= j) * le(g2, (i, j) => i <= j));
  add('htft', 'weh', 'AWEH', `${A} win either half`, 1 - le(g1, (i, j) => i >= j) * le(g2, (i, j) => i >= j));
  add('htft', 'sbh', 'SBH Home', `${H} score in both halves`, le(g1, (i) => i > 0) * le(g2, (i) => i > 0));
  add('htft', 'sbh', 'SBH Away', `${A} score in both halves`, le(g1, (i, j) => j > 0) * le(g2, (i, j) => j > 0));

  // 4) combos: result × goals filter
  const RES = { '1': (i, j) => i > j, X: (i, j) => i === j, '2': (i, j) => i < j, '1X': (i, j) => i >= j, X2: (i, j) => i <= j, '12': (i, j) => i !== j };
  const RDESC = { '1': `${H} win`, X: 'Draw', '2': `${A} win`, '1X': `${H} or draw`, X2: `${A} or draw`, '12': 'either team wins' };
  const FIL = {
    GG: ['cgg', (i, j) => i > 0 && j > 0, 'both teams score'], NG: ['cgg', (i, j) => i === 0 || j === 0, 'not both teams score'],
    'OV 1.5': ['cov', (i, j) => tg(i, j) > 1.5, '2+ goals'], 'OV 2.5': ['cov', (i, j) => tg(i, j) > 2.5, '3+ goals'],
    'UN 2.5': ['cov', (i, j) => tg(i, j) < 2.5, '2 or fewer goals'], 'UN 3.5': ['cov', (i, j) => tg(i, j) < 3.5, '3 or fewer goals'],
  };
  for (const [rk, rf] of Object.entries(RES)) for (const [fk, [grp, ff, fd]] of Object.entries(FIL)) {
    add('combo', grp, `${rk} & ${fk}`, `${RDESC[rk]} AND ${fd}`, P((i, j) => rf(i, j) && ff(i, j)));
  }

  // 5) handicaps
  add('hcp', 'eu', '1 (-1)', `${H} win by 2+ goals`, P((i, j) => i - j >= 2));
  add('hcp', 'eu', '2 (-1)', `${A} win by 2+ goals`, P((i, j) => j - i >= 2));
  add('hcp', 'eu', '2 (+1)', `${A} win or draw (starts +1)`, P((i, j) => i - j <= 0));
  add('hcp', 'eu', '1 (+1)', `${H} win or draw (starts +1)`, P((i, j) => j - i <= 0));
  const gridT = transpose(grid);
  for (const [side, g, name] of [['Home', grid, H], ['Away', gridT, A]]) {
    for (const line of [-1.5, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 1.5]) {
      const r = ahProb(g, line);
      add('hcp', 'ah', `AH ${side} ${fmtLine(line)}`, `${name} ${fmtLine(line)}`, r.w + r.l > 0 ? r.w / (r.w + r.l) : 0, { push: r.pu, win: r.w });
    }
  }

  // 6) corners & cards (rough: league-average totals nudged by expected goals — there is no corner/card data feed)
  const tilt = Math.max(-0.12, Math.min(0.12, 0.05 * (lh + la - 2.7)));
  const cMu = CORNERS_AVG * (1 + tilt), kMu = CARDS_AVG * (1 - tilt / 2);
  for (const L of [8.5, 9.5, 10.5, 11.5]) { const o = nbOver(cMu, 1.5, L); add('cc', 'corners', `Corners OV ${L}`, `${Math.floor(L) + 1}+ corners`, o); add('cc', 'corners', `Corners UN ${L}`, `${Math.floor(L)} or fewer corners`, 1 - o); }
  for (const L of [2.5, 3.5, 4.5, 5.5]) { const o = nbOver(kMu, 1.3, L); add('cc', 'cards', `Cards OV ${L}`, `${Math.floor(L) + 1}+ bookings`, o); add('cc', 'cards', `Cards UN ${L}`, `${Math.floor(L)} or fewer bookings`, 1 - o); }
  return { mk, lh, la, grid, cMu, kMu, approx };
}

const mkLine = (x) => `<b>${esc(x.code)}</b> ${esc(x.desc)} — <b>${pct(x.p)}%</b> · fair ${fair(x.p)}` + (x.push != null && x.push >= 0.05 ? ` · stake back ${pct(x.push)}%` : '');

// Daily board: for each of today's matches the best pick in each group of the category (inside its probability band)
async function renderMarketBoard(u, cat) {
  const C = MK_CATS[cat];
  const rows = [];
  const all = await analysedToday();
  const mine = byLeague(all, u);
  let modelled = 0;
  for (const { m, st } of mine) {
    const b = buildMarkets(m, st);
    if (!b) continue;
    modelled++;
    const lines = [];
    for (const g of C.groups) {
      const [lo, hi] = GROUP_BAND[g] || [MK_MIN_P, MK_MAX_P];
      const best = b.mk.filter((x) => x.cat === cat && x.group === g && x.p >= lo && x.p <= hi).sort((a, c) => c.p - a.p)[0];
      if (best) lines.push({ g, x: best });
    }
    if (lines.length) rows.push({ m, st, b, lines, score: Math.max(...lines.map((l) => l.x.p / (GROUP_BAND[l.g] ? GROUP_BAND[l.g][0] : 1))) });
  }
  rows.sort((a, c) => c.score - a.score);
  if (!rows.length) {
    const why = !all.length ? 'No matches left to play today — everything has kicked off or been played. Check back when tomorrow\'s fixtures load (or open /today).'
      : !mine.length ? `${all.length} match${all.length === 1 ? '' : 'es'} left today, but none in your selected leagues. Change that in ⚙️ Settings → Leagues.`
      : !modelled ? 'The remaining matches have no odds or team data yet, so they cannot be priced.'
      : `${modelled} match${modelled === 1 ? '' : 'es'} priced, but none has a market inside the ${C.short} probability band right now.`;
    return `${C.icon} <b>${C.title}</b>\n\n${why}\n\n` + t(u, 'disc');
  }
  let tx = `${C.icon} <b>${C.title}</b>\n<i>${C.sub}. Chances come from the goal model tuned to the blended 1X2. Fair odds = 1 ÷ chance — only bet if your bookie pays MORE. Tap a match below for every market.</i>\n\n`;
  const shown = rows.slice(0, 8);
  shown.forEach(({ m, b, lines }, i) => {
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)} · ${esc(m.compName)}\n`;
    for (const { g, x } of lines) tx += `   ▸ <i>${GROUP_TITLE[g]}</i> · ${mkLine(x)}\n`;
    tx += `   ⚽ Expected goals ${b.approx ? '≈' : ''}${b.lh.toFixed(1)}–${b.la.toFixed(1)}\n\n`;
  });
  if (shown.some((r) => r.b.approx)) tx += '<i>≈ = goal rates inferred from the bookmaker 1X2 odds (no team-stats data for that match).</i>\n\n';
  const nums = shown.map((r, i) => Markup.button.callback(`${i + 1} 📋`, `am:${r.m.id}`));
  const kbRows = [];
  for (let i = 0; i < nums.length; i += 4) kbRows.push(nums.slice(i, i + 4));
  const others = Object.keys(MK_CATS).filter((c) => c !== cat).map((c) => Markup.button.callback(`${MK_CATS[c].icon} ${MK_CATS[c].short}`, `mk:${c}`));
  for (let i = 0; i < others.length; i += 2) kbRows.push(others.slice(i, i + 2));
  return { text: tx + t(u, 'disc'), kb: Markup.inlineKeyboard(kbRows) };
}

// Every market for ONE match
async function renderAllMarkets(m, u) {
  const st = bestKnown(m, await deepAnalysis(m));
  const b = buildMarkets(m, st);
  let tx = `📋 <b>All markets</b>\n<b>${esc(m.home)} vs ${esc(m.away)}</b>\n🏆 ${esc(m.compName)} · 🕒 ${fmtDate(m.utc)} ${fmtTime(m.utc)}\n`;
  if (!b) return { text: tx + '\nNot enough data to price this match.\n\n' + t(u, 'disc'), kb: null };
  const g = (grp, cat) => b.mk.filter((x) => x.group === grp && (!cat || x.cat === cat));
  const row = (arr, f = (x) => `${x.code} ${pct(x.p)}%`) => arr.map(f).join(' · ');
  tx += `⚽ Expected goals ${b.approx ? '≈' : ''}${b.lh.toFixed(1)}–${b.la.toFixed(1)}${b.approx ? ' <i>(inferred from the 1X2 odds — no team-stats data)</i>' : ''}\n<i>Fair odds = 100 ÷ chance (50% → 2.00). Bet only if the bookie pays more.</i>\n`;
  tx += `\n🏁 <b>Result</b>\n1X2: ${row(g('x12'))}\nDouble chance: ${row(g('dc'))}\nDraw no bet: ${row(g('dnb'), (x) => `${x.code} ${pct(x.p)}%`)} <i>(stake back on a draw ${pct(g('dnb')[0].push)}%)</i>\n`;
  tx += `\n⚽ <b>Goals</b>\nBTTS: ${row(g('bts'))}\n`;
  for (const L of [0.5, 1.5, 2.5, 3.5]) { const o = b.mk.find((x) => x.code === `OV ${L}`), un = b.mk.find((x) => x.code === `UN ${L}`); tx += `${L}: Over ${pct(o.p)}% (fair ${fair(o.p)}) · Under ${pct(un.p)}% (fair ${fair(un.p)})\n`; }
  tx += `Team goals: ${row(g('team'), (x) => `${x.code.replace('Home', esc(m.home)).replace('Away', esc(m.away))} ${pct(x.p)}%`)}\nMulti-goals: ${row(g('multi'))}\n`;
  tx += `\n⏱ <b>Half-time / Full-time</b>\n${row(g('htft').slice(0, 3))}\n${row(g('htft').slice(3, 6))}\n${row(g('htft').slice(6, 9))}\n`;
  tx += `First half: ${row(g('half').slice(0, 3), (x) => `${x.code.replace('HT ', '')} ${pct(x.p)}%`)}\nSecond half: ${row(g('half').slice(3), (x) => `${x.code.replace('2HT ', '')} ${pct(x.p)}%`)}\n`;
  tx += `Win either half: ${row(g('weh'))}\nScore both halves: ${row(g('sbh'), (x) => `${x.code.replace('SBH ', '')} ${pct(x.p)}%`)}\n`;
  const pick = (codes) => codes.map((c) => b.mk.find((x) => x.code === c)).filter(Boolean);
  tx += `\n🧩 <b>Combos</b>\n` + pick(['1 & GG', '2 & GG', 'X & GG', '1 & OV 2.5', '2 & OV 2.5', '1X & UN 3.5', 'X2 & UN 3.5']).map((x) => `${x.code} ${pct(x.p)}% (fair ${fair(x.p)})`).join('\n') + '\n';
  tx += `\n📐 <b>Handicaps</b>\n${row(g('eu'), (x) => `${x.code} ${pct(x.p)}%`)}\n`;
  const ah = g('ah');
  for (const side of ['Home', 'Away']) {
    tx += `AH ${side === 'Home' ? esc(m.home) : esc(m.away)}: ` + ah.filter((x) => x.code.startsWith(`AH ${side} `)).filter((x) => ['-1.5', '-1', '-0.5', '+0.5', '+1', '+1.5'].includes(x.code.split(' ')[2])).map((x) => `${x.code.split(' ')[2]} ${pct(x.win)}%${x.push >= 0.05 ? ' (push ' + pct(x.push) + '%)' : ''}`).join(' · ') + '\n';
  }
  tx += `<i>AH shows the chance the bet fully wins; on whole lines a push returns your stake. Quarter lines (±0.25 / ±0.75) are on the Handicaps board.</i>\n`;
  tx += `\n🟨 <b>Corners &amp; cards</b> <i>(rough — league averages, no live corner/card feed)</i>\nCorners (≈${b.cMu.toFixed(1)}): ` + [8.5, 9.5, 10.5, 11.5].map((L) => `OV ${L} ${pct(b.mk.find((x) => x.code === `Corners OV ${L}`).p)}%`).join(' · ') + '\n';
  tx += `Cards (≈${b.kMu.toFixed(1)}): ` + [2.5, 3.5, 4.5, 5.5].map((L) => `OV ${L} ${pct(b.mk.find((x) => x.code === `Cards OV ${L}`).p)}%`).join(' · ') + '\n';
  tx += '\n' + t(u, 'disc');
  const kb = Markup.inlineKeyboard([[Markup.button.callback('🔮 Full prediction', `m:${m.id}:0`), Markup.button.callback('📈 Markets hub', 'hub:markets')]]);
  return { text: tx, kb };
}

// ----- Live scores -----
const LIVE_CACHE_MS = Number(process.env.LIVE_CACHE_MS || 30e3); // how long a live-score fetch is shared between users
async function getLive() {
  const f = await getGames();
  const today = localDate(new Date()), now = Date.now();
  const list = f.games.filter((g) => {
    if (g.status === 'LIVE' || g.status === 'HT') return g.hs != null || now - g.utc < 3 * 3600e3;
    if (g.status === 'FT') return localDate(new Date(g.utc)) === today || g.utc > now - 6 * 3600e3;
    return g.status === 'NS' && g.utc <= now && now - g.utc < 3 * 3600e3; // kicked off but no source has the score yet
  });
  return { at: f.at, list, ok: f.ok };
}
const liveKb = () => Markup.inlineKeyboard([[Markup.button.callback('🔄 Refresh', 'live')], [Markup.button.callback('⚽ Today', 'today'), Markup.button.callback('🏠 Menu', 'menu')]]);
async function renderLive() {
  const { at, list, ok } = await getLive();
  const stamp = new Intl.DateTimeFormat('en-GB', { timeZone: TZ, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }).format(new Date(at));
  const mark = (v) => (v === null ? '–' : v ? '✓' : '✗');
  const srcLine = `📡 ESPN ${mark(ok.espn)} · TheSportsDB ${mark(ok.sdb)} · football-data ${mark(ok.fd)}`;
  if (!list.length) return { text: `🔴 <b>Scores</b>\n\nNo live or finished matches right now in the covered leagues.\n\n🕒 Updated ${stamp}\n${srcLine}`, kb: liveKb() };
  const live = list.filter((g) => g.status === 'LIVE' || g.status === 'HT');
  const done = list.filter((g) => g.status === 'FT');
  const wait = list.filter((g) => g.status === 'NS');
  const line = (g) => {
    const hn = esc(g.home), an = esc(g.away);
    if (g.status === 'NS') return `⏳ ${hn} vs ${an} — kicked off ${fmtTime(g.utc)}, no score yet\n`;
    const sc = `<b>${g.hs == null ? 0 : g.hs}-${g.as == null ? 0 : g.as}</b>`;
    const tag = g.status === 'FT' ? (g.aet ? '✅ AET' : '✅ FT') : g.status === 'HT' ? '⏸ HT' : g.minute ? `🔴 ${esc(g.minute)}` : '🔴 LIVE';
    return `${tag} ${hn} ${sc} ${an}\n`;
  };
  const section = (title, arr, max) => {
    if (!arr.length) return '';
    let out = `\n${title}\n`, last = '';
    const sorted = arr.slice().sort((a, b) => String(a.league).localeCompare(String(b.league)) || a.utc - b.utc);
    for (const g of sorted.slice(0, max)) {
      if (g.league !== last) { out += `🏆 <b>${esc(g.league)}</b>\n`; last = g.league; }
      out += line(g);
    }
    if (arr.length > max) out += `…and ${arr.length - max} more\n`;
    return out;
  };
  let tx = `🔴 <b>Scores</b> — ${live.length} live · ${done.length} finished` + (wait.length ? ` · ${wait.length} awaiting score` : '') + '\n';
  tx += section('<b>Live now</b>', live, 40);
  tx += section('<b>Finished</b>', done.sort((a, b) => b.utc - a.utc).slice(0, 45), 45);
  tx += section('<b>Waiting for a score</b>', wait, 10);
  if (tx.length > 3550) tx = tx.slice(0, tx.lastIndexOf('\n', 3500)) + '\n…list shortened';
  tx += `\n🕒 Updated ${stamp} (${TZ}) · tap 🔄 to refresh\n${srcLine}`;
  return { text: tx, kb: liveKb() };
}
const liveH = async (ctx) => {
  const fromBtn = !!ctx.callbackQuery;
  try {
    if (!fromBtn) await ctx.sendChatAction('typing');
    const r = await renderLive();
    const msg = fromBtn && ctx.callbackQuery.message;
    if (msg && /Scores/.test(msg.text || '')) return await editOrReply(ctx, r.text, r.kb); // refresh in place instead of piling up messages
    return await reply(ctx, r.text, r.kb);
  } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
};

// ----- League table (built from the season data already downloaded: no extra API request) -----
async function renderTable(code) {
  if (!DOMESTIC.has(code)) return { text: 'Pick a league:', kb: tableKb() };
  const season = await getSeason(code);
  const tbl = buildTable(season);
  if (!tbl.rows.length) return { text: 'No table data yet for this league.', kb: tableKb() };
  const names = new Map();
  for (const m of season) { names.set(m.homeTeam.id, m.homeTeam.shortName || m.homeTeam.name); names.set(m.awayTeam.id, m.awayTeam.shortName || m.awayTeam.name); }
  let tx = `📋 <b>${esc(LEAGUE_NAMES[code] || code)}</b>\n<pre>`;
  tx += ' #  Team            P  GD  Pts\n';
  for (const r of tbl.rows.slice(0, 20)) {
    const nm = esc(String(names.get(r.id) || r.id).slice(0, 15)).padEnd(15);
    tx += `${String(r.rank).padStart(2)}  ${nm} ${String(r.p).padStart(2)} ${String(r.gd >= 0 ? '+' + r.gd : r.gd).padStart(3)}  ${String(r.pts).padStart(3)}\n`;
  }
  tx += '</pre>';
  return { text: tx, kb: tableKb() };
}
const tableKb = () => withNav(Markup.inlineKeyboard([...DOMESTIC].reduce((rows, c, i) => { if (i % 3 === 0) rows.push([]); rows[rows.length - 1].push(Markup.button.callback(LEAGUE_NAMES[c] || c, `tb:${c}`)); return rows; }, [])));

// ----- Head-to-head history -----
function getH2H(id) {
  return cached('h2h:' + id, 12 * 3600e3, async () => fdGet(`/matches/${id}/head2head?limit=8`));
}
async function renderH2H(m, u) {
  if (m.extra) return '🔁 No head-to-head data available for this match (it is outside the main data plan).';
  try {
    const d = await getH2H(m.id);
    const ms = (d.matches || []).filter((x) => x.score && x.score.fullTime && x.score.fullTime.home != null).slice(0, 8);
    if (!ms.length) throw new Error('empty');
    let tx = `🔁 <b>${esc(m.home)} vs ${esc(m.away)} — last ${ms.length} meetings</b>\n\n`;
    for (const x of ms) tx += `• ${fmtDate(x.utcDate)} ${esc(x.homeTeam.shortName || x.homeTeam.name)} <b>${x.score.fullTime.home}-${x.score.fullTime.away}</b> ${esc(x.awayTeam.shortName || x.awayTeam.name)}\n`;
    const a = d.aggregates;
    if (a && a.homeTeam && a.awayTeam) tx += `\n📊 ${esc(a.homeTeam.name)}: ${a.homeTeam.wins}W ${a.homeTeam.draws}D ${a.homeTeam.losses}L · ${esc(a.awayTeam.name)}: ${a.awayTeam.wins}W ${a.awayTeam.draws}D ${a.awayTeam.losses}L`;
    return tx;
  } catch (e) {
    const st = peek('stat:' + m.id);
    if (st && st.h2h && st.h2h.length) return `🔁 <b>H2H this season</b>\n${st.h2h.map(esc).join('\n')}`;
    return '🔁 No head-to-head data available for this match on the current data plan.';
  }
}

// ----- Account / referral / settings views -----
async function renderAccount(ctx) {
  const u = ctx.u;
  const used = await quotaUsed(u.id);
  const refs = (await q('SELECT count(*)::int AS n FROM users WHERE referred_by = $1', [u.id])).rows[0].n;
  let tx = '👤 <b>Your account</b>\n\n';
  tx += ctx.isAdmin ? '🛡 Plan: <b>Admin — all features unlocked</b>\n' : `⭐ Plan: <b>${ctx.premium ? t(u, 'plan_active', fmtUntil(u.plan_until)) : t(u, 'plan_free')}</b>\n`;
  if (!ctx.premium) tx += `🆓 Free analyses today: ${Math.min(used, FREE_DAILY_LIMIT)}/${FREE_DAILY_LIMIT}\n`;
  tx += `🔔 Daily picks: ${u.sub_daily ? 'ON' : 'off'}\n👥 Referrals: ${refs}\n`;
  if (u.excluded_until && new Date(u.excluded_until) > new Date()) tx += `⛔ Self-exclusion until ${fmtUntil(u.excluded_until)}\n`;
  tx += '\n🎟 /redeem a promo code · 🎁 /refer for free days · 📌 /myrecord · 🛡 /responsible';
  return tx;
}
function settingsView(u, premium) {
  const lg = userLeagues(u);
  const tx = '⚙️ <b>Settings</b>\n\n' +
    `🔔 Daily picks: <b>${u.sub_daily ? 'ON' : 'off'}</b> — ${String(u.push_hour).padStart(2, '0')}:00 (${esc(u.tz || TZ)})\n` +
    `🏆 Leagues: <b>${lg.length ? lg.map((c) => esc(LEAGUE_NAMES[c] || c)).join(', ') : 'all'}</b>\n` +
    `🎚 Min pick odds: <b>${Number(u.min_odds) > 0 ? Number(u.min_odds).toFixed(2) : 'none'}</b> (/minodds 1.5)\n` +
    `💵 Bankroll: <b>${u.bankroll ? money(u.bankroll) : 'not set'}</b> (/bankroll 50000) · Kelly fraction <b>${Math.round((Number(u.kelly) || 0.25) * 100)}%</b> (/kelly 0.25)\n` +
    `🌍 Language: <b>${{ en: 'English', pcm: 'Pidgin', fr: 'Français' }[u.lang] || 'English'}</b>\n` +
    (premium ? '' : `📬 Free daily tips: <b>${u.teaser_daily ? 'ON' : 'off'}</b> (${String(TEASER_HOUR).padStart(2, '0')}:00)\n`) +
    (premium ? '' : '\n<i>Daily push needs Premium.</i>');
  const kb = Markup.inlineKeyboard([
    [Markup.button.callback(u.sub_daily ? '🔕 Turn daily picks off' : '🔔 Turn daily picks on', 'set:sub')],
    [Markup.button.callback('🕘 −1h', 'set:h-'), Markup.button.callback('🕘 +1h', 'set:h+'), Markup.button.callback('🏆 Leagues', 'set:lg')],
    [Markup.button.callback('🌍 Language', 'set:lang'), Markup.button.callback('🛡 Responsible', 'resp')],
    ...(premium ? [] : [[Markup.button.callback(u.teaser_daily ? '📬 Free tips: ON ✅' : '📬 Free tips: off', 'tsr:toggle')]]),
    [MENU_BTN],
  ]);
  return { text: tx, kb };
}
function leaguesKb(u) {
  const sel = new Set(userLeagues(u));
  const rows = [];
  LEAGUES.forEach((c, i) => {
    if (i % 2 === 0) rows.push([]);
    rows[rows.length - 1].push(Markup.button.callback((sel.has(c) ? '✅ ' : '▫️ ') + (LEAGUE_NAMES[c] || c), `lg:${c}`));
  });
  rows.push([Markup.button.callback('🌐 All leagues', 'lg:all'), Markup.button.callback('✔️ Done', 'lg:done')]);
  return Markup.inlineKeyboard(rows);
}
const langKb = () => Markup.inlineKeyboard([[Markup.button.callback('🇬🇧 English', 'lang:en'), Markup.button.callback('🇳🇬 Pidgin', 'lang:pcm'), Markup.button.callback('🇫🇷 Français', 'lang:fr')]]);

// ----- Premium screen -----
function premiumView(u) {
  const keys = Object.keys(PLANS);
  const perDay = (k) => PLANS[k].ngn / PLANS[k].days;
  const save = (k) => Math.round((1 - perDay(k) / perDay('day')) * 100);
  const price = (k) => [PAYSTACK_KEY ? '₦' + money(PLANS[k].ngn) : '', STARS_ON ? PLANS[k].stars + ' ⭐' : ''].filter(Boolean).join(' or ');
  const icon = { day: '🎟', week: '📅', month: '🔥' };
  const proof = proofLine();
  const active = hasPremium(u) && !isAdminId(u.id);
  let tx = '💎 <b>Premium</b>\n━━━━━━━━━━━━━━\n' + (active ? `✅ Active until <b>${fmtUntil(u.plan_until)}</b> — renew any time to add days.\n\n` : '') +
    '🏆 Top 5 · 💎 Best 3 · 🥇 Best 1\n🔟⭐ Best 10 Pro · 🌟 Pick of the Week\n📈 Value · ⚽ Goals · 🛡 Double chance · 🎯 Correct score\n🎯 Accas · 🛡 Safe slip · 🏖 Weekend special\n🔔 Daily push + kickoff, result &amp; price-move alerts\n♾ Unlimited match analysis and /predict\n\n';
  if (proof) tx += proof + '\n<i>Past results don\'t guarantee future results.</i>\n\n';
  tx += keys.map((k) => `${icon[k] || '⭐'} <b>${PLANS[k].label}</b> (${PLANS[k].days} day${PLANS[k].days > 1 ? 's' : ''}) — ${price(k)}${k !== 'day' && save(k) > 0 ? ` · <i>save ${save(k)}%</i>` : ''}`).join('\n');
  tx += '\n\n🎁 Earn free days with /refer · 🎟 Have a promo code? /redeem';
  const rows = buyRows();
  rows.push([Markup.button.callback('🎟 Promo code', 'redeem'), MENU_BTN]);
  return { text: tx, kb: Markup.inlineKeyboard(rows) };
}

// ----- Watchlist -----
async function addWatch(u, m) {
  const n = (await q('SELECT count(*)::int AS n FROM watch WHERE user_id = $1 AND utc > $2', [u.id, Date.now() - 864e5])).rows[0].n;
  if (n >= MAX_WATCH) return 'limit';
  let lean = null;
  try { const p = topPick(m, bestKnown(m, await deepAnalysis(m)).final); if (p) lean = `${p.label}${p.odds ? ' @ ' + p.odds.toFixed(2) : ''}`; } catch {}
  const r = await q(`INSERT INTO watch (user_id, match_id, comp, home, away, utc, lean) VALUES ($1,$2,$3,$4,$5,$6,$7) ON CONFLICT DO NOTHING RETURNING match_id`, [u.id, m.id, m.comp, m.home, m.away, m.utc, lean]);
  return r.rows.length ? 'added' : 'exists';
}
async function renderWatchlist(u) {
  const { rows } = await q('SELECT * FROM watch WHERE user_id = $1 AND utc > $2 ORDER BY utc', [u.id, Date.now() - 6 * 3600e3]);
  if (!rows.length) return { text: '⭐ Your watchlist is empty. Open a match from /today and tap ⭐ Watch to get a kickoff alert and the result.', kb: navKb() };
  let tx = '⭐ <b>Watchlist</b>\n\n';
  const btns = [];
  rows.forEach((w, i) => {
    tx += `${i + 1}. <b>${esc(w.home)} vs ${esc(w.away)}</b> — ${fmtDate(w.utc)} ${fmtTime(w.utc)}${w.lean ? `\n   lean: ${esc(w.lean)}` : ''}\n`;
    btns.push(Markup.button.callback(`❌ ${i + 1}`, `uw:${w.match_id}`));
  });
  return { text: tx, kb: withNav(Markup.inlineKeyboard([btns])) };
}

// =====================================================================
//  FEATURE PACK — conversion, new pick types, engagement, safety, navigation
// =====================================================================
const isUpcoming = (x) => x.m.utc > Date.now();
const tierOf = (p) => (p >= 0.6 ? '🟢 High' : p >= 0.5 ? '🟡 Medium' : '🟠 Lower');
const dotOf = (p) => tierOf(p).split(' ')[0];
const pickName = (home, away, k) => (k === 'H' ? `${home} win` : k === 'A' ? `${away} win` : 'Draw');
const FOLLOW_HINT = '📌 <i>Tap a number below to follow a pick — I\'ll message you the result and track your record (/myrecord).</i>\n\n';
const LEGEND = '<i>🟢 High (60%+) · 🟡 Medium (50–60%) · 🟠 Lower (under 50%)</i>\n';

// ----- track record used as social proof (Premium picks only, last N days) -----
function proofLine(days = 30) {
  const since = Date.now() - days * 864e5;
  const rows = predLog.filter((r) => r.res && r.final && r.tags && (r.tags.includes('top5') || r.tags.includes('best3')) && r.utc > since);
  if (rows.length < 5) return '';
  const hit = rows.filter((r) => argmax(r.final) === r.res).length;
  const ro = roiOf(rows);
  return `📈 Premium picks, last ${days} days: <b>${hit}/${rows.length}</b> correct (${pct(hit / rows.length)}%)` + (ro.n >= 5 ? ` · ROI ${signed(ro.roi * 100)}%` : '');
}

// ----- keyboards: follow buttons, buy buttons -----
function followKb(shown, kb) {
  const base = (kb && kb.reply_markup && kb.reply_markup.inline_keyboard) || [];
  const btns = shown.map((x, i) => Markup.button.callback(`📌 ${i + 1}`, `fp:${x.m.id}`));
  const rows = [];
  for (let i = 0; i < btns.length; i += 5) rows.push(btns.slice(i, i + 5));
  return Markup.inlineKeyboard([...rows, ...base]);
}
function buyRows() {
  const rows = [];
  for (const k of Object.keys(PLANS)) {
    const p = PLANS[k], row = [];
    if (STARS_ON) row.push(Markup.button.callback(`⭐ ${p.label} · ${p.stars}`, `buy:xtr:${k}`));
    if (PAYSTACK_KEY) row.push(Markup.button.callback(`💳 ${p.label} · ₦${money(p.ngn)}`, `buy:ps:${k}`));
    if (row.length) rows.push(row);
  }
  return rows;
}
function renewKb() {
  const rows = buyRows();
  if (!rows.length) rows.push([Markup.button.callback('💎 See plans', 'up')]);
  return Markup.inlineKeyboard(rows);
}
function quickBuyKb() {
  const p = PLANS.day, row = [];
  if (STARS_ON) row.push(Markup.button.callback(`⭐ Day pass · ${p.stars}`, 'buy:xtr:day'));
  if (PAYSTACK_KEY) row.push(Markup.button.callback(`💳 Day pass · ₦${money(p.ngn)}`, 'buy:ps:day'));
  const rows = row.length ? [row] : [];
  rows.push([Markup.button.callback('💎 See all plans', 'up')]);
  return Markup.inlineKeyboard(rows);
}

// ----- locked-pick teaser: free users see WHICH matches are picked, not the pick itself -----
async function renderTeaser(u, kind) {
  const now = Date.now();
  let rows;
  if (kind === 'top5') rows = (await computeTop5()).filter((x) => x.m.utc > now);
  else {
    const b3 = await computeBest3();
    rows = b3.list.filter((x) => x.m.utc > now);
    if (kind === 'best1') rows = rows.slice(0, 1);
  }
  rows = applyPrefs(rows, u);
  const head = { top5: '🏆 <b>Top 5 picks</b>', best3: '💎 <b>Best 3 — under-the-radar</b>', best1: '🥇 <b>Best pick of the day</b>' }[kind];
  const proof = proofLine();
  if (!rows.length) return { text: `${head}\n━━━━━━━━━━━━━━\n🔒 <b>Premium feature.</b> Today's picks aren't ready yet — Premium members see them the moment they are.\n\n🆓 Free today: /best10` + (proof ? `\n\n${proof}` : ''), kb: quickBuyKb() };
  let tx = `${head}\n━━━━━━━━━━━━━━\n`;
  rows.forEach(({ m, sc }, i) => {
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b>\n   🕒 ${fmtTime(m.utc)} · ${esc(m.compName)}\n   👉 🔒 <i>Pick hidden</i> · ${tierOf(sc.pick.p)}\n`;
  });
  tx += `\n🔓 <b>Unlock ${rows.length === 1 ? 'this pick' : 'all ' + rows.length + ' picks'} with Premium</b> — from ₦${money(PLANS.day.ngn)} for a day pass.\n`;
  if (proof) tx += `\n${proof}\n<i>Past results don't guarantee future results.</i>\n`;
  tx += '\n🆓 Free today: /best10';
  return { text: tx, kb: quickBuyKb() };
}

// ----- Best 10 Pro (Premium): the 10 top-ranked picks with confidence tiers + model breakdown -----
async function renderBest10Pro(u) {
  const all = (await computeRanked()).filter(isUpcoming);
  const list = applyPrefs(all, u).slice(0, 10);
  if (!list.length) return { text: all.length ? t(u, 'none_prefs') : t(u, 'none_qual'), kb: null };
  let tx = '🔟⭐ <b>Best 10 Pro</b>\n<i>The 10 highest-ranked picks today, with confidence tiers and the model breakdown.</i>\n' + LEGEND + '\n' + FOLLOW_HINT;
  list.forEach(({ m, st, sc }, i) => {
    const p = sc.pick, k = p.k;
    tx += `${i + 1}. ${dotOf(p.p)} <b>${esc(m.home)} vs ${esc(m.away)}</b>\n   🕒 ${fmtTime(m.utc)} · ${esc(m.compName)}\n`;
    tx += `   👉 <b>${esc(p.label)}</b> (${p.sign}) @ ${p.odds.toFixed(2)}${bestPriceTxt(m, k)} · ${pct(p.p)}% · ${sc.agreeN}/${sc.total} sources agree\n`;
    tx += `   📊 Market ${pct(m.odds.probs[k])}%` + (st.form ? ` · Form ${pct(st.form[k])}%` : '') + (st.ai ? ` · AI ${pct(st.ai.probs[k])}%` : '') + '\n';
    if (st.mk && st.form) tx += `   ⚽ Expected ${st.form.lh.toFixed(1)}–${st.form.la.toFixed(1)} · likeliest ${st.mk.score} · O2.5 ${pct(st.mk.over25)}% · BTTS ${pct(st.mk.btts)}%\n`;
    const r = reasonOf(m, st);
    if (r) tx += `   <i>${esc(r)}</i>\n`;
    tx += stakeTxt(u, p.p, m.odds.best ? m.odds.best[k] : p.odds) + '\n';
  });
  return { text: tx + t(u, 'disc'), kb: followKb(list, null) };
}

// ----- Safe slip (Premium): safest combination of 1-3 picks that reaches the target combined odds -----
async function renderSafe(u) {
  const pool = byLeague(await accaPool(), u)
    .filter((x) => x.sc.pick.p >= 0.5 && x.sc.pick.odds >= MIN_PICK_ODDS)
    .sort((a, b) => b.sc.pick.p - a.sc.pick.p)
    .slice(0, 10);
  let best = null;
  const tryCombo = (legs) => {
    const odds = legs.reduce((a, x) => a * x.sc.pick.odds, 1);
    const prob = legs.reduce((a, x) => a * x.sc.pick.p, 1);
    if (odds < SAFE_TARGET_ODDS) return;
    if (!best || prob > best.prob + 1e-9 || (Math.abs(prob - best.prob) < 1e-9 && legs.length < best.legs.length)) best = { legs, odds, prob };
  };
  for (let i = 0; i < pool.length; i++) {
    tryCombo([pool[i]]);
    for (let j = i + 1; j < pool.length; j++) {
      tryCombo([pool[i], pool[j]]);
      for (let k = j + 1; k < pool.length; k++) tryCombo([pool[i], pool[j], pool[k]]);
    }
  }
  if (!best) return { text: `🛡 <b>Safe ${SAFE_TARGET_ODDS.toFixed(1)}-odds slip</b>\n\nNo safe combination reaches ${SAFE_TARGET_ODDS.toFixed(1)} odds today. Check back later, or try /acca safe.\n\n` + t(u, 'disc'), kb: null };
  let tx = `🛡 <b>Safe ${SAFE_TARGET_ODDS.toFixed(1)}-odds slip</b>\n<i>The most likely combination of up to 3 picks that still reaches ${SAFE_TARGET_ODDS.toFixed(1)} total odds.</i>\n\n` + FOLLOW_HINT;
  best.legs.forEach(({ m, sc }, i) => {
    const p = sc.pick;
    tx += `${i + 1}. ${dotOf(p.p)} <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)}\n   👉 <b>${esc(p.label)}</b> (${p.sign}) @ ${p.odds.toFixed(2)} · ${pct(p.p)}%\n`;
  });
  tx += `\n📦 Combined odds: <b>${best.odds.toFixed(2)}</b> · 🎲 Chance all land: <b>${(best.prob * 100).toFixed(1)}%</b> · EV ${signed((best.prob * best.odds - 1) * 100, 0)}%` + stakeTxt(u, best.prob, best.odds) + '\n\n';
  tx += '<i>Even the safest slip loses often enough — keep stakes small.</i>\n\n' + t(u, 'disc');
  return { text: tx, kb: followKb(best.legs, null) };
}

// ----- Correct-score suggestions (Premium) from the goal model -----
async function renderScores(u) {
  const rows = [];
  for (const { m, st } of byLeague(await analysedToday(), u)) {
    if (!st.form) continue;
    const all = [];
    scoreGrid(st.form.lh, st.form.la).forEach((row, i) => row.forEach((p, j) => all.push({ s: `${i}-${j}`, p })));
    all.sort((a, b) => b.p - a.p);
    rows.push({ m, st, top: all.slice(0, 3) });
  }
  rows.sort((a, b) => b.top[0].p - a.top[0].p);
  if (!rows.length) return '🎯 <b>Correct score</b>\n\nNot enough goal-model data today.\n\n' + t(u, 'disc');
  let tx = '🎯 <b>Correct-score suggestions</b>\n<i>From the Dixon-Coles goal model. Correct score is high-variance — even the likeliest score lands only ~10–15% of the time, so stake small.</i>\n\n';
  rows.slice(0, 8).forEach(({ m, st, top }, i) => {
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)} · ${esc(m.compName)}\n   🎯 ` + top.map((x, j) => `${j === 0 ? '<b>' + x.s + '</b>' : x.s} (${(x.p * 100).toFixed(0)}% · fair ${fair(x.p)})`).join(' · ') + `\n   ⚽ Expected goals ${st.form.lh.toFixed(1)}–${st.form.la.toFixed(1)}\n\n`;
  });
  return tx + t(u, 'disc');
}

// ----- Pick of the Week (Premium): best Top-5 pick of the last 7 days (+ best one still to play) -----
function renderPotw(u) {
  const since = Date.now() - 7 * 864e5, now = Date.now();
  const rows = predLog
    .filter((r) => r.final && r.tags && r.tags.includes('top5') && r.utc > since)
    .map((r) => { const k = r.pubPick || argmax(r.final); const o = (r.pubOdds || r.odds || {})[k]; return { r, k, o, p: r.final[k] }; })
    .filter((x) => x.o && x.o >= MIN_PICK_ODDS);
  if (!rows.length) return { text: '🌟 <b>Pick of the Week</b>\n\nNo qualifying Top 5 pick in the last 7 days yet. Check back after the next Top 5.', kb: null };
  const line = ({ r, k, o, p }) => {
    const icon = !r.res ? '⏳' : r.res === k ? '✅' : '❌';
    return `${icon} <b>${esc(r.home)} vs ${esc(r.away)}</b>\n   🕒 ${fmtDate(r.utc)} ${fmtTime(r.utc)} · ${esc(LEAGUE_NAMES[r.comp] || r.comp)}\n   👉 <b>${esc(pickName(r.home, r.away, k))}</b> (${sign1x2[k]}) @ ${o.toFixed(2)} · ${dotOf(p)} ${pct(p)}%\n`;
  };
  const best = rows.slice().sort((a, b) => b.p - a.p)[0];
  const upc = rows.filter((x) => !x.r.res && x.r.utc > now).sort((a, b) => b.p - a.p)[0];
  let tx = '🌟 <b>Pick of the Week</b>\n<i>The highest-confidence Top 5 pick published in the last 7 days.</i>\n\n' + line(best);
  if (upc && upc.r.id !== best.r.id) tx += '\n⏭ <b>Best pick still to play</b>\n' + line(upc);
  return { text: tx + '\n' + t(u, 'disc'), kb: null };
}

// ----- Weekend special (Premium): 4-leg accumulator on Fri / Sat / Sun -----
async function renderWeekend(u) {
  const wd = new Intl.DateTimeFormat('en-US', { timeZone: TZ, weekday: 'short' }).format(new Date());
  if (!['Fri', 'Sat', 'Sun'].includes(wd)) return { text: '🏖 <b>Weekend special</b>\n\nThe weekend accumulator opens on <b>Friday</b>: a 4-leg slip built from that day\'s strongest picks. Meanwhile, try /safe for today\'s safest slip.', kb: null };
  const r = await renderAcca(u, 'balanced', 4);
  return { text: r.text.replace(/🎯 <b>(\d)-leg accumulator — balanced<\/b>/, '🏖 <b>Weekend special — $1-leg accumulator</b>'), kb: null };
}

// ----- Engagement: personal record + leaderboard -----
async function renderMyRecord(u) {
  const s = (await q(`SELECT count(*) FILTER (WHERE won IS NOT NULL)::int AS n, count(*) FILTER (WHERE won)::int AS w,
    COALESCE(sum(profit), 0) AS profit, count(*) FILTER (WHERE won IS NULL)::int AS pending FROM followed WHERE user_id = $1`, [u.id])).rows[0] || { n: 0, w: 0, profit: 0, pending: 0 };
  if (!s.n && !s.pending) return { text: '📌 <b>My record</b>\n━━━━━━━━━━━━━━\nYou are not following any picks yet.\n\nOpen /best10 (or any pick list) and tap a 📌 number to follow a pick. I\'ll tell you the result and keep your win rate and flat-stake ROI here.', kb: Markup.inlineKeyboard([[Markup.button.callback('🔟 Open Best 10', 'best10')]]) };
  const rows = (await q('SELECT * FROM followed WHERE user_id = $1 ORDER BY utc DESC LIMIT 12', [u.id])).rows;
  let tx = '📌 <b>My record</b>\n━━━━━━━━━━━━━━\n';
  tx += s.n ? `✅ <b>${s.w}/${s.n}</b> correct (${pct(s.w / s.n)}%) · ROI <b>${signed((Number(s.profit) / s.n) * 100)}%</b> <i>(1 unit flat)</i>\n` : 'No settled picks yet.\n';
  if (s.pending) tx += `⏳ ${s.pending} pending\n`;
  tx += '\n';
  for (const f of rows) {
    const icon = f.won == null ? '⏳' : f.won ? '✅' : '❌';
    tx += `${icon} ${fmtDate(f.utc)} <b>${esc(f.home)} vs ${esc(f.away)}</b> — ${esc(f.label)}${f.odds ? ' @ ' + Number(f.odds).toFixed(2) : ''}\n`;
  }
  return { text: tx + '\n🏅 See how you rank: /leaders', kb: null };
}
const maskName = (n) => (n ? esc(String(n).slice(0, 1)) + '•••' : 'Player');
async function renderLeaders(u) {
  const rows = (await q(
    `SELECT f.user_id, u.first_name, count(*)::int AS n, count(*) FILTER (WHERE f.won)::int AS w, COALESCE(sum(f.profit), 0) AS profit
     FROM followed f JOIN users u ON u.id = f.user_id
     WHERE f.won IS NOT NULL AND f.utc > $1 GROUP BY f.user_id, u.first_name HAVING count(*) >= $2
     ORDER BY (count(*) FILTER (WHERE f.won))::float / count(*) DESC, count(*) DESC LIMIT 50`, [Date.now() - 30 * 864e5, LB_MIN_PICKS])).rows;
  let tx = `🏅 <b>Leaderboard — last 30 days</b>\n<i>Followed picks only · minimum ${LB_MIN_PICKS} settled picks · names are hidden for privacy.</i>\n\n`;
  if (!rows.length) return { text: tx + 'Nobody qualifies yet — follow picks with the 📌 buttons and be the first!', kb: null };
  const medal = ['🥇', '🥈', '🥉'];
  rows.slice(0, 10).forEach((r, i) => {
    tx += `${medal[i] || (i + 1) + '.'} ${r.user_id === u.id ? '<b>You</b>' : maskName(r.first_name)} — ${r.w}/${r.n} (${pct(r.w / r.n)}%) · ROI ${signed((Number(r.profit) / r.n) * 100, 0)}%\n`;
  });
  const me = rows.findIndex((r) => r.user_id === u.id);
  tx += me >= 10 ? `\n📍 Your rank: <b>#${me + 1}</b> of ${rows.length}` : me < 0 ? `\n📍 Settle ${LB_MIN_PICKS} followed picks to enter the board (/myrecord).` : '';
  return { text: tx, kb: null };
}

// ----- Promo codes -----
async function redeemCode(u, raw) {
  const code = String(raw || '').trim().toUpperCase();
  if (!/^[A-Z0-9_-]{3,24}$/.test(code)) return { ok: false, msg: '🎟 Send your code like this: <code>/redeem WELCOME7</code>' };
  const c = (await q('SELECT * FROM promo_codes WHERE code = $1', [code])).rows[0];
  if (!c || (c.expires_at && new Date(c.expires_at) < new Date())) return { ok: false, msg: '❌ That code is invalid or has expired.' };
  if ((await q('SELECT 1 FROM promo_redemptions WHERE code = $1 AND user_id = $2', [code, u.id])).rows.length) return { ok: false, msg: 'ℹ️ You already used this code.' };
  const upd = await q('UPDATE promo_codes SET uses = uses + 1 WHERE code = $1 AND uses < max_uses AND (expires_at IS NULL OR expires_at > now()) RETURNING days', [code]);
  if (!upd.rows.length) return { ok: false, msg: '❌ This code has been fully redeemed.' };
  const ins = await q('INSERT INTO promo_redemptions (code, user_id) VALUES ($1, $2) ON CONFLICT DO NOTHING RETURNING code', [code, u.id]);
  if (!ins.rows.length) { await q('UPDATE promo_codes SET uses = uses - 1 WHERE code = $1', [code]); return { ok: false, msg: 'ℹ️ You already used this code.' }; }
  const days = upd.rows[0].days;
  const nu = await grantDays(u.id, days);
  return { ok: true, u: nu, msg: `🎉 <b>Code accepted!</b> +${days} day${days > 1 ? 's' : ''} of Premium — active until <b>${fmtUntil(nu.plan_until)}</b>.` };
}

// ----- Responsible play + self-exclusion -----
const RG_TIPS = '• Only stake money you can afford to lose — never money for rent, bills or school fees.\n• Set a budget and a time limit before you start, and stick to it.\n• Never chase losses. Picks are estimates, not guarantees — any bet can lose.\n• Betting should be fun. If it stops being fun, take a break and talk to someone you trust.\n• 18+ only.';
function responsibleView(u) {
  const ex = u.excluded_until && new Date(u.excluded_until) > new Date();
  let tx = '🛡 <b>Responsible play</b>\n━━━━━━━━━━━━━━\n' + RG_TIPS + '\n\n';
  if (ex) return { text: tx + `⛔ Self-exclusion is active until <b>${fmtUntil(u.excluded_until)}</b>. Picks, tips and alerts are paused.`, kb: navKb() };
  tx += '⏸ <b>Need a break?</b> Choose a self-exclusion period. While it runs, picks, tips and alerts are switched off — and it cannot be undone early.';
  return { text: tx, kb: navKb([[Markup.button.callback('24 hours', 'se:1'), Markup.button.callback('7 days', 'se:7')], [Markup.button.callback('30 days', 'se:30'), Markup.button.callback('90 days', 'se:90')]]) };
}

// ----- Navigation: main menu + section hubs -----
function mainText(u) {
  const plan = isAdminId(u.id) ? '🛡 Admin' : hasPremium(u) ? `💎 Premium · until ${fmtUntil(u.plan_until)}` : '🆓 Free plan';
  const today = new Intl.DateTimeFormat('en-GB', { timeZone: TZ, weekday: 'long', day: '2-digit', month: 'short' }).format(new Date());
  return `⚽ <b>Football Predictions</b>\n━━━━━━━━━━━━━━\n👋 ${esc(u.first_name || 'there')} · ${plan}\n📅 ${today}\n\nChoose a section 👇`;
}
const mainKb = (u) => Markup.inlineKeyboard([
  [Markup.button.callback('🔥 Daily Picks', 'hub:picks'), Markup.button.callback('⚽ Matches Today', 'today')],
  [Markup.button.callback('📈 Markets', 'hub:markets'), Markup.button.callback('🎯 Accumulators', 'hub:acca')],
  [Markup.button.callback('🔴 Live Scores', 'live'), Markup.button.callback('📊 Tables', 'tbl')],
  [Markup.button.callback('👤 My Zone', 'hub:me'), Markup.button.callback('⚙️ Settings', 'st')],
  [Markup.button.callback(hasPremium(u) ? '💎 My Premium' : '💎 Go Premium', 'up'), Markup.button.callback('🛡 Responsible', 'resp'), Markup.button.callback('❓ Help', 'help')],
]);
const LBTN = (u, label, data, prem = true) => Markup.button.callback((prem && !hasPremium(u) ? '🔒 ' : '') + label, data);
function hubView(name, u) {
  const back = [Markup.button.callback('⬅️ Menu', 'menu')];
  const free = hasPremium(u) ? '' : '\n<i>🔒 = Premium. Tap one to preview it.</i>';
  if (name === 'picks') {
    const proof = proofLine();
    return {
      text: '🔥 <b>Daily Picks</b>\n━━━━━━━━━━━━━━\n🔟 <b>Best 10</b> — free ranked picks\n🏆 <b>Top 5</b> — the sharpest picks of the day\n💎 <b>Best 3</b> — under-the-radar picks\n🥇 <b>Best 1</b> — the single best pick\n🔟⭐ <b>Best 10 Pro</b> — top 10 with confidence tiers\n🌟 <b>Pick of the Week</b>' + (proof ? `\n\n${proof}` : '') + free,
      kb: Markup.inlineKeyboard([
        [LBTN(u, '🔟 Best 10 · Free', 'best10', false), LBTN(u, '🏆 Top 5', 'top5')],
        [LBTN(u, '💎 Best 3', 'best3'), LBTN(u, '🥇 Best 1', 'best1')],
        [LBTN(u, '🔟⭐ Best 10 Pro', 'best10pro'), LBTN(u, '🌟 Pick of the Week', 'potw')],
        back,
      ]),
    };
  }
  if (name === 'markets') {
    return {
      text: '📈 <b>Markets</b>\n━━━━━━━━━━━━━━\n📈 <b>Value bets</b> — model beats the best bookie price\n🏁 <b>Result</b> — 1X2, Double Chance, Draw No Bet\n⚽ <b>Goals</b> — GG/NG, Over/Under 0.5–3.5, team goals, multi-goals\n⏱ <b>HT/FT</b> — HT/FT, HT, 2HT, win either half, score both halves\n🧩 <b>Combos</b> — 1 &amp; GG, 1 &amp; OV 2.5, 1X &amp; UN 3.5…\n📐 <b>Handicaps</b> — 1 (-1), 2 (+1), Asian handicap\n🛡 <b>Double chance</b> — safer 1X / X2 / DNB picks\n🎯 <b>Correct score</b> — likeliest scorelines\n📋 <b>Any match</b> — every market for one game (also corners &amp; cards)' + free,
      kb: Markup.inlineKeyboard([
        [LBTN(u, '📈 Value bets', 'value'), LBTN(u, '🏁 Result', 'mk:res')],
        [LBTN(u, '⚽ Goals', 'mk:goals'), LBTN(u, '⏱ HT/FT', 'mk:htft')],
        [LBTN(u, '🧩 Combos', 'mk:combo'), LBTN(u, '📐 Handicaps', 'mk:hcp')],
        [LBTN(u, '🛡 Double chance', 'dc'), LBTN(u, '🎯 Correct score', 'scores')],
        [LBTN(u, '📋 Any match', 'mkhelp'), LBTN(u, '⚽ O/U 2.5 &amp; BTTS', 'goals')],
        back,
      ]),
    };
  }
  if (name === 'acca') {
    return {
      text: '🎯 <b>Accumulators</b>\n━━━━━━━━━━━━━━\n🎯 <b>Acca builder</b> — safe / balanced / risky, 2–6 legs\n🛡 <b>Safe slip</b> — safest combo reaching ' + SAFE_TARGET_ODDS.toFixed(1) + ' odds\n🏖 <b>Weekend special</b> — 4-leg slip on Fri–Sun' + free,
      kb: Markup.inlineKeyboard([[LBTN(u, '🎯 Acca builder', 'acca:balanced:3'), LBTN(u, '🛡 Safe slip', 'safe')], [LBTN(u, '🏖 Weekend special', 'weekend')], back]),
    };
  }
  return {
    text: '👤 <b>My Zone</b>\n━━━━━━━━━━━━━━\n📌 <b>My record</b> — your followed picks and win rate\n🏅 <b>Leaderboard</b> — top followers this month\n⭐ <b>Watchlist</b> — kickoff alerts and results\n🎁 <b>Refer &amp; earn</b> — free Premium days\n🎟 <b>Redeem code</b> — promo codes\n📊 <b>Track record</b> — how the model is doing',
    kb: Markup.inlineKeyboard([
      [Markup.button.callback('📌 My record', 'myrec'), Markup.button.callback('🏅 Leaderboard', 'leaders')],
      [LBTN(u, '⭐ Watchlist', 'wl'), Markup.button.callback('🎁 Refer & earn', 'refer')],
      [Markup.button.callback('🎟 Redeem code', 'redeem'), Markup.button.callback('👤 Account', 'acct')],
      [Markup.button.callback('📊 Track record', 'trk'), Markup.button.callback('⚙️ Settings', 'st')],
      back,
    ]),
  };
}

// ----- Background jobs added by the feature pack -----
// 1) free daily tips for non-premium users (opt-out any time)
let teaserBackoffUntil = 0;
async function renderTeaserPush(u) {
  const list = applyPrefs((await computeBest10()).filter(isUpcoming), u).slice(0, 3);
  if (!list.length) return { empty: true };
  let tx = '☀️ <b>Today\'s free tips</b>\n━━━━━━━━━━━━━━\n';
  list.forEach(({ m, sc }, i) => {
    const p = sc.pick;
    tx += `${i + 1}. <b>${esc(m.home)} vs ${esc(m.away)}</b> · ${fmtTime(m.utc)}\n   👉 <b>${esc(p.label)}</b> @ ${p.odds.toFixed(2)} · ${dotOf(p.p)} ${pct(p.p)}%\n`;
  });
  tx += '\n🔒 <i>Today\'s Top 5, Best 3 and Best 1 are ready for Premium members.</i>';
  const proof = proofLine();
  if (proof) tx += '\n' + proof;
  tx += '\n\n⚠️ <i>18+ · bet responsibly. Estimates only, not guaranteed.</i>';
  return { empty: false, text: tx, kb: Markup.inlineKeyboard([[Markup.button.callback('🔟 Full Best 10', 'best10'), Markup.button.callback('💎 Unlock Premium', 'up')], [Markup.button.callback('🔕 Stop daily tips', 'tsr:off')]]) };
}
async function teaserTick() {
  if (Date.now() < teaserBackoffUntil) return;
  const now = new Date();
  const { rows } = await q(
    `SELECT * FROM users WHERE teaser_daily AND NOT banned AND NOT blocked AND (plan_until IS NULL OR plan_until <= now())
     AND (excluded_until IS NULL OR excluded_until <= now()) AND NOT (id = ANY($1::bigint[]))`, [ADMIN_IDS]);
  const due = rows.filter((u) => { const h = hourTz(now, u.tz); return u.last_teaser !== localDateTz(now, u.tz) && h >= TEASER_HOUR && h < TEASER_HOUR + 4; });
  if (!due.length) return;
  if (!(await getMatchesToday()).length) { teaserBackoffUntil = Date.now() + 30 * 60e3; return; }
  for (const u of due) {
    const r = await renderTeaserPush(u);
    if (r.empty) { teaserBackoffUntil = Date.now() + 20 * 60e3; return; } // odds not out yet — retry later
    const claim = await q('UPDATE users SET last_teaser = $2 WHERE id = $1 AND last_teaser IS DISTINCT FROM $2 RETURNING id', [u.id, localDateTz(now, u.tz)]);
    if (!claim.rows.length) continue;
    await sendTo(u.id, r.text, { reply_markup: r.kb.reply_markup });
    await sleep(60);
  }
}

// 2) price-move alerts: a published pick's odds moved by STALE_DRIFT or more (sent to people who follow it + admins)
async function staleTick() {
  const now = Date.now();
  const cand = predLog.filter((r) => r.tags && r.tags.length && !r.res && !r.staleAlerted && r.pubOdds && r.pubPick && r.utc > now && r.utc < now + 8 * 3600e3);
  if (!cand.length) return;
  const matches = await getMatchesToday().catch(() => []);
  for (const r of cand) {
    const m = matches.find((x) => x.id === r.id);
    if (!m || !m.odds) continue;
    const pub = r.pubOdds[r.pubPick], cur = m.odds.odds[r.pubPick];
    if (!pub || !cur) continue;
    const drift = (cur - pub) / pub;
    if (Math.abs(drift) < STALE_DRIFT) continue;
    r.staleAlerted = true; predDirty.add(r.id); savePredLog();
    const txt = `⚠️ <b>Price move</b> — ${esc(r.home)} vs ${esc(r.away)}\nOur pick (<b>${esc(pickName(r.home, r.away, r.pubPick))}</b>) was <b>${pub.toFixed(2)}</b> when published, now <b>${cur.toFixed(2)}</b> (${signed(drift * 100, 0)}%).\n` +
      (drift < 0 ? 'The market has shortened — the value may be gone. Check your bookie before betting.' : 'The price has drifted out — the market is less sure (team news?). Re-check before betting.');
    const ids = new Set(ADMIN_IDS);
    (await q('SELECT user_id FROM followed WHERE match_id = $1 AND won IS NULL', [r.id])).rows.forEach((x) => ids.add(x.user_id));
    for (const id of ids) { await sendTo(id, txt); await sleep(40); }
  }
}

// 3) settle followed picks + personal result messages
async function followTick() {
  const { rows } = await q('SELECT * FROM followed WHERE won IS NULL AND utc < $1 AND utc > $2', [Date.now() - 2 * 3600e3, Date.now() - 5 * 864e5]);
  for (const f of rows) {
    const r = predById.get(f.match_id);
    if (!r || !r.res) continue;
    const won = f.pick === r.res;
    const profit = won ? (f.odds ? Number(f.odds) - 1 : 0) : -1;
    const upd = await q('UPDATE followed SET res = $3, won = $4, profit = $5 WHERE user_id = $1 AND match_id = $2 AND won IS NULL RETURNING user_id', [f.user_id, f.match_id, r.res, won, profit]);
    if (!upd.rows.length) continue;
    await sendTo(f.user_id, `${won ? '✅' : '❌'} <b>Your pick ${won ? 'won' : 'lost'}</b>\n${esc(f.home)} vs ${esc(f.away)}${r.score ? ' (' + esc(r.score) + ')' : ''}\nYour pick: ${esc(f.label)}${f.odds ? ' @ ' + Number(f.odds).toFixed(2) : ''}\n📌 Your record: /myrecord`);
    await sleep(40);
  }
}

// ---------- Background jobs ----------
let pushBackoffUntil = 0;
async function pushTick() {
  if (Date.now() < pushBackoffUntil) return;
  const now = new Date();
  const { rows } = await q(
    `SELECT * FROM users WHERE sub_daily AND NOT banned AND NOT blocked AND (excluded_until IS NULL OR excluded_until <= now()) AND (plan_until > now() OR id = ANY($1::bigint[]))`, [ADMIN_IDS]);
  const due = rows.filter((u) => {
    const h = hourTz(now, u.tz);
    return u.last_push !== localDateTz(now, u.tz) && h >= u.push_hour && h < u.push_hour + 4;
  });
  if (!due.length) return;
  const matches = await getMatchesToday();
  if (!matches.length) { pushBackoffUntil = Date.now() + 30 * 60e3; return; }
  for (const u of due) {
    const [a, b] = [await renderTop5(u, { full: true }), await renderBest3(u, { full: true })];
    if (a.empty && b.empty) { pushBackoffUntil = Date.now() + 20 * 60e3; return; } // odds not out yet — retry later
    const claim = await q(`UPDATE users SET last_push = $2 WHERE id = $1 AND last_push IS DISTINCT FROM $2 RETURNING id`, [u.id, localDateTz(now, u.tz)]);
    if (!claim.rows.length) continue;
    await sendTo(u.id, `${t(u, 'daily_head')}`);
    if (!a.empty) await sendTo(u.id, a.text, a.kb ? { reply_markup: a.kb.reply_markup } : {});
    if (!b.empty) await sendTo(u.id, b.text, b.kb ? { reply_markup: b.kb.reply_markup } : {});
    await sleep(60);
  }
}
async function kickoffTick() {
  const now = Date.now();
  const { rows } = await q(
    `SELECT w.*, u.lang FROM watch w JOIN users u ON u.id = w.user_id WHERE NOT w.ko_sent AND w.utc BETWEEN $1 AND $2`, [now, now + 30 * 60e3]);
  for (const w of rows) {
    await q('UPDATE watch SET ko_sent = true WHERE user_id = $1 AND match_id = $2', [w.user_id, w.match_id]);
    await sendTo(w.user_id, t({ lang: w.lang }, 'kickoff', w.home, w.away, Math.max(1, Math.round((w.utc - now) / 60e3)), w.lean));
    await sleep(40);
  }
}
async function expiryTick() {
  // 24h before the end: one-tap renew buttons
  const { rows } = await q(`SELECT * FROM users WHERE plan_until BETWEEN now() AND now() + interval '24 hours' AND NOT warned_exp AND NOT blocked`);
  for (const u of rows) {
    await q('UPDATE users SET warned_exp = true WHERE id = $1', [u.id]);
    await sendTo(u.id, t(u, 'exp_warn', fmtUntil(u.plan_until)) + '\n\n👇 Renew in one tap:', { reply_markup: renewKb().reply_markup });
    await sleep(40);
  }
  // win-back: sent once, within 2 days after Premium ended
  const gone = await q(`SELECT * FROM users WHERE plan_until BETWEEN now() - interval '2 days' AND now() AND NOT expired_notified AND NOT blocked AND (excluded_until IS NULL OR excluded_until <= now())`);
  for (const u of gone.rows) {
    await q('UPDATE users SET expired_notified = true WHERE id = $1', [u.id]);
    const proof = proofLine();
    await sendTo(u.id, '💤 <b>Your Premium has ended.</b>\nTomorrow\'s Top 5, Best 3 and Best 1 will be locked.' + (proof ? '\n\n' + proof : '') + '\n\n🎟 Come back with a <b>Day pass</b>, or renew for a week or a month:', { reply_markup: renewKb().reply_markup });
    await sleep(40);
  }
}
const sign1x2 = { H: '1', D: 'X', A: '2' };
async function settleTick() {
  const changed = await resolveLog();
  await followTick().catch((e) => console.log('followTick failed:', e.message));
  // 1) watched matches: individual result messages
  const { rows } = await q(`SELECT w.*, u.lang FROM watch w JOIN users u ON u.id = w.user_id WHERE NOT w.res_sent AND w.utc < $1`, [Date.now() - 2 * 3600e3]);
  for (const w of rows) {
    const r = predById.get(w.match_id);
    if (r && r.res) {
      const k = r.final ? argmax(r.final) : null;
      const txt = `🏁 <b>${esc(w.home)} vs ${esc(w.away)}</b> finished${r.score ? ' ' + r.score : ''} — result <b>${sign1x2[r.res]}</b>.` +
        (k ? `\nOur lean (${sign1x2[k]}): ${k === r.res ? '✅ correct' : '❌ missed'}` : '');
      await q('UPDATE watch SET res_sent = true WHERE user_id = $1 AND match_id = $2', [w.user_id, w.match_id]);
      await sendTo(w.user_id, txt);
      await sleep(40);
    } else if (w.utc < Date.now() - 3 * 864e5) {
      await q('UPDATE watch SET res_sent = true WHERE user_id = $1 AND match_id = $2', [w.user_id, w.match_id]);
    }
  }
  // 2) digest of newly settled Top 5 / Best 3 picks for daily-push subscribers
  const fresh = predLog.filter((r) => r.res && r.final && r.tags && r.tags.length && !r.notified);
  if (!fresh.length) return;
  fresh.forEach((r) => { r.notified = true; predDirty.add(r.id); });
  savePredLog();
  let won = 0, tx = '📊 <b>Results of our recent picks</b>\n\n';
  for (const r of fresh.sort((a, b) => a.utc - b.utc)) {
    const k = argmax(r.final);
    if (k === r.res) won++;
    tx += `${k === r.res ? '✅' : '❌'} ${esc(r.home)} vs ${esc(r.away)} — pick ${sign1x2[k]}, result ${sign1x2[r.res]}${r.score ? ' (' + r.score + ')' : ''}\n`;
  }
  tx += `\n<b>${won}/${fresh.length}</b> correct. Full record: /stats · /history`;
  const subs = (await q(`SELECT id FROM users WHERE sub_daily AND NOT banned AND NOT blocked AND (excluded_until IS NULL OR excluded_until <= now()) AND (plan_until > now() OR id = ANY($1::bigint[]))`, [ADMIN_IDS])).rows;
  for (const s of subs) { await sendTo(s.id, tx); await sleep(40); }
}
let tickN = 0, ticking = false;
async function schedulerTick() {
  if (ticking || !dbReady) return;
  ticking = true;
  try {
    tickN++;
    await kickoffTick();
    await pushTick();
    await teaserTick();
    if (tickN % 5 === 0) await recheckPendingPaystack();
    if (tickN % 10 === 0) await expiryTick();
    if (tickN % 15 === 0) await staleTick();
    if (tickN % 30 === 0) await settleTick();
  } catch (e) { console.log('Scheduler error:', e.message); }
  ticking = false;
}

// ---------- Bot ----------
const bot = new Telegraf(BOT_TOKEN, { handlerTimeout: 10 * 60 * 1000 });
const HTML = { parse_mode: 'HTML', link_preview_options: { is_disabled: true } };
const reply = (ctx, text, kb) => ctx.reply(text, { ...HTML, ...(kb || {}) });
const detach = (fn) => (ctx) => { fn(ctx).catch((e) => console.error(e)); }; // long jobs never block Telegraf
async function replyLong(ctx, text, kb) {
  const parts = splitMsg(text);
  for (let i = 0; i < parts.length; i++) await reply(ctx, parts[i], i === parts.length - 1 ? kb : null);
}
const editOrReply = async (ctx, text, kb) => {
  try { await ctx.editMessageText(text, { ...HTML, ...(kb || {}) }); }
  catch (e) { if (!/not modified/i.test(e.message)) await reply(ctx, text, kb); }
};

// ----- users + access -----
bot.use(async (ctx, next) => {
  if (!ctx.from || ctx.from.is_bot) return next();
  const lang = (ctx.from.language_code || '').toLowerCase().startsWith('fr') ? 'fr' : 'en';
  try {
    const { rows } = await q(
      `INSERT INTO users (id, username, first_name, tz, lang) VALUES ($1, $2, $3, $4, $5)
       ON CONFLICT (id) DO UPDATE SET username = EXCLUDED.username, first_name = EXCLUDED.first_name, last_seen = now(), blocked = false
       RETURNING *, (xmax = 0) AS inserted`,
      [ctx.from.id, ctx.from.username || null, ctx.from.first_name || null, TZ, lang]);
    ctx.u = rows[0];
    ctx.isNew = !!ctx.u.inserted;
  } catch (e) {
    console.log('User upsert failed:', e.message);
    ctx.u = { id: ctx.from.id, lang, tz: TZ, leagues: [], min_odds: 0, push_hour: 9, kelly: 0.25 };
  }
  ctx.isAdmin = isAdminId(ctx.from.id);
  if (ctx.u.banned && !ctx.isAdmin) return;
  if (ctx.u.excluded_until && new Date(ctx.u.excluded_until) > new Date() && !ctx.isAdmin) { // self-exclusion: only /responsible, /account, /help and payments stay reachable
    const txt = (ctx.message && ctx.message.text) || '';
    const cb = (ctx.callbackQuery && ctx.callbackQuery.data) || '';
    const ok = /^\/(responsible|selfexclude|account|help)\b/i.test(txt) || ['resp', 'acct', 'help'].includes(cb) || (ctx.message && ctx.message.successful_payment) || ctx.preCheckoutQuery;
    if (!ok) {
      const note = `🛡 Self-exclusion is active until ${fmtUntil(ctx.u.excluded_until)}. Picks, tips and alerts are paused. /responsible has support tips.`;
      if (ctx.callbackQuery) return ctx.answerCbQuery(note, { show_alert: true }).catch(() => {});
      if (ctx.inlineQuery) return ctx.answerInlineQuery([], { cache_time: 60, is_personal: true }).catch(() => {});
      if (ctx.message) return ctx.reply(note);
      return;
    }
  }
  ctx.premium = hasPremium(ctx.u);
  return next();
});
const paywall = (ctx, text) => {
  const proof = proofLine();
  return reply(ctx, (text || t(ctx.u, 'premium_only')) + (proof ? '\n\n' + proof : ''), quickBuyKb());
};
const needPremium = (fn) => (ctx) => (ctx.premium ? fn(ctx) : paywall(ctx));
const adminOnly = (fn) => (ctx) => (ctx.isAdmin ? fn(ctx) : undefined);

async function applyReferral(ctx, refId) {
  if (!Number.isFinite(refId) || refId === ctx.from.id) return;
  const ref = (await q('SELECT id FROM users WHERE id = $1 AND NOT banned', [refId])).rows[0];
  if (!ref) return;
  await q('UPDATE users SET referred_by = $2 WHERE id = $1 AND referred_by IS NULL', [ctx.from.id, refId]);
  const n = (await q('SELECT count(*)::int AS n FROM users WHERE referred_by = $1', [refId])).rows[0].n;
  if (REF_NEW_DAYS > 0) { ctx.u = (await grantDays(ctx.from.id, REF_NEW_DAYS)) || ctx.u; ctx.premium = hasPremium(ctx.u); }
  if (REF_BONUS_DAYS > 0 && n <= REF_MAX) {
    const ru = await grantDays(refId, REF_BONUS_DAYS);
    if (ru) await sendTo(refId, `🎁 A friend joined with your link — <b>+${REF_BONUS_DAYS} Premium days</b> added (until ${fmtUntil(ru.plan_until)}).`);
  }
}

// ----- basics -----
// ----- GUI helpers: Menu button on every screen, bottom keyboard -----
const MENU_BTN = Markup.button.callback('🏠 Menu', 'menu');
const withNav = (kb, rows = []) => Markup.inlineKeyboard([...((kb && kb.reply_markup && kb.reply_markup.inline_keyboard) || []), ...rows, [MENU_BTN]]);
const navKb = (rows = []) => withNav(null, rows);
const replyMenu = Markup.keyboard([
  ['🔥 Picks', '⚽ Today', '🔴 Live'],
  ['📈 Markets', '🎯 Accas', '👤 My Zone'],
  ['💎 Premium', '🛡 Responsible', '🏠 Menu'],
]).resize();
if (replyMenu.reply_markup) { replyMenu.reply_markup.is_persistent = true; replyMenu.reply_markup.input_field_placeholder = 'Pick an option…'; }
const showToday = async (ctx) => {
  await ctx.sendChatAction('typing').catch(() => {});
  try { const { text, kb } = await renderPage(0, ctx.u); await reply(ctx, text, kb); }
  catch (e) { console.error(e); await ctx.reply('⚠️ Could not load matches right now. Please try again in a minute.'); }
};
const MENU_ACTIONS = {
  '🔥 Picks': (ctx) => showHub('picks')(ctx),
  '📈 Markets': (ctx) => showHub('markets')(ctx),
  '🎯 Accas': (ctx) => showHub('acca')(ctx),
  '👤 My Zone': (ctx) => showHub('me')(ctx),
  '💎 Premium': (ctx) => showPremium(ctx),
  '🛡 Responsible': (ctx) => showResponsible(ctx),
  '🏠 Menu': (ctx) => reply(ctx, mainText(ctx.u), mainKb(ctx.u)),
  '⚽ Today': (ctx) => showToday(ctx),
  '🔟 Best 10': (ctx) => picksHandler('best10')(ctx),
  '🏆 Top 5': (ctx) => picksHandler('top5')(ctx),
  '💎 Best 3': (ctx) => picksHandler('best3')(ctx),
  '📈 Value': (ctx) => (ctx.premium ? valueH(ctx) : paywall(ctx)),
  '🎯 Acca': async (ctx) => { if (!ctx.premium) return paywall(ctx); const r = await renderAcca(ctx.u, 'balanced', 3); return reply(ctx, r.text, r.kb); },
  '🔴 Live': (ctx) => liveH(ctx),
  '📊 Tables': async (ctx) => { const r = await renderTable(''); return reply(ctx, r.text, r.kb); },
  '⭐ Watchlist': async (ctx) => { if (!ctx.premium) return paywall(ctx); const r = await renderWatchlist(ctx.u); return reply(ctx, r.text, r.kb); },
  '⚙️ Settings': (ctx) => { const v = settingsView(ctx.u, ctx.premium); return reply(ctx, v.text, v.kb); },
  '💳 Premium': (ctx) => showPremium(ctx),
  '👤 Account': async (ctx) => reply(ctx, await renderAccount(ctx), navKb()),
  '❓ Help': (ctx) => helpH(ctx),
};
bot.hears(Object.keys(MENU_ACTIONS), (ctx) => { Promise.resolve(MENU_ACTIONS[ctx.message.text](ctx)).catch((e) => console.error(e)); });
bot.command('menu', (ctx) => reply(ctx, mainText(ctx.u), mainKb(ctx.u)));
bot.action('menu', async (ctx) => { await ctx.answerCbQuery().catch(() => {}); await editOrReply(ctx, mainText(ctx.u), mainKb(ctx.u)); });
for (const [id, label] of [['tbl', '📊 Tables'], ['wl', '⭐ Watchlist'], ['st', '⚙️ Settings'], ['acct', '👤 Account'], ['help', '❓ Help']]) {
  bot.action(id, async (ctx) => { await ctx.answerCbQuery().catch(() => {}); Promise.resolve(MENU_ACTIONS[label](ctx)).catch((e) => console.error(e)); });
}
bot.start(async (ctx) => {
  const payload = ctx.startPayload || '';
  if (ctx.isNew && payload.startsWith('ref_')) await applyReferral(ctx, Number(payload.slice(4))).catch((e) => console.log('Referral failed:', e.message));
  if (payload.startsWith('promo_')) {
    const pr = await redeemCode(ctx.u, payload.slice(6)).catch(() => null);
    if (pr) { await reply(ctx, pr.msg); if (pr.ok && pr.u) { ctx.u = pr.u; ctx.premium = hasPremium(pr.u); } }
  }
  await reply(ctx, t(ctx.u, 'start', ctx.from.first_name) + '\n\n' + t(ctx.u, 'disc'), mainKb(ctx.u));
  await ctx.reply('👇 Your quick menu is now at the bottom of the chat.', replyMenu);
});

const helpH = (ctx) => {
  const tot = W_MKT + W_FORM + W_AI;
  return reply(ctx,
    '<b>How it works</b>\n' +
      `• Odds: European bookmakers (sharp books weighted higher), margin removed → market probabilities (${pct(W_MKT / tot)}%)\n` +
      `• Form: opponent-adjusted team ratings, recent games weighted more → goal model (${pct(W_FORM / tot)}%)\n` +
      `• AI: an online AI model reads the stats and gives its own estimate, kept close to the market (${pct(W_AI / tot)}%)\n\n` +
      'Top 5 = matches where the blended probability of one outcome is highest.\n' +
      'Best 3 = same blend, but only matches where neither team is a big-name club or in the top ' + TOP_RANK_EXCLUDE + ' of its table.\n' +
      'Best 10 (free) = the next-ranked picks after the Premium Top 5. Best 10 Pro = the 10 top-ranked picks with confidence tiers. Safe slip = the most likely combination reaching ' + SAFE_TARGET_ODDS.toFixed(1) + ' total odds.\n' +
      'Value = model probability beats the best bookmaker price by a margin. Acca = picks combined from today\'s best matches.\n\n' +
      CMD_LIST + '\n\n' + t(ctx.u, 'disc'), navKb());
};
bot.help(helpH);

bot.command('today', async (ctx) => {
  await ctx.sendChatAction('typing');
  try { const { text, kb } = await renderPage(0, ctx.u); await reply(ctx, text, kb); }
  catch (e) { console.error(e); await ctx.reply('⚠️ Could not load matches right now. Please try again in a minute.'); }
});
bot.action('today', async (ctx) => {
  await ctx.answerCbQuery();
  try { const { text, kb } = await renderPage(0, ctx.u); await reply(ctx, text, kb); } catch (e) { console.error(e); }
});
bot.action('noop', (ctx) => ctx.answerCbQuery().catch(() => {}));
bot.action(/^pd:(\d):(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery().catch(() => {});
  try { const { text, kb } = await renderPage(Number(ctx.match[2]), ctx.u, Number(ctx.match[1])); await editOrReply(ctx, text, kb); }
  catch (e) { console.error(e); await ctx.reply('⚠️ Could not load that day right now.'); }
});
const lastRefresh = new Map(); // protects football-data / Odds API quota from refresh spam
bot.action(/^rf:(\d):(\d+)$/, async (ctx) => {
  const day = Number(ctx.match[1]), k = 'd' + day;
  if (Date.now() - (lastRefresh.get(k) || 0) < 120e3) return ctx.answerCbQuery('✅ Already fresh — updated moments ago').catch(() => {});
  lastRefresh.set(k, Date.now());
  await ctx.answerCbQuery('Refreshing…').catch(() => {});
  cache.delete(day === 0 ? 'today:' + localDate(new Date()) : 'upcoming:' + localDate(new Date()));
  try { const { text, kb } = await renderPage(Number(ctx.match[2]), ctx.u, day); await editOrReply(ctx, text, kb); }
  catch (e) { console.error(e); await ctx.reply('⚠️ Refresh failed, try again in a minute.'); }
});
bot.action(/^pg:(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery();
  try { const { text, kb } = await renderPage(Number(ctx.match[1]), ctx.u); await editOrReply(ctx, text, kb); } catch (e) { console.error(e); }
});

// match analysis (free users: FREE_DAILY_LIMIT per day)
bot.action(/^m:(\d+):(\d+)$/, async (ctx) => {
  if (!(await useQuota(ctx))) {
    await ctx.answerCbQuery('🔒 Daily free limit reached', { show_alert: true });
    return paywall(ctx, t(ctx.u, 'limit_reached', FREE_DAILY_LIMIT));
  }
  await ctx.answerCbQuery('Analysing…');
  try {
    const m = await findMatchAny(Number(ctx.match[1]));
    if (!m) return editOrReply(ctx, 'Match not found (list refreshed). Use /today again.', null);
    const enc = Number(ctx.match[2]), dd = Math.floor(enc / 100), pn = enc % 100;
    const { text, kb } = await renderDetailFor(m, `pd:${dd}:${pn}`, ctx.u);
    await editOrReply(ctx, text, kb);
  } catch (e) { console.error(e); await ctx.reply('⚠️ Could not analyse that match right now.'); }
});

// Best 10 is free; Top 5 / Best 3 / Best 1 are Premium (free users get a teaser: the matches, not the picks)
const picksHandler = (kind) => async (ctx) => {
  const wait = await ctx.reply(t(ctx.u, kind === 'top5' || kind === 'best10' ? 'wait_top5' : 'wait_best3'));
  try {
    await ctx.sendChatAction('typing');
    const locked = kind !== 'best10' && !ctx.premium;
    const r = locked ? await renderTeaser(ctx.u, kind)
      : kind === 'best10' ? await renderBest10(ctx.u)
      : await (kind === 'top5' ? renderTop5 : renderBest3)(ctx.u, { full: true, only1: kind === 'best1' });
    await ctx.telegram.deleteMessage(ctx.chat.id, wait.message_id).catch(() => {});
    const SIB = { best10: ['🔟 Best 10', false], top5: ['🏆 Top 5', true], best3: ['💎 Best 3', true], best1: ['🥇 Best 1', true] };
    const others = Object.keys(SIB).filter((x) => x !== kind).map((x) => Markup.button.callback((SIB[x][1] && !ctx.premium ? '🔒 ' : '') + SIB[x][0], x));
    await replyLong(ctx, r.text, withNav(r.kb, [others, [Markup.button.callback('🔥 Picks hub', 'hub:picks'), Markup.button.callback('⚽ Today', 'today')]]));
  } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
};
bot.command('top5', detach(picksHandler('top5')));
bot.command('best3', detach(picksHandler('best3')));
bot.action('top5', async (ctx) => { await ctx.answerCbQuery(); picksHandler('top5')(ctx).catch(console.error); });
bot.action('best3', async (ctx) => { await ctx.answerCbQuery(); picksHandler('best3')(ctx).catch(console.error); });
bot.command('best10', detach(picksHandler('best10')));
bot.action('best10', async (ctx) => { await ctx.answerCbQuery(); picksHandler('best10')(ctx).catch(console.error); });
bot.command('best1', detach(picksHandler('best1')));
bot.action('best1', async (ctx) => { await ctx.answerCbQuery(); picksHandler('best1')(ctx).catch(console.error); });

// premium-only analysis tools
const toolHandler = (render, again) => async (ctx) => {
  const wait = await ctx.reply(t(ctx.u, 'wait_generic'));
  try {
    await ctx.sendChatAction('typing');
    const r = await render(ctx);
    await ctx.telegram.deleteMessage(ctx.chat.id, wait.message_id).catch(() => {});
    if (typeof r === 'string') await replyLong(ctx, r, navKb(again ? [[Markup.button.callback('🔄 Refresh', again)]] : [])); else await replyLong(ctx, r.text, r.kb ? withNav(r.kb) : navKb());
  } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
};
const valueH = toolHandler((ctx) => renderValue(ctx.u), 'value');
const goalsH = toolHandler((ctx) => renderGoals(ctx.u));
const dcH = toolHandler((ctx) => renderDC(ctx.u));
const accaFromText = (ctx) => {
  const a = argText(ctx).toLowerCase().split(/\s+/);
  const mode = ['safe', 'balanced', 'risky'].find((x) => a.includes(x)) || 'balanced';
  const n = Number(a.find((x) => /^[2-6]$/.test(x))) || (mode === 'risky' ? 4 : 3);
  return renderAcca(ctx.u, mode, n);
};
bot.command('value', needPremium(detach(valueH)));
bot.command('goals', needPremium(detach(goalsH)));
bot.command('dc', needPremium(detach(dcH)));
bot.command('acca', needPremium(detach(toolHandler(accaFromText))));
bot.action('value', async (ctx) => { await ctx.answerCbQuery(); if (!ctx.premium) return paywall(ctx); valueH(ctx).catch(console.error); });
bot.action(/^acca:(safe|balanced|risky):(\d)$/, async (ctx) => {
  await ctx.answerCbQuery();
  if (!ctx.premium) return paywall(ctx);
  try {
    const r = await renderAcca(ctx.u, ctx.match[1], Number(ctx.match[2]));
    if (ctx.callbackQuery.message && /accumulator/i.test(ctx.callbackQuery.message.text || '')) await editOrReply(ctx, r.text, r.kb);
    else await reply(ctx, r.text, r.kb);
  } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
});

// free tools
bot.command('live', detach(liveH));
bot.action('live', async (ctx) => { await ctx.answerCbQuery('Updating…').catch(() => {}); liveH(ctx).catch(console.error); });
bot.command('table', async (ctx) => {
  const code = argText(ctx).toUpperCase();
  try { const r = await renderTable(code); await reply(ctx, r.text, r.kb); } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
});
bot.action(/^tb:(\w+)$/, async (ctx) => {
  await ctx.answerCbQuery();
  try { const r = await renderTable(ctx.match[1]); await editOrReply(ctx, r.text, r.kb); } catch (e) { console.error(e); }
});
bot.action(/^h2h:(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery('Loading…');
  try {
    const m = await findMatchAny(Number(ctx.match[1]));
    if (m) await reply(ctx, await renderH2H(m, ctx.u));
  } catch (e) { console.error(e); }
});
bot.command('stats', async (ctx) => {
  await ctx.sendChatAction('typing');
  try { await replyLong(ctx, await renderStats()); } catch (e) { console.error(e); await ctx.reply('⚠️ Could not compute stats right now.'); }
});
bot.command('history', async (ctx) => { await reply(ctx, renderHistory()); });
bot.command('calibration', async (ctx) => {
  try { await reply(ctx, await renderCalibration()); } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
});

// predict any upcoming match (free users: counts toward the daily limit)
bot.command('predict', detach(async (ctx) => {
  const query = argText(ctx);
  if (!query) return void (await reply(ctx, 'Usage: <code>/predict Arsenal Chelsea</code> or <code>/predict Real Madrid</code> (next match of a team). Covers the next 7 days.'));
  if (!(await useQuota(ctx))) return void (await paywall(ctx, t(ctx.u, 'limit_reached', FREE_DAILY_LIMIT)));
  const wait = await ctx.reply(t(ctx.u, 'wait_generic'));
  try {
    const m = await findFixture(query);
    if (!m) { await ctx.telegram.deleteMessage(ctx.chat.id, wait.message_id).catch(() => {}); return void (await reply(ctx, `🔎 No fixture found for “${esc(query)}” in the next 7 days. Try the team name as it appears in /today.`)); }
    const r = await renderDetailFor(m, null, ctx.u);
    await ctx.telegram.deleteMessage(ctx.chat.id, wait.message_id).catch(() => {});
    await reply(ctx, r.text, r.kb);
  } catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
}));

// ----- stake calculator -----
bot.command('stake', async (ctx) => {
  const nums = argText(ctx).split(/\s+/).map(Number).filter((x) => Number.isFinite(x) && x > 0);
  if (nums.length < 2) return reply(ctx, 'Usage: <code>/stake 2.10 55</code> → odds 2.10, you think it wins 55%.\nSet your bankroll with <code>/bankroll 50000</code> to get a stake amount.');
  const o = nums[0];
  const p = nums[1] > 1 ? nums[1] / 100 : nums[1];
  if (o <= 1 || p <= 0 || p >= 1) return reply(ctx, 'Odds must be above 1.00 and probability between 1 and 99%.');
  const ev = p * o - 1;
  const fr = Number(ctx.u.kelly) || 0.25;
  const k = kellyStake(p, o, ctx.u.bankroll ? Number(ctx.u.bankroll) : null, fr);
  let tx = `🧮 <b>Stake calculator</b>\nOdds <b>${o.toFixed(2)}</b> · your probability <b>${(p * 100).toFixed(1)}%</b> (break-even ${(100 / o).toFixed(1)}%)\nEV per 1 staked: <b>${signed(ev * 100, 1)}%</b>\n\n`;
  if (k.f <= 0) tx += '❌ No edge at this price — Kelly says <b>don\'t bet</b>.';
  else {
    tx += `Full Kelly: ${(k.f * 100).toFixed(1)}% of bankroll\nSuggested (${Math.round(fr * 100)}% Kelly, capped at ${KELLY_CAP * 100}%): <b>${(k.adj * 100).toFixed(2)}%</b>`;
    tx += k.stake ? ` = <b>${money(k.stake)}</b>` : '\n<i>Set /bankroll to see the amount.</i>';
  }
  return reply(ctx, tx + '\n\n<i>Kelly assumes your probability is right. If it\'s overconfident, the stake is too big — that\'s why a fraction is used.</i>');
});
bot.command('bankroll', async (ctx) => {
  const n = Number(argText(ctx).replace(/[, ]/g, ''));
  if (!Number.isFinite(n) || n <= 0) return reply(ctx, 'Usage: <code>/bankroll 50000</code> — money you can afford to lose.');
  ctx.u = await updUser(ctx.u.id, { bankroll: n });
  return reply(ctx, `✅ Bankroll set to <b>${money(n)}</b>. Stake ideas now appear on picks. Change the Kelly fraction with /kelly.`);
});
bot.command('kelly', async (ctx) => {
  const n = Number(argText(ctx));
  if (!Number.isFinite(n) || n < 0.05 || n > 1) return reply(ctx, 'Usage: <code>/kelly 0.25</code> — fraction of full Kelly (0.05–1). Lower = safer.');
  ctx.u = await updUser(ctx.u.id, { kelly: n });
  return reply(ctx, `✅ Kelly fraction set to <b>${Math.round(n * 100)}%</b>.`);
});

// ----- subscriptions, settings, watchlist -----
bot.command('subscribe', needPremium(async (ctx) => {
  ctx.u = await updUser(ctx.u.id, { sub_daily: true });
  await reply(ctx, t(ctx.u, 'sub_on', ctx.u.push_hour, ctx.u.tz || TZ));
}));
bot.command('unsubscribe', async (ctx) => { ctx.u = await updUser(ctx.u.id, { sub_daily: false }); await reply(ctx, t(ctx.u, 'sub_off')); });
bot.command('pushtime', async (ctx) => {
  const h = Number(argText(ctx));
  if (!Number.isInteger(h) || h < 0 || h > 23) return reply(ctx, 'Usage: <code>/pushtime 8</code> — hour of day (0–23) in your timezone.');
  ctx.u = await updUser(ctx.u.id, { push_hour: h });
  return reply(ctx, `✅ Daily picks will arrive from ${String(h).padStart(2, '0')}:00 (${esc(ctx.u.tz || TZ)}).`);
});
bot.command('timezone', async (ctx) => {
  const tz = argText(ctx);
  if (!tz || !validTz(tz)) return reply(ctx, 'Usage: <code>/timezone Africa/Lagos</code> — any IANA timezone name, e.g. Europe/London, America/New_York.');
  ctx.u = await updUser(ctx.u.id, { tz });
  return reply(ctx, `✅ Timezone set to <b>${esc(tz)}</b>.`);
});
bot.command('minodds', async (ctx) => {
  const n = Number(argText(ctx));
  if (!Number.isFinite(n) || n < 0 || n > 10) return reply(ctx, 'Usage: <code>/minodds 1.5</code> (0 = no minimum). Picks below this price are hidden.');
  ctx.u = await updUser(ctx.u.id, { min_odds: n });
  return reply(ctx, n > 0 ? `✅ Picks priced under <b>${n.toFixed(2)}</b> are now hidden.` : '✅ Minimum odds filter removed.');
});
bot.command('leagues', (ctx) => reply(ctx, '🏆 <b>Choose your leagues</b>\nTap to toggle. Picks, /today, /value and daily push follow this filter. None selected = all.', leaguesKb(ctx.u)));
bot.command('lang', (ctx) => reply(ctx, '🌍 Choose your language:', langKb()));
bot.command('settings', (ctx) => { const v = settingsView(ctx.u, ctx.premium); return reply(ctx, v.text, v.kb); });
bot.action('set:sub', async (ctx) => {
  await ctx.answerCbQuery();
  if (!ctx.u.sub_daily && !ctx.premium) return paywall(ctx);
  ctx.u = await updUser(ctx.u.id, { sub_daily: !ctx.u.sub_daily });
  const v = settingsView(ctx.u, ctx.premium); await editOrReply(ctx, v.text, v.kb);
});
bot.action(/^set:h([+-])$/, async (ctx) => {
  await ctx.answerCbQuery();
  const h = (((ctx.u.push_hour || 9) + (ctx.match[1] === '+' ? 1 : -1)) % 24 + 24) % 24;
  ctx.u = await updUser(ctx.u.id, { push_hour: h });
  const v = settingsView(ctx.u, ctx.premium); await editOrReply(ctx, v.text, v.kb);
});
bot.action('set:lg', async (ctx) => { await ctx.answerCbQuery(); await editOrReply(ctx, '🏆 <b>Choose your leagues</b>\nTap to toggle. None selected = all.', leaguesKb(ctx.u)); });
bot.action('set:lang', async (ctx) => { await ctx.answerCbQuery(); await editOrReply(ctx, '🌍 Choose your language:', langKb()); });
bot.action(/^lg:(\w+)$/, async (ctx) => {
  const c = ctx.match[1];
  await ctx.answerCbQuery();
  if (c === 'done') { const v = settingsView(ctx.u, ctx.premium); return editOrReply(ctx, v.text, v.kb); }
  let sel = new Set(userLeagues(ctx.u));
  if (c === 'all') sel = new Set();
  else if (LEAGUES.includes(c)) { if (sel.has(c)) sel.delete(c); else sel.add(c); }
  ctx.u = await updUser(ctx.u.id, { leagues: [...sel] });
  await editOrReply(ctx, '🏆 <b>Choose your leagues</b>\nTap to toggle. None selected = all.', leaguesKb(ctx.u));
});
bot.action(/^lang:(en|pcm|fr)$/, async (ctx) => {
  await ctx.answerCbQuery();
  ctx.u = await updUser(ctx.u.id, { lang: ctx.match[1] });
  await editOrReply(ctx, t(ctx.u, 'lang_set'), null);
});

bot.action(/^w:(\d+)$/, async (ctx) => {
  if (!ctx.premium) { await ctx.answerCbQuery('🔒 Premium feature', { show_alert: true }); return paywall(ctx); }
  const m = await findMatchAny(Number(ctx.match[1]));
  if (!m) return ctx.answerCbQuery('Match not found', { show_alert: true });
  const r = await addWatch(ctx.u, m);
  return ctx.answerCbQuery(r === 'added' ? '⭐ Watching — you\'ll get a kickoff alert and the result' : r === 'exists' ? 'Already on your watchlist' : `Watchlist is full (${MAX_WATCH})`, { show_alert: r !== 'added' });
});
bot.command('watchlist', needPremium(async (ctx) => { const r = await renderWatchlist(ctx.u); await reply(ctx, r.text, r.kb); }));
bot.action(/^uw:(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery('Removed');
  await q('DELETE FROM watch WHERE user_id = $1 AND match_id = $2', [ctx.u.id, Number(ctx.match[1])]);
  const r = await renderWatchlist(ctx.u); await editOrReply(ctx, r.text, r.kb);
});

// ----- account, referral, premium + payments -----
bot.command('account', async (ctx) => reply(ctx, await renderAccount(ctx)));
const showPremium = (ctx) => { const v = premiumView(ctx.u); return reply(ctx, v.text, v.kb); };
bot.command('premium', showPremium);
bot.command('upgrade', showPremium);
bot.action('up', async (ctx) => { await ctx.answerCbQuery(); return showPremium(ctx); });

bot.action(/^buy:xtr:(day|week|month)$/, async (ctx) => {
  await ctx.answerCbQuery();
  try { await sendStarsInvoice(ctx, ctx.match[1]); } catch (e) { console.error(e); await ctx.reply('⚠️ Could not open the Stars invoice. Please try again.'); }
});
bot.action(/^buy:ps:(day|week|month)$/, async (ctx) => {
  await ctx.answerCbQuery();
  if (!PAYSTACK_KEY) return;
  try {
    const { url, ref } = await paystackInit(ctx.u, ctx.match[1]);
    await reply(ctx, `💳 <b>Pay ₦${money(PLANS[ctx.match[1]].ngn)} — Premium ${PLANS[ctx.match[1]].label}</b>\nOpen the secure Paystack page, pay, then come back. Premium activates automatically (or tap “I have paid”).`,
      Markup.inlineKeyboard([[Markup.button.url('💳 Pay with Paystack', url)], [Markup.button.callback('✅ I have paid', `pv:${ref}`)]]));
  } catch (e) { console.error(e); await ctx.reply('⚠️ Could not start the Paystack checkout. Please try again shortly.'); }
});
bot.action(/^pv:([\w-]+)$/, async (ctx) => {
  const ref = ctx.match[1];
  const own = (await q('SELECT 1 FROM payments WHERE ref = $1 AND user_id = $2', [ref, ctx.from.id])).rows.length;
  if (!own) return ctx.answerCbQuery('Payment not found', { show_alert: true });
  try {
    const r = await settlePaystack(ref);
    if (r.ok) return ctx.answerCbQuery(r.already ? '✅ Already active' : '✅ Payment confirmed!', { show_alert: true });
    return ctx.answerCbQuery(r.reason === 'pending' ? 'Not confirmed yet — finish the payment, then tap again.' : 'Payment amount mismatch — contact support.', { show_alert: true });
  } catch (e) { console.error(e); return ctx.answerCbQuery('Could not verify right now. Try again in a minute.', { show_alert: true }); }
});
bot.on('pre_checkout_query', async (ctx) => {
  const planKey = String((ctx.preCheckoutQuery || {}).invoice_payload || '').replace(/^plan:/, '');
  if (!PLANS[planKey]) return ctx.answerPreCheckoutQuery(false, 'This plan is no longer available.');
  return ctx.answerPreCheckoutQuery(true);
});
bot.on('successful_payment', async (ctx) => { try { await onStarsPaid(ctx); } catch (e) { console.error('Stars payment handling failed:', e); notifyAdmins(`⚠️ Stars payment from <code>${ctx.from.id}</code> could not be activated automatically: ${esc(e.message)}`); } });

// ----- inline mode: type @yourbot arsenal in any chat -----
bot.on('inline_query', async (ctx) => {
  const qs = (ctx.inlineQuery.query || '').trim();
  const list = peek('today:' + localDate(new Date()));
  if (!list || !list.length) {
    return ctx.answerInlineQuery([], { cache_time: 5, is_personal: true, button: { text: 'Open the bot for today\'s picks', start_parameter: 'inline' } });
  }
  const matches = list.filter((m) => m.utc > Date.now() - 2 * 3600e3 && (!qs || teamHit(qs, m) >= 0.99 || nameScore(qs, m.compName) >= 0.99));
  const results = matches.slice(0, 20).map((m) => {
    let tx = `⚽ <b>${esc(m.home)} vs ${esc(m.away)}</b>\n🏆 ${esc(m.compName)} · 🕒 ${fmtTime(m.utc)}\n`;
    if (m.odds) tx += `1 <b>${m.odds.odds.H.toFixed(2)}</b> · X <b>${m.odds.odds.D.toFixed(2)}</b> · 2 <b>${m.odds.odds.A.toFixed(2)}</b>\n📊 Market: ${pct(m.odds.probs.H)}% / ${pct(m.odds.probs.D)}% / ${pct(m.odds.probs.A)}%\n`;
    const d = ctx.premium ? peek('deep:' + m.id) : null;
    const pick = d ? topPick(m, bestKnown(m, d).final) : null;
    if (pick) tx += `👉 Lean: <b>${esc(pick.label)}</b> (${pct(pick.p)}%)\n`;
    return {
      type: 'article', id: String(m.id), title: `${m.home} vs ${m.away}`,
      description: `${fmtTime(m.utc)} · ${m.compName}` + (m.odds ? ` · 1 ${m.odds.odds.H.toFixed(2)} X ${m.odds.odds.D.toFixed(2)} 2 ${m.odds.odds.A.toFixed(2)}` : ''),
      input_message_content: { message_text: tx + '\n' + DISCLAIMER_EN(), parse_mode: 'HTML', link_preview_options: { is_disabled: true } },
    };
  });
  return ctx.answerInlineQuery(results, { cache_time: 60, is_personal: true });
});

// ----- admin tools (admins have every feature, no limits) -----
const ADMIN_HELP =
  '🛡 <b>Admin</b>\n/users – user &amp; revenue stats\n/payments – last payments\n/credits – Odds API credits\n/grant &lt;id&gt; &lt;days&gt; · /revoke &lt;id&gt;\n/ban &lt;id&gt; · /unban &lt;id&gt;\n' +
  '/broadcast &lt;text&gt; – everyone · /broadcastp &lt;text&gt; – premium only\n/forceupdate – clear caches\n/resolve – settle pending results now\n/feedstatus – score-feed health\n/userinfo &lt;id&gt;\n/dashboard – revenue, conversion &amp; churn\n/promo – create / list / delete promo codes\n/unexclude &lt;id&gt; – lift a self-exclusion';
bot.command('admin', adminOnly((ctx) => reply(ctx, ADMIN_HELP)));
bot.command('resolve', adminOnly(detach(async (ctx) => {
  const now = Date.now();
  const openNow = () => predLog.filter((r) => !r.res && r.utc < now - 3 * 3600e3 && r.utc > now - 5 * 864e5);
  const before = openNow().length;
  cache.forEach((v, k) => { if (k.startsWith('feedres:') || k === 'feed:games') cache.delete(k); });
  const changed = await resolveLog();
  const left = openNow();
  let tx = `✅ Settled <b>${changed.length}</b> of ${before} open matches.`;
  if (left.length) tx += `\n⏳ Still pending (${left.length}):\n` + left.slice(0, 15).map((r) => `• ${esc(r.home)} vs ${esc(r.away)} (${esc(r.comp)}, ${fmtDate(r.utc)})`).join('\n');
  await reply(ctx, tx);
})));
bot.command('feedstatus', adminOnly(detach(async (ctx) => {
  cache.delete('feed:games');
  try { await getGames(); } catch (e) { /* diag below shows why */ }
  await reply(ctx, `📡 <b>Score feed</b> (updated ${fmtTime(feedDiag.at || Date.now())})\nESPN: ${esc(feedDiag.espn)}\nTheSportsDB: ${esc(feedDiag.sdb)}\nfootball-data: ${esc(feedDiag.fd)}`);
})));
bot.command('users', adminOnly(async (ctx) => {
  const s = (await q(`SELECT count(*)::int total, count(*) FILTER (WHERE plan_until > now())::int premium, count(*) FILTER (WHERE sub_daily)::int subs,
    count(*) FILTER (WHERE joined_at > now() - interval '1 day')::int new24, count(*) FILTER (WHERE last_seen > now() - interval '1 day')::int active24,
    count(*) FILTER (WHERE blocked)::int blocked, count(*) FILTER (WHERE banned)::int banned FROM users`)).rows[0];
  const rev = (await q(`SELECT currency, sum(amount) AS total, count(*)::int AS n FROM payments WHERE status = 'success' GROUP BY currency`)).rows;
  const r30 = (await q(`SELECT currency, sum(amount) AS total FROM payments WHERE status = 'success' AND created_at > now() - interval '30 days' GROUP BY currency`)).rows;
  const fmt = (rows, key = 'total') => rows.map((r) => (r.currency === 'XTR' ? `${money(r[key])} ⭐` : `₦${money(r[key])}`)).join(' + ') || '0';
  await reply(ctx, `📈 <b>Users</b>\nTotal ${s.total} · Premium ${s.premium} · Daily push ${s.subs}\nNew 24h ${s.new24} · Active 24h ${s.active24}\nBlocked bot ${s.blocked} · Banned ${s.banned}\n\n💰 <b>Revenue</b>\nAll time: ${fmt(rev)} (${rev.reduce((a, r) => a + r.n, 0)} payments)\nLast 30 days: ${fmt(r30)}`);
}));
bot.command('payments', adminOnly(async (ctx) => {
  const rows = (await q(`SELECT * FROM payments ORDER BY id DESC LIMIT 12`)).rows;
  if (!rows.length) return reply(ctx, 'No payments yet.');
  await reply(ctx, '💳 <b>Last payments</b>\n' + rows.map((p) => `${p.status === 'success' ? '✅' : '⏳'} <code>${p.user_id}</code> ${p.provider} ${p.plan} ${p.currency === 'XTR' ? p.amount + '⭐' : '₦' + money(p.amount)} · ${fmtDate(p.created_at)}`).join('\n'));
}));
bot.command('fixtures', adminOnly(async (ctx) => {
  cache.delete('today:' + localDate(new Date()));
  const all = await getMatchesToday().catch((e) => { extraDiag.firstErr = extraDiag.firstErr || e.message; return []; });
  const ex = all.filter((m) => m.extra).length;
  return reply(ctx, `🔧 <b>Fixture sources</b> (${localDate(new Date())})\nfootball-data: <b>${all.length - ex}</b>\nESPN: <b>${extraDiag.espnCount}</b> (${extraDiag.espnOk}/${extraDiag.espnLeagues} leagues answered, ${extraDiag.espnMissing} unknown slugs${extraDiag.firstErr ? ' · first error: ' + esc(extraDiag.firstErr) : ''})\nTheSportsDB: <b>${extraDiag.sdbCount}</b>${extraDiag.sdbErr ? ' · error: ' + esc(extraDiag.sdbErr) : ''}\nOdds API events: <b>${extraDiag.oxCount}</b>${extraDiag.oxErr ? ' · error: ' + esc(extraDiag.oxErr) : ''}\nShown in /today: <b>${all.length}</b> (${extraDiag.withOdds} extras with odds)`);
}));
bot.command('credits', adminOnly((ctx) => reply(ctx, oddsCredits.remaining == null
  ? '🎟 No Odds API call has been made since the bot started.'
  : `🎟 The Odds API: <b>${oddsCredits.remaining}</b> credits left${oddsCredits.used != null ? ` (${oddsCredits.used} used)` : ''} — as of ${fmtTime(oddsCredits.at)}.`)));
bot.command('grant', adminOnly(async (ctx) => {
  const [id, days] = argText(ctx).split(/\s+/).map(Number);
  if (!id || !days) return reply(ctx, 'Usage: /grant &lt;userId&gt; &lt;days&gt;');
  const u = await grantDays(id, days);
  if (!u) return reply(ctx, 'User not found (they must have started the bot).');
  await sendTo(id, `🎁 You received <b>${days} days</b> of Premium (until ${fmtUntil(u.plan_until)}).`);
  return reply(ctx, `✅ ${days} days granted to <code>${id}</code>.`);
}));
bot.command('revoke', adminOnly(async (ctx) => {
  const id = Number(argText(ctx));
  if (!id) return reply(ctx, 'Usage: /revoke &lt;userId&gt;');
  await q('UPDATE users SET plan_until = NULL, sub_daily = false WHERE id = $1', [id]);
  return reply(ctx, `✅ Premium revoked for <code>${id}</code>.`);
}));
bot.command('ban', adminOnly(async (ctx) => { const id = Number(argText(ctx)); if (!id) return reply(ctx, 'Usage: /ban &lt;userId&gt;'); await q('UPDATE users SET banned = true, sub_daily = false WHERE id = $1', [id]); return reply(ctx, `🚫 Banned <code>${id}</code>.`); }));
bot.command('unban', adminOnly(async (ctx) => { const id = Number(argText(ctx)); if (!id) return reply(ctx, 'Usage: /unban &lt;userId&gt;'); await q('UPDATE users SET banned = false WHERE id = $1', [id]); return reply(ctx, `✅ Unbanned <code>${id}</code>.`); }));
bot.command('userinfo', adminOnly(async (ctx) => {
  const id = Number(argText(ctx));
  const u = id ? (await q('SELECT * FROM users WHERE id = $1', [id])).rows[0] : null;
  if (!u) return reply(ctx, 'User not found.');
  return reply(ctx, `👤 <code>${u.id}</code> ${esc(u.first_name || '')} ${u.username ? '@' + esc(u.username) : ''}\nJoined ${fmtDate(u.joined_at)} · last seen ${fmtDate(u.last_seen)}\nPlan: ${hasPremium(u) ? 'Premium until ' + fmtUntil(u.plan_until || Date.now()) : 'Free'} · push ${u.sub_daily ? 'on' : 'off'} · lang ${u.lang}\nBanned ${u.banned} · blocked bot ${u.blocked} · referred by ${u.referred_by || '—'}`);
}));
const broadcast = (premiumOnly) => adminOnly(async (ctx) => {
  const text = ctx.message.text.replace(/^\/\S+\s*/, '');
  if (!text) return reply(ctx, 'Usage: /broadcast &lt;message&gt; (HTML allowed)');
  const ids = (await q(`SELECT id FROM users WHERE NOT banned AND NOT blocked AND (excluded_until IS NULL OR excluded_until <= now()) ${premiumOnly ? 'AND plan_until > now()' : ''}`)).rows.map((r) => r.id);
  await reply(ctx, `📣 Sending to ${ids.length} users…`);
  (async () => {
    let ok = 0;
    for (const id of ids) { if (await sendTo(id, text)) ok++; await sleep(50); } // ~20 msgs/s, under Telegram's limit
    await sendTo(ctx.from.id, `📣 Broadcast done: ${ok}/${ids.length} delivered.`);
  })().catch(console.error);
});
bot.command('broadcast', broadcast(false));
bot.command('broadcastp', broadcast(true));
bot.command('forceupdate', adminOnly(async (ctx) => {
  const n = cache.size;
  cache.clear();
  seasonFreshAt.clear();
  await reply(ctx, `♻️ Cleared ${n} cached entries. The next /today or /top5 re-downloads everything (uses API credits).`);
}));

// =====================================================================
//  FEATURE PACK — handlers (hubs, new Premium tools, follow/record, promo, safety, admin dashboard)
// =====================================================================
const answered = (h) => async (ctx) => { await ctx.answerCbQuery().catch(() => {}); return h(ctx); };
const sendR = (fn) => async (ctx) => {
  try { const r = await fn(ctx); await replyLong(ctx, r.text, r.kb ? withNav(r.kb) : navKb()); }
  catch (e) { console.error(e); await ctx.reply(t(ctx.u, 'fail')); }
};

// ----- section hubs -----
const showHub = (name) => async (ctx) => { const v = hubView(name, ctx.u); return reply(ctx, v.text, v.kb); };
bot.command('picks', showHub('picks'));
bot.command('markets', showHub('markets'));
bot.command('accas', showHub('acca'));
bot.command('myzone', showHub('me'));
bot.action(/^hub:(picks|markets|acca|me)$/, async (ctx) => {
  await ctx.answerCbQuery().catch(() => {});
  const v = hubView(ctx.match[1], ctx.u);
  await editOrReply(ctx, v.text, v.kb);
});

// ----- new Premium tools (+ button handlers for the existing goals / dc) -----
const PREM_TOOLS = {
  best10pro: ['🔟⭐ Best 10 Pro', (c) => renderBest10Pro(c.u)],
  safe: ['🛡 Safe slip', (c) => renderSafe(c.u)],
  scores: ['🎯 Correct score', (c) => renderScores(c.u)],
  potw: ['🌟 Pick of the Week', (c) => renderPotw(c.u)],
  weekend: ['🏖 Weekend special', (c) => renderWeekend(c.u)],
  goals: ['⚽ Goals markets', (c) => renderGoals(c.u)],
  dc: ['🛡 Double chance', (c) => renderDC(c.u)],
};
for (const [name, [label, fn]] of Object.entries(PREM_TOOLS)) {
  const h = toolHandler(fn);
  if (!['goals', 'dc'].includes(name)) bot.command(name, needPremium(detach(h)));
  bot.action(name, async (ctx) => {
    await ctx.answerCbQuery().catch(() => {});
    if (!ctx.premium) return paywall(ctx, `🔒 <b>${label}</b> is a Premium feature.`);
    h(ctx).catch(console.error);
  });
}

// ----- full market boards + per-match "all markets" -----
const mkBoardH = (cat) => toolHandler((ctx) => renderMarketBoard(ctx.u, cat));
bot.action(/^mk:(res|goals|htft|combo|hcp)$/, async (ctx) => {
  await ctx.answerCbQuery().catch(() => {});
  const cat = ctx.match[1];
  if (!ctx.premium) return paywall(ctx, `🔒 <b>${MK_CATS[cat].title}</b> is a Premium feature.`);
  mkBoardH(cat)(ctx).catch(console.error);
});
for (const [cmd, cat] of [['result', 'res'], ['htft', 'htft'], ['combos', 'combo'], ['handicaps', 'hcp']]) bot.command(cmd, needPremium(detach(mkBoardH(cat))));
const allMarketsFor = (m) => toolHandler((ctx) => renderAllMarkets(m, ctx.u));
bot.action(/^am:(\d+)$/, async (ctx) => {
  await ctx.answerCbQuery().catch(() => {});
  if (!ctx.premium) return paywall(ctx, '🔒 <b>All markets</b> is a Premium feature.');
  const m = await findMatchAny(Number(ctx.match[1]));
  if (!m) return void (await reply(ctx, 'Match not found (list refreshed). Use /today again.'));
  allMarketsFor(m)(ctx).catch(console.error);
});
bot.command('market', needPremium(detach(async (ctx) => {
  const query = argText(ctx);
  if (!query) return void (await reply(ctx, 'Usage: <code>/market Arsenal Chelsea</code> or <code>/market Real Madrid</code> — every market (1X2, GG, Over/Under, HT/FT, combos, handicaps, corners &amp; cards) for the next match.'));
  const m = await findFixture(query);
  if (!m) return void (await reply(ctx, `🔎 No fixture found for “${esc(query)}” in the next 7 days. Try the team name as it appears in /today.`));
  await allMarketsFor(m)(ctx);
})));
bot.action('mkhelp', async (ctx) => {
  await ctx.answerCbQuery().catch(() => {});
  if (!ctx.premium) return paywall(ctx, '🔒 <b>All markets</b> is a Premium feature.');
  await editOrReply(ctx, '📋 <b>Every market for one match</b>\n\n• Open <b>⚽ Today</b>, tap a match, then <b>📋 All markets</b>\n• Or type <code>/market Arsenal Chelsea</code> (or just one team name)\n• On any market board, tap the numbered 📋 buttons', navKb([[Markup.button.callback('⚽ Today', 'today')]]));
});

// ----- follow a pick / personal record / leaderboard -----
bot.action(/^fp:(\d+)$/, async (ctx) => {
  const id = Number(ctx.match[1]);
  const r = predById.get(id);
  if (!r || !r.final || r.utc <= Date.now()) return ctx.answerCbQuery('This match has started — it can no longer be followed.', { show_alert: true });
  const today = (await q(`SELECT count(*)::int AS n FROM followed WHERE user_id = $1 AND created_at > now() - interval '1 day'`, [ctx.u.id])).rows[0].n;
  if (today >= MAX_FOLLOW_DAY) return ctx.answerCbQuery(`You can follow up to ${MAX_FOLLOW_DAY} picks a day.`, { show_alert: true });
  const k = r.pubPick || argmax(r.final);
  const odds = (r.pubOdds || r.odds || {})[k] || null;
  const ins = await q(
    `INSERT INTO followed (user_id, match_id, home, away, utc, pick, label, odds) VALUES ($1,$2,$3,$4,$5,$6,$7,$8) ON CONFLICT DO NOTHING RETURNING match_id`,
    [ctx.u.id, id, r.home, r.away, r.utc, k, pickName(r.home, r.away, k), odds]);
  return ctx.answerCbQuery(ins.rows.length ? '📌 Following — I\'ll message you the result.' : 'You already follow this pick.', { show_alert: !ins.rows.length });
});
bot.command('myrecord', sendR((c) => renderMyRecord(c.u)));
bot.action('myrec', answered(sendR((c) => renderMyRecord(c.u))));
bot.command('leaders', sendR((c) => renderLeaders(c.u)));
bot.action('leaders', answered(sendR((c) => renderLeaders(c.u))));
bot.action('trk', answered(async (ctx) => {
  await ctx.sendChatAction('typing').catch(() => {});
  try { await replyLong(ctx, await renderStats(), navKb([[Markup.button.callback('📜 Recent picks', 'hist')]])); }
  catch (e) { console.error(e); await ctx.reply('⚠️ Could not compute stats right now.'); }
}));
bot.action('hist', answered((ctx) => reply(ctx, renderHistory(), navKb())));

// ----- referral screen (invite + paying-friend bonuses) -----
async function sendRefer(ctx) {
  const me = ctx.botInfo || bot.botInfo || {};
  const n = (await q('SELECT count(*)::int AS n FROM users WHERE referred_by = $1', [ctx.u.id])).rows[0].n;
  const paying = (await q(`SELECT count(DISTINCT p.user_id)::int AS n FROM payments p JOIN users x ON x.id = p.user_id WHERE x.referred_by = $1 AND p.status = 'success'`, [ctx.u.id])).rows[0].n;
  const link = `https://t.me/${me.username}?start=ref_${ctx.u.id}`;
  const next = REF_MILESTONE_N > 0 ? REF_MILESTONE_N - (paying % REF_MILESTONE_N) : 0;
  const tx = '🎁 <b>Invite friends, earn Premium</b>\n━━━━━━━━━━━━━━\n' +
    `• A friend joins: you get <b>+${REF_BONUS_DAYS} days</b> (up to ${REF_MAX}), they get <b>${REF_NEW_DAYS} free day${REF_NEW_DAYS > 1 ? 's' : ''}</b>\n` +
    `• A friend pays for the first time: you get <b>+${REF_PAID_DAYS} days</b>\n` +
    (REF_MILESTONE_N > 0 ? `• Every <b>${REF_MILESTONE_N} paying friends</b>: bonus <b>+${REF_MILESTONE_DAYS} days</b> 🏅\n` : '') +
    `\n🔗 Your link:\n${link}\n\n👥 Joined: <b>${n}</b> · 💎 Paying: <b>${paying}</b>` + (next ? ` · next milestone in <b>${next}</b>` : '');
  await reply(ctx, tx, Markup.inlineKeyboard([[Markup.button.url('📤 Share with friends', `https://t.me/share/url?url=${encodeURIComponent(link)}&text=${encodeURIComponent('Daily football picks with odds & AI ⚽')}`)], [MENU_BTN]]));
}
bot.command('refer', sendRefer);
bot.action('refer', answered(sendRefer));

// ----- promo codes -----
bot.command('redeem', async (ctx) => {
  const r = await redeemCode(ctx.u, argText(ctx));
  await reply(ctx, r.msg, r.ok ? mainKb(r.u || ctx.u) : undefined);
});
bot.action('redeem', answered((ctx) => reply(ctx, '🎟 <b>Redeem a promo code</b>\nSend it like this: <code>/redeem YOURCODE</code>', navKb())));

// ----- responsible play, self-exclusion, free-tips toggle -----
const showResponsible = async (ctx) => { const v = responsibleView(ctx.u); return reply(ctx, v.text, v.kb); };
bot.command('responsible', showResponsible);
bot.command('selfexclude', showResponsible);
bot.action('resp', answered(async (ctx) => { const v = responsibleView(ctx.u); await editOrReply(ctx, v.text, v.kb); }));
bot.action(/^se:(1|7|30|90)$/, answered(async (ctx) => {
  const d = Number(ctx.match[1]);
  await editOrReply(ctx, `⚠️ <b>Confirm self-exclusion</b>\nYou will be locked out of picks, tips and alerts for <b>${d} day${d > 1 ? 's' : ''}</b>. This cannot be undone early.`,
    Markup.inlineKeyboard([[Markup.button.callback(`✅ Yes, pause me for ${d} day${d > 1 ? 's' : ''}`, `sec:${d}`)], [Markup.button.callback('↩️ Cancel', 'resp')]]));
}));
bot.action(/^sec:(1|7|30|90)$/, answered(async (ctx) => {
  const d = Number(ctx.match[1]);
  const { rows } = await q('UPDATE users SET excluded_until = now() + make_interval(days => $2::int), sub_daily = false, teaser_daily = false WHERE id = $1 RETURNING *', [ctx.u.id, d]);
  await editOrReply(ctx, `🛡 <b>Self-exclusion started</b> — until <b>${fmtUntil(rows[0].excluded_until)}</b>.\nTake care of yourself. Picks, tips and alerts are paused; /responsible stays available.`);
}));
bot.action(/^tsr:(on|off|toggle)$/, async (ctx) => {
  await ctx.answerCbQuery().catch(() => {});
  const mode = ctx.match[1];
  const on = mode === 'toggle' ? !ctx.u.teaser_daily : mode === 'on';
  ctx.u = (await updUser(ctx.u.id, { teaser_daily: on })) || ctx.u;
  if (mode === 'toggle') { const v = settingsView(ctx.u, ctx.premium); return editOrReply(ctx, v.text, v.kb); }
  return ctx.reply(on ? '📬 Free daily tips are back ON.' : '🔕 Free daily tips are OFF. You can turn them back on in ⚙️ Settings.');
});

// ----- admin: dashboard, promo codes, lift an exclusion -----
bot.command(['dashboard', 'dash'], adminOnly(async (ctx) => {
  const sum = async (iv) => (await q(`SELECT currency, sum(amount) AS total, count(*)::int AS n FROM payments WHERE status = 'success' AND created_at > now() - interval '${iv}' GROUP BY currency`)).rows;
  const f = (rows) => rows.map((r) => (r.currency === 'XTR' ? `${money(r.total)} ⭐` : `₦${money(r.total)}`)).join(' + ') || '0';
  const cnt = (rows) => rows.reduce((a, r) => a + r.n, 0);
  const [d1, d7, d30] = [await sum('1 day'), await sum('7 days'), await sum('30 days')];
  const u = (await q(`SELECT count(*)::int total,
    count(*) FILTER (WHERE joined_at > now() - interval '1 day')::int new1, count(*) FILTER (WHERE joined_at > now() - interval '7 days')::int new7,
    count(*) FILTER (WHERE joined_at > now() - interval '30 days')::int new30, count(*) FILTER (WHERE last_seen > now() - interval '1 day')::int act1,
    count(*) FILTER (WHERE last_seen > now() - interval '7 days')::int act7, count(*) FILTER (WHERE plan_until > now())::int premium,
    count(*) FILTER (WHERE plan_until > now() AND plan_until < now() + interval '3 days')::int expiring,
    count(*) FILTER (WHERE excluded_until > now())::int excluded, count(*) FILTER (WHERE teaser_daily AND NOT blocked AND NOT banned)::int tips,
    count(*) FILTER (WHERE blocked)::int blocked FROM users`)).rows[0];
  const pay = (await q(`SELECT count(DISTINCT user_id)::int AS payers, count(*) FILTER (WHERE rn > 1)::int AS renewals
    FROM (SELECT user_id, row_number() OVER (PARTITION BY user_id ORDER BY created_at) rn FROM payments WHERE status = 'success') x`)).rows[0];
  const lapsed = (await q(`SELECT count(*)::int AS n FROM users WHERE plan_until BETWEEN now() - interval '30 days' AND now()`)).rows[0].n;
  const mix = (await q(`SELECT plan, count(*)::int AS n FROM payments WHERE status = 'success' AND created_at > now() - interval '30 days' GROUP BY plan ORDER BY n DESC`)).rows;
  const promo = (await q(`SELECT (SELECT count(*)::int FROM promo_redemptions WHERE redeemed_at > now() - interval '30 days') AS red30,
    (SELECT count(*)::int FROM promo_codes WHERE uses < max_uses AND (expires_at IS NULL OR expires_at > now())) AS active`)).rows[0];
  const fol = (await q(`SELECT count(*) FILTER (WHERE created_at > now() - interval '1 day')::int AS d1, count(DISTINCT user_id)::int AS users FROM followed`)).rows[0];
  const conv = u.total ? (pay.payers / u.total) * 100 : 0;
  const churn = lapsed + u.premium ? (lapsed / (lapsed + u.premium)) * 100 : 0;
  const tx = '📊 <b>Dashboard</b>\n━━━━━━━━━━━━━━\n' +
    `💰 <b>Revenue</b>\nToday: ${f(d1)} (${cnt(d1)})\n7 days: ${f(d7)} (${cnt(d7)})\n30 days: ${f(d30)} (${cnt(d30)})\n` +
    `Plans sold (30d): ${mix.map((x) => `${PLANS[x.plan] ? PLANS[x.plan].label : x.plan} ×${x.n}`).join(' · ') || '—'}\n\n` +
    `👥 <b>Users</b> (${u.total})\nNew: ${u.new1} today · ${u.new7} / 7d · ${u.new30} / 30d\nActive: ${u.act1} / 24h · ${u.act7} / 7d\nBlocked bot: ${u.blocked} · Free tips ON: ${u.tips}\n\n` +
    `💎 <b>Premium</b>\nActive: ${u.premium} (${u.expiring} expiring in 3 days)\nConversion (ever paid ÷ all users): <b>${conv.toFixed(1)}%</b> (${pay.payers} payers)\nRenewal payments (all time): ${pay.renewals}\nChurn, approx. 30d: <b>${churn.toFixed(1)}%</b> (${lapsed} lapsed vs ${u.premium} active)\n\n` +
    `🎟 <b>Promo</b>: ${promo.active} active codes · ${promo.red30} redeemed (30d)\n📌 <b>Follows</b>: ${fol.d1} today · ${fol.users} users ever\n🛡 Self-excluded now: ${u.excluded}`;
  await reply(ctx, tx);
}));
bot.command('promo', adminOnly(async (ctx) => {
  const a = argText(ctx).split(/\s+/).filter(Boolean);
  const sub = (a[0] || '').toLowerCase();
  if (sub === 'create') {
    const code = String(a[1] || '').toUpperCase();
    const days = Math.round(Number(a[2])), max = Math.round(Number(a[3] || 1)), exp = a[4] ? Math.round(Number(a[4])) : null;
    if (!/^[A-Z0-9_-]{3,24}$/.test(code) || !(days > 0) || !(max > 0)) return reply(ctx, 'Usage: /promo create &lt;CODE&gt; &lt;days&gt; [maxUses=1] [expiresInDays]\nExample: <code>/promo create WELCOME7 7 100 30</code>');
    await q(`INSERT INTO promo_codes (code, days, max_uses, expires_at) VALUES ($1, $2, $3, CASE WHEN $4::int IS NULL THEN NULL ELSE now() + make_interval(days => $4::int) END)
      ON CONFLICT (code) DO UPDATE SET days = EXCLUDED.days, max_uses = EXCLUDED.max_uses, expires_at = EXCLUDED.expires_at`, [code, days, max, exp]);
    const me = ctx.botInfo || bot.botInfo || {};
    return reply(ctx, `✅ Code <code>${code}</code>: ${days} day(s) Premium · ${max} use(s)${exp ? ` · expires in ${exp} day(s)` : ''}\nUsers redeem with <code>/redeem ${code}</code> or this link:\nhttps://t.me/${me.username}?start=promo_${code}`);
  }
  if (sub === 'list') {
    const rows = (await q('SELECT * FROM promo_codes ORDER BY created_at DESC LIMIT 25')).rows;
    if (!rows.length) return reply(ctx, 'No promo codes yet. Create one: /promo create &lt;CODE&gt; &lt;days&gt; [maxUses] [expiresInDays]');
    return reply(ctx, '🎟 <b>Promo codes</b>\n' + rows.map((r) => `<code>${esc(r.code)}</code> · ${r.days}d · ${r.uses}/${r.max_uses}${r.expires_at ? ' · exp ' + fmtDate(r.expires_at) : ''}`).join('\n'));
  }
  if (sub === 'del' || sub === 'delete') {
    const code = String(a[1] || '').toUpperCase();
    const r = await q('DELETE FROM promo_codes WHERE code = $1 RETURNING code', [code]);
    return reply(ctx, r.rows.length ? `🗑 Deleted <code>${esc(code)}</code>.` : 'Code not found.');
  }
  return reply(ctx, '🎟 <b>Promo codes</b>\n/promo create &lt;CODE&gt; &lt;days&gt; [maxUses=1] [expiresInDays]\n/promo list\n/promo del &lt;CODE&gt;');
}));
bot.command('unexclude', adminOnly(async (ctx) => {
  const id = Number(argText(ctx));
  if (!id) return reply(ctx, 'Usage: /unexclude &lt;userId&gt;');
  await q('UPDATE users SET excluded_until = NULL WHERE id = $1', [id]);
  return reply(ctx, `✅ Self-exclusion lifted for <code>${id}</code>.`);
}));

bot.catch((err) => console.error('Bot error:', err));
process.on('unhandledRejection', (e) => console.error('Unhandled rejection:', e));
process.on('uncaughtException', (e) => console.error('Uncaught exception:', e));

const PUBLIC_COMMANDS = [
  { command: 'menu', description: 'Open the main menu' },
  { command: 'best10', description: 'Best 10 picks of the day (free)' },
  { command: 'picks', description: 'All pick lists' },
  { command: 'top5', description: 'Top 5 picks of the day (Premium)' },
  { command: 'best3', description: 'Best 3 under-the-radar picks (Premium)' },
  { command: 'best1', description: 'Best single pick of the day (Premium)' },
  { command: 'best10pro', description: 'Best 10 Pro with confidence tiers (Premium)' },
  { command: 'potw', description: 'Pick of the Week (Premium)' },
  { command: 'today', description: "Today's matches with odds" },
  { command: 'live', description: 'Live scores' },
  { command: 'table', description: 'League table' },
  { command: 'predict', description: 'Predict any upcoming match' },
  { command: 'value', description: 'Value bets (Premium)' },
  { command: 'goals', description: 'Goals markets (Premium)' },
  { command: 'dc', description: 'Safer markets: DC / DNB (Premium)' },
  { command: 'scores', description: 'Correct-score suggestions (Premium)' },
  { command: 'result', description: '1X2, Double Chance, DNB board (Premium)' },
  { command: 'htft', description: 'HT/FT, win either half, score both halves (Premium)' },
  { command: 'combos', description: 'Combo markets: 1 & GG, 1 & OV 2.5… (Premium)' },
  { command: 'handicaps', description: 'Handicaps and Asian handicap (Premium)' },
  { command: 'market', description: 'Every market for one match (Premium)' },
  { command: 'acca', description: 'Accumulator builder (Premium)' },
  { command: 'safe', description: 'Safe 2-odds slip (Premium)' },
  { command: 'weekend', description: 'Weekend special acca (Premium)' },
  { command: 'myrecord', description: 'Your followed picks and win rate' },
  { command: 'leaders', description: 'Monthly leaderboard' },
  { command: 'watchlist', description: 'Matches you follow (Premium)' },
  { command: 'stake', description: 'Kelly stake calculator' },
  { command: 'subscribe', description: 'Daily picks in your chat (Premium)' },
  { command: 'settings', description: 'Leagues, push time, language' },
  { command: 'premium', description: 'Unlock everything' },
  { command: 'redeem', description: 'Redeem a promo code' },
  { command: 'refer', description: 'Invite friends, earn free days' },
  { command: 'account', description: 'Your plan and usage' },
  { command: 'stats', description: 'Model accuracy & ROI' },
  { command: 'history', description: 'Recent published picks' },
  { command: 'calibration', description: 'Are the probabilities honest?' },
  { command: 'responsible', description: 'Responsible play & self-exclusion' },
  { command: 'help', description: 'How predictions work' },
];

// ----- startup -----
async function main() {
  await initDb();
  await loadPredLog();
  console.log(`PostgreSQL ready · ${predLog.length} logged predictions · admins: ${ADMIN_IDS.length || 'NONE (set ADMIN_IDS)'} · Paystack: ${PAYSTACK_KEY ? 'on' : 'off'} · Stars: ${STARS_ON ? 'on' : 'off'}`);
  bot.telegram.setMyCommands(PUBLIC_COMMANDS).catch(() => {});
  for (const id of ADMIN_IDS) {
    bot.telegram.setMyCommands([...PUBLIC_COMMANDS, { command: 'admin', description: 'Admin tools' }], { scope: { type: 'chat', chat_id: id } }).catch(() => {});
  }
  if (AI_ON && !process.env.AI_MODELS) discoveryDone = discoverModels().catch(() => false); // AI_MODELS set in .env = manual choice, respected
  setInterval(() => schedulerTick().catch((e) => console.log('tick failed:', e.message)), 60e3).unref();
  setInterval(() => flushPredLog().catch(() => {}), 60e3).unref();
  bot.launch({ allowedUpdates: ['message', 'callback_query', 'inline_query', 'pre_checkout_query'] })
    .then(() => console.log(`Football bot started · AI: ${AI_ON ? PROVIDER + ' (' + HF_MODELS.join(', ') + ')' : 'off (odds + form model only)'}`))
    .catch((e) => { console.error('Launch failed:', e); process.exit(1); });
}
main().catch((e) => { console.error('Startup failed:', e); process.exit(1); });
const shutdown = (sig) => { flushPredLog().catch(() => {}).finally(() => { try { bot.stop(sig); } catch {} setTimeout(() => process.exit(0), 500); }); };
process.once('SIGINT', () => shutdown('SIGINT'));
process.once('SIGTERM', () => shutdown('SIGTERM'));