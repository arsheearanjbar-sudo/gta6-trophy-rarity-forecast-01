// GTA VI trophy rarity forecast: a counts-based forecaster.
//
// An Exophase earn rate is a ratio, earners over players tracked, and both
// sides move. In launch week the tracked population grows fast (new players
// keep joining) while earners accumulate as players play. So instead of
// extrapolating the rate itself, this program models the two counts
// separately and divides at the target:
//
//   * N(t), the players tracked. Fitted as a power law in time since the
//     game's release, log N = a + b * log(1 + t): launch-week growth is fast
//     and concave. When the history shows no growth at all, the page is
//     treated as stale: the rules record the launch-night page sitting
//     unchanged for hours while the true population grew underneath it, so a
//     conservative surge prior (0.30 per hour from the first read) stands in.
//   * E_i(t), the earners of trophy i. Every player who has not earned the
//     trophy earns it at a per-trophy hazard lambda_i per hour, so
//     dE/dt = lambda_i * (N(t) - E(t)). lambda_i is fitted per trophy by
//     least squares on the logit scale over the case's reads.
//
// The median is the ODE integrated to the target. The spread is honest about
// four things: the fit's residuals, the population forecast's uncertainty,
// the fact that the answering read shows the page as of its last refresh
// (which may lag the target by hours), and a random-walk floor for what the
// data cannot see.
//
// Node 22, standard library only. Deterministic: no randomness, no clock,
// no network. Reads every <id>.in in the input directory and writes
// <id>.out to the output directory.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The spread of a rate's move, in logit units per square-root day. */
const SIGMA = 0.5;
/** Hours a page refresh may lag the target by (the rules saw ~4.5 h of staleness). */
const STALENESS_HOURS = 5;
/** Surge rate per hour used when the history shows no population growth. */
const STALE_SURGE_PER_HOUR = 0.3;
/** Largest population growth factor the forecast ever claims. */
const MAX_GROWTH_FACTOR = 50;
/** ODE integration step, in hours. */
const STEP_HOURS = 0.25;

interface Trophy {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes?: string;
}

const clampPct = (x: number): number => (x < 0.01 ? 0.01 : x > 99.99 ? 99.99 : x);
const logitOf = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10000) / 10000;
const hoursBetween = (aMs: number, bMs: number): number => (bMs - aMs) / 3600000;

/** The population model: N(t) for any time, in hours since release. */
interface Population {
  /** Players tracked at a time t (hours since release). */
  nAt: (t: number) => number;
  /** Log-growth rate of the forecast N at the target, per hour. */
  growthAtTarget: number;
  /** Log-uncertainty of the forecast N at the target. */
  sigmaLogN: number;
  /** Players tracked at the last read. */
  nLast: number;
  /** Forecast players tracked at the target. */
  nTarget: number;
}

/**
 * Fit the tracked population. Distinct (time, N) points from the history
 * feed a power law in time since release; when the history is flat, a
 * stale-page surge prior stands in instead.
 */
function fitPopulation(
  c: CaseFile,
  tReleaseMs: number,
  tLastMs: number,
  tTargetMs: number,
): Population {
  const points: Array<{ t: number; n: number }> = [];
  for (const h of c.history) {
    const t = hoursBetween(tReleaseMs, Date.parse(h.at));
    const n = h.players_tracked;
    if (!Number.isFinite(t) || !Number.isFinite(n) || n <= 0) continue;
    const prev = points[points.length - 1];
    if (prev && prev.t === t) continue;
    points.push({ t, n });
  }
  const nLast = points.length > 0 ? points[points.length - 1].n : 1;
  const tLast = hoursBetween(tReleaseMs, tLastMs);
  const tTarget = hoursBetween(tReleaseMs, tTargetMs);
  const distinctN = new Set(points.map((p) => p.n)).size;

  if (points.length >= 3 && distinctN >= 2) {
    const xs = points.map((p) => Math.log(1 + p.t));
    const ys = points.map((p) => Math.log(p.n));
    const mx = xs.reduce((s, x) => s + x, 0) / xs.length;
    const my = ys.reduce((s, y) => s + y, 0) / ys.length;
    let sxx = 0;
    let sxy = 0;
    for (let k = 0; k < xs.length; k++) {
      sxx += (xs[k] - mx) ** 2;
      sxy += (xs[k] - mx) * (ys[k] - my);
    }
    const b = sxx > 1e-12 ? Math.max(0, sxy / sxx) : 0;
    const a = my - b * mx;
    let ss = 0;
    for (let k = 0; k < xs.length; k++) ss += (ys[k] - (a + b * xs[k])) ** 2;
    const rho = Math.sqrt(ss / xs.length);
    const growthFactor = Math.min(
      MAX_GROWTH_FACTOR,
      Math.max(1, ((1 + tTarget) / (1 + tLast)) ** b),
    );
    const nTarget = nLast * growthFactor;
    // Two distinct values pin a line exactly but say nothing about its
    // error, so a fit resting on them carries extra uncertainty.
    const thinFitPenalty = distinctN < 3 ? 0.25 : 0;
    const sigma = Math.min(
      1.0,
      Math.max(
        0.25,
        Math.sqrt(rho * rho + 0.01 * Math.log(growthFactor) ** 2) + thinFitPenalty,
      ),
    );
    const nAt = (t: number): number => {
      if (t <= tLast) {
        // Piecewise-linear interpolation of the observed population.
        if (t <= points[0].t) return points[0].n;
        for (let k = 1; k < points.length; k++) {
          if (t <= points[k].t) {
            const f = (t - points[k - 1].t) / (points[k].t - points[k - 1].t);
            return points[k - 1].n + f * (points[k].n - points[k - 1].n);
          }
        }
        return nLast;
      }
      return nLast * ((1 + t) / (1 + tLast)) ** b;
    };
    return {
      nAt,
      growthAtTarget: b / (1 + tTarget),
      sigmaLogN: sigma,
      nLast,
      nTarget,
    };
  }

  // Flat history: the page is stale. The true population has been growing
  // underneath it at the surge prior since the first read.
  const tFirst = points.length > 0 ? points[0].t : tLast;
  const surgeHours = Math.max(0, tTarget - tFirst);
  const growthFactor = Math.min(
    MAX_GROWTH_FACTOR,
    Math.exp(STALE_SURGE_PER_HOUR * surgeHours),
  );
  const nTarget = nLast * growthFactor;
  const nAt = (t: number): number => {
    if (t <= tLast) return nLast;
    return nLast * Math.exp(STALE_SURGE_PER_HOUR * (t - tFirst));
  };
  return {
    nAt,
    growthAtTarget: STALE_SURGE_PER_HOUR,
    sigmaLogN: 0.7,
    nLast,
    nTarget,
  };
}

/** One trophy's observations: time (hours since release), N, rate in percent. */
interface Observation {
  t: number;
  n: number;
  rate: number;
}

/**
 * Integrate the earner ODE dE/dt = lambda * (N(t) - E(t)) from t0 to t1,
 * starting at E0, in STEP_HOURS steps. Returns the earners at t1.
 */
function integrateEarners(
  lambda: number,
  e0: number,
  t0: number,
  t1: number,
  nAt: (t: number) => number,
): number {
  let e = e0;
  let t = t0;
  while (t < t1 - 1e-9) {
    const dt = Math.min(STEP_HOURS, t1 - t);
    const n = nAt(t + dt / 2);
    e += lambda * (n - e) * dt;
    if (e < 0) e = 0;
    t += dt;
  }
  return e;
}

/**
 * Fit one trophy's earn hazard by grid search over log10 lambda, scored by
 * the squared logit error of the predicted rates at the observed reads.
 * Each candidate integrates the ODE once, stepping through the reads in
 * order and recording the predicted rate at each read time. Returns the
 * fitted lambda, the fit's RMS logit residual, and the median rate at the
 * target.
 */
function fitTrophy(
  obs: Observation[],
  pop: Population,
  tLast: number,
  tTarget: number,
): { lambda: number; rho: number; medianPct: number } {
  const first = obs[0];
  const e0 = (first.rate / 100) * first.n;
  const t0 = first.t;

  // Candidate hazards, ascending so ties prefer the smaller (more cautious) one.
  const candidates: number[] = [0];
  for (let k = -100; k <= 10; k++) candidates.push(10 ** (k / 20));

  let bestLambda = 0;
  let bestSse = Number.POSITIVE_INFINITY;
  for (const lambda of candidates) {
    // One pass: integrate from the first read to the last, sampling the
    // predicted rate at every observed read on the way.
    let sse = 0;
    let e = e0;
    let t = t0;
    for (const o of obs) {
      while (t < o.t - 1e-9) {
        const dt = Math.min(STEP_HOURS, o.t - t);
        const n = pop.nAt(t + dt / 2);
        e += lambda * (n - e) * dt;
        if (e < 0) e = 0;
        t += dt;
      }
      const rPred = (e / o.n) * 100;
      const d = logitOf(rPred) - logitOf(o.rate);
      sse += d * d;
    }
    if (sse < bestSse - 1e-12) {
      bestSse = sse;
      bestLambda = lambda;
    }
  }
  const rho = obs.length > 0 ? Math.sqrt(bestSse / obs.length) : 0;
  const eTarget = integrateEarners(bestLambda, e0, t0, tTarget, pop.nAt);
  const medianPct = clampPct((eTarget / pop.nTarget) * 100);
  return { lambda: bestLambda, rho, medianPct };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) {
    // No reads at all: a neutral, wide forecast rather than a crash.
    const trophies = c.trophies.map((t) => ({
      id: t.id,
      q: Z.map((z) => round4(invLogit(logitOf(1) + z * 1.5))),
    }));
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
    continue;
  }

  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const tReleaseMs = Number.isFinite(releaseMs)
    ? releaseMs
    : Date.parse(c.history[0].at);
  const tLastMs = Date.parse(c.history[c.history.length - 1].at);
  const tTargetMs = Date.parse(c.target);
  const tLast = hoursBetween(tReleaseMs, tLastMs);
  const tTarget = hoursBetween(tReleaseMs, tTargetMs);
  const horizonHours = Math.max(0, hoursBetween(tLastMs, tTargetMs));
  const horizonDays = horizonHours / 24;
  const histHours = Math.max(
    0.5,
    hoursBetween(tReleaseMs, tLastMs) - hoursBetween(tReleaseMs, Date.parse(c.history[0].at)),
  );

  const pop = fitPopulation(c, tReleaseMs, tLastMs, tTargetMs);
  const last = c.history[c.history.length - 1];

  const trophies = c.trophies.map((t, i) => {
    const lastRate = last.rates[i] ?? 0;
    // A trophy every tracked player has earned stays earned: the refresh
    // the rules record left the one rate at 100% exactly where it was.
    if (lastRate >= 99.99) {
      return { id: t.id, q: [99.9, 99.97, 99.99, 100, 100] };
    }
    const obs: Observation[] = [];
    for (const h of c.history) {
      const r = h.rates[i];
      if (r === null || r === undefined) continue;
      obs.push({ t: hoursBetween(tReleaseMs, Date.parse(h.at)), n: h.players_tracked, rate: r });
    }
    if (obs.length === 0) {
      // The case never saw this trophy: a wide band around a small rate.
      return { id: t.id, q: Z.map((z) => round4(invLogit(logitOf(0.5) + z * 1.5))) };
    }
    const { lambda, rho, medianPct } = fitTrophy(obs, pop, tLast, tTarget);

    // The spread: four honest parts, combined in quadrature.
    const sigmaFloor = SIGMA * Math.sqrt(horizonDays);
    const sigmaLambda = rho * Math.sqrt(1 + horizonHours / histHours) + 0.15;
    const sigmaN = pop.sigmaLogN;
    // The answering read shows the page as of its last refresh, which may
    // lag the target by up to STALENESS_HOURS: the rate drifts that far.
    const rFrac = medianPct / 100;
    const driftPerHour = Math.abs(lambda * (1 - rFrac) - rFrac * pop.growthAtTarget) / Math.max(1e-9, rFrac * (1 - rFrac));
    const sigmaStale = Math.min(1.0, driftPerHour * STALENESS_HOURS);
    const sigma = Math.min(
      2.0,
      Math.max(
        0.2,
        Math.sqrt(sigmaFloor ** 2 + sigmaLambda ** 2 + sigmaN ** 2 + sigmaStale ** 2),
      ),
    );
    const l = logitOf(medianPct);
    return { id: t.id, q: Z.map((z) => round4(clampPct(invLogit(l + z * sigma)))) };
  });
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
