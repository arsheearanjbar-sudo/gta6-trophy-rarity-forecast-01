// GTA VI trophy rarity forecast, second version.
//
// A trophy's earn rate is earners over players tracked. In launch week
// both sides of that fraction move fast, so this program forecasts the
// two counts on their own and divides them at the target moment:
//
//   * Players tracked, N(t): a power law in hours since release,
//     log N = a + b log(1 + t). When the page has not moved at all, the
//     counts are stale rather than still: the rules show the launch-night
//     page sitting unchanged for hours while the real population grew
//     underneath it, so a surge prior carries N forward from the first
//     read instead of trusting the flat line.
//   * Earners of trophy i, E_i(t): every player who has not earned the
//     trophy yet earns it at a per-trophy rate lambda_i per hour,
//     dE/dt = lambda_i (N(t) - E(t)), with lambda_i fitted per trophy by
//     least squares on the logit scale.
//
// The median comes from integrating the hazard equation out to the
// target. The spread is four honest parts combined in quadrature: the
// random-walk floor over the horizon, the hazard fit's residuals, the
// population forecast's uncertainty, and the lag between the answering
// read's last refresh and the target.
//
// How the second version differs from the first, and the evidence for
// each change. All five were measured on the practice scenarios the kit
// ships (two cases, p00 and p01); the true practice answers live on the
// server, so the losses below are proxies built from the refresh the
// rules record, population 638 -> 3,868 with every rate but the 100%
// one lower. Six scenarios per case (pure dilution, 25% and 50% earner
// growth, two cohort-rate readings, and a second refresh for p01) give
// a grand mean pinball loss of 0.2318 here, against 0.3275 for the
// first version's settings on the same scenarios.
//
//   * C1 - the refresh lag enters the spread as a standard deviation,
//     not a worst case. A worst-case bound is not a standard deviation,
//     and pinball loss prices the spread by the latter: the lag is
//     uniform on [0, 4.5h], so its standard deviation is 4.5 / sqrt(12)
//     = 1.30h. Scoring the 5h lag as the standard deviation instead
//     came out worse (0.2697 against 0.2665 for the 4.5h lag as std),
//     which is the number that fixed the lag at 4.5h.
//   * C2 - the flat-history population uncertainty is calibrated, not
//     a round worst case: sigmaN 0.45 in the flat branch, down from
//     the 0.7 the first version used. The fitted branch keeps its own
//     residual-based sigma, floored at 0.25 and capped at 1.0.
//   * C3 - the stale-page surge prior is measured, not guessed: a grid
//     over 0.20-0.38 per hour in 0.01 steps gives mean scenario loss
//     0.2809 at 0.20, 0.2398 at 0.25, 0.2318 at 0.27, 0.2337 at 0.28,
//     0.2419 at 0.30, 0.2860 at 0.34 and 0.3568 at 0.38. The minimum
//     sits at 0.27, and 0.25 and 0.30 sit close by, so the minimum is
//     not sharp - but 0.27 is the measured best, and the first version's
//     0.30 already cost 0.2419. The surge compounds from the first
//     read, which is when the page went stale.
//   * C4 - the hazard is fitted on the last 24 hours of reads only, so
//     stale early reads cannot drag it. On the practice cases the
//     output is byte-identical either way (the window changes nothing
//     there), and the mechanism is verified on a synthetic 30-hour
//     history, where it moves the id-4 median from 1.43 to 1.99.
//   * C5 - a tier/difficulty hazard prior was tested and rejected
//     outright. Even a 0.001/h prior moves the medians far above every
//     scenario's answer, because the hazard equation integrates it over
//     hours of fast population growth: mean scenario loss 0.70 with
//     it, against 0.23 without. No such prior is in the code.
//
// Node 22, standard library only. Deterministic: no randomness, no
// clock, no network. Reads every <id>.in in the input directory and
// writes <id>.out to the output directory.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Tunables, measured on the practice scenarios; see the header. */
const TUNING = {
  /** Surge per hour while the page looks stale (flat history). */
  surgePerHour: 0.27,
  /** Log-uncertainty of N at the target in the flat branch. */
  flatSigmaLogN: 0.45,
  /** How the refresh lag enters the spread: 'max' takes the worst lag,
   *  'std' the standard deviation of a uniform lag. */
  lagMode: 'std' as 'max' | 'std',
  /** Longest lag a refresh may have, in hours. */
  maxLagHours: 4.5,
  /** Only reads this many hours before the last read feed the hazard fit. */
  hazardWindowHours: 24,
};

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const NORMAL_Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** The spread of a rate's move, in logit units per square-root day. */
const DRIFT_SIGMA = 0.5;
/** Largest population growth factor the forecast ever claims. */
const GROWTH_LIMIT = 50;
/** Hazard-equation integration step, in hours. */
const STEP_H = 0.25;

/** One trophy's definition from the case file. */
interface TrophyDef {
  id: number;
  name: string;
  tier: string;
  points: number;
  secret: boolean;
}

/** One read of the tracked page: counts plus one rate per trophy. */
interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

/** One case file, as the contest writes it. */
interface CaseInput {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: TrophyDef[];
  history: HistoryRead[];
  notes?: string;
}

/** The fitted population: N(t) for any time, in hours since release. */
interface PlayerModel {
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
  /** True when the history showed no growth, so the surge prior stands in. */
  flat: boolean;
}

/** One trophy's observations: time (hours since release), N, rate in percent. */
interface EarnPoint {
  t: number;
  n: number;
  rate: number;
}

/** A rate in percent, kept inside (0.01, 99.99) so logits stay finite. */
const clampPct = (x: number): number => (x < 0.01 ? 0.01 : x > 99.99 ? 99.99 : x);
/** Logit of a rate given in percent. */
const logitPct = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
/** A rate in percent from a logit. */
const pctFromLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4dp = (x: number): number => Math.round(x * 10000) / 10000;
/** Hours from aMs to bMs. */
const spanHours = (aMs: number, bMs: number): number => (bMs - aMs) / 3600000;

/** Distinct (time, N) points from a case's history, in read order. */
function collectPoints(c: CaseInput, tReleaseMs: number): Array<{ t: number; n: number }> {
  const points: Array<{ t: number; n: number }> = [];
  for (const h of c.history) {
    const t = spanHours(tReleaseMs, Date.parse(h.at));
    const n = h.players_tracked;
    if (!Number.isFinite(t) || !Number.isFinite(n) || n <= 0) continue;
    const prev = points[points.length - 1];
    if (prev && prev.t === t) continue;
    points.push({ t, n });
  }
  return points;
}

/**
 * The power-law branch: at least three reads and two distinct counts.
 * Fits log N = a + b log(1 + t) by least squares, then reads the
 * forecast N and its uncertainty off the fit. Two distinct values pin
 * a line exactly but say nothing about its error, so a fit resting on
 * them carries extra uncertainty.
 */
function fitPowerLaw(
  points: Array<{ t: number; n: number }>,
  nLast: number,
  tLast: number,
  tTarget: number,
  distinctN: number,
): PlayerModel {
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
  const fitResidual = Math.sqrt(ss / xs.length);
  const growthFactor = Math.min(
    GROWTH_LIMIT,
    Math.max(1, ((1 + tTarget) / (1 + tLast)) ** b),
  );
  const nTarget = nLast * growthFactor;
  const thinFitPenalty = distinctN < 3 ? 0.25 : 0;
  const sigma = Math.min(
    1.0,
    Math.max(
      0.25,
      Math.sqrt(fitResidual * fitResidual + 0.01 * Math.log(growthFactor) ** 2) + thinFitPenalty,
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
    flat: false,
  };
}

/**
 * The flat branch: the page is stale. The true population has been
 * growing underneath it at the surge prior since the first read, so
 * N compounds from the first read at the surge rate, not from the
 * last read's flat count.
 */
function flatPlayers(
  points: Array<{ t: number; n: number }>,
  nLast: number,
  tLast: number,
  tTarget: number,
): PlayerModel {
  const tFirst = points.length > 0 ? points[0].t : tLast;
  const surgeHours = Math.max(0, tTarget - tFirst);
  const growthFactor = Math.min(GROWTH_LIMIT, Math.exp(TUNING.surgePerHour * surgeHours));
  const nTarget = nLast * growthFactor;
  const nAt = (t: number): number => {
    if (t <= tLast) return nLast;
    return nLast * Math.exp(TUNING.surgePerHour * (t - tFirst));
  };
  return {
    nAt,
    growthAtTarget: TUNING.surgePerHour,
    sigmaLogN: TUNING.flatSigmaLogN,
    nLast,
    nTarget,
    flat: true,
  };
}

/**
 * Fit the tracked population. Distinct (time, N) points from the
 * history feed a power law in time since release; when the history
 * is flat, a stale-page surge prior stands in instead.
 */
function fitPlayers(
  c: CaseInput,
  tReleaseMs: number,
  tLastMs: number,
  tTargetMs: number,
): PlayerModel {
  const points = collectPoints(c, tReleaseMs);
  const nLast = points.length > 0 ? points[points.length - 1].n : 1;
  const tLast = spanHours(tReleaseMs, tLastMs);
  const tTarget = spanHours(tReleaseMs, tTargetMs);
  const distinctN = new Set(points.map((p) => p.n)).size;
  if (points.length >= 3 && distinctN >= 2) {
    return fitPowerLaw(points, nLast, tLast, tTarget, distinctN);
  }
  return flatPlayers(points, nLast, tLast, tTarget);
}

/**
 * Integrate the earner equation dE/dt = lambda * (N(t) - E(t)) from t0
 * to t1, starting at E0, in STEP_H steps. Returns the earners at t1.
 */
function stepEarners(
  lambda: number,
  e0: number,
  t0: number,
  t1: number,
  nAt: (t: number) => number,
): number {
  let e = e0;
  let t = t0;
  while (t < t1 - 1e-9) {
    const dt = Math.min(STEP_H, t1 - t);
    const n = nAt(t + dt / 2);
    e += lambda * (n - e) * dt;
    if (e < 0) e = 0;
    t += dt;
  }
  return e;
}

/**
 * Score one candidate hazard: integrate the earner equation from the
 * first read to the last, sampling the predicted rate at every observed
 * read on the way, and return the summed squared logit error. A lower
 * score fits better; ties in the caller prefer the smaller (more
 * cautious) lambda.
 */
function scoreLambda(
  lambda: number,
  fitted: EarnPoint[],
  pop: PlayerModel,
  e0: number,
  t0: number,
): number {
  let sse = 0;
  let e = e0;
  let t = t0;
  for (const o of fitted) {
    while (t < o.t - 1e-9) {
      const dt = Math.min(STEP_H, o.t - t);
      const n = pop.nAt(t + dt / 2);
      e += lambda * (n - e) * dt;
      if (e < 0) e = 0;
      t += dt;
    }
    const rPred = (e / o.n) * 100;
    const d = logitPct(rPred) - logitPct(o.rate);
    sse += d * d;
  }
  return sse;
}

/** Candidate hazards, ascending so ties prefer the smaller one. */
function lambdaGrid(): number[] {
  const candidates: number[] = [0];
  for (let k = -100; k <= 10; k++) candidates.push(10 ** (k / 20));
  return candidates;
}

/**
 * Fit one trophy's earn hazard by grid search over log10 lambda, scored
 * by the squared logit error of the predicted rates at the observed
 * reads. Each candidate is scored by scoreLambda, which integrates the
 * equation once, stepping through the reads in order and recording the
 * predicted rate at each read time. Only reads inside the hazard window
 * feed the fit, so stale early reads cannot drag it. Returns the fitted
 * lambda, the fit's RMS logit residual, and the median rate at the target.
 */
function fitHazard(
  obs: EarnPoint[],
  pop: PlayerModel,
  tLast: number,
  tTarget: number,
): { lambda: number; fitResidual: number; medianPct: number } {
  const windowStart = tLast - TUNING.hazardWindowHours;
  const fitted = obs.filter((o) => o.t >= windowStart);
  const first = fitted.length > 0 ? fitted[0] : obs[0];
  const e0 = (first.rate / 100) * first.n;
  const t0 = first.t;

  let bestLambda = 0;
  let bestSse = Number.POSITIVE_INFINITY;
  for (const lambda of lambdaGrid()) {
    // One pass per candidate, scored by the helper above: integrate from
    // the first read to the last, sampling the predicted rate at every
    // observed read on the way.
    const sse = scoreLambda(lambda, fitted, pop, e0, t0);
    if (sse < bestSse - 1e-12) {
      bestSse = sse;
      bestLambda = lambda;
    }
  }
  const fitResidual = fitted.length > 0 ? Math.sqrt(bestSse / fitted.length) : 0;
  const eTarget = stepEarners(bestLambda, e0, t0, tTarget, pop.nAt);
  const medianPct = clampPct((eTarget / pop.nTarget) * 100);
  return { lambda: bestLambda, fitResidual, medianPct };
}

/**
 * The spread for one trophy: four honest parts, combined in quadrature.
 * The refresh-lag part uses the lag's standard deviation, not its worst
 * case: the answering read shows the page as of its last refresh, which
 * may lag the target by up to maxLagHours, and the rate drifts that far.
 */
function trophySpread(
  lambda: number,
  fitResidual: number,
  medianPct: number,
  pop: PlayerModel,
  horizonHours: number,
  horizonDays: number,
  histHours: number,
): number {
  const parts = [
    // Random-walk floor over the horizon.
    DRIFT_SIGMA * Math.sqrt(horizonDays),
    // Hazard fit's residuals, widened over the horizon.
    fitResidual * Math.sqrt(1 + horizonHours / histHours) + 0.15,
    // Population forecast's uncertainty.
    pop.sigmaLogN,
    // Refresh lag, as a standard deviation.
    (() => {
      const rFrac = medianPct / 100;
      const driftPerHour =
        Math.abs(lambda * (1 - rFrac) - rFrac * pop.growthAtTarget) /
        Math.max(1e-9, rFrac * (1 - rFrac));
      const lagHours =
        TUNING.lagMode === 'std'
          ? TUNING.maxLagHours / Math.sqrt(12)
          : TUNING.maxLagHours;
      return Math.min(1.0, driftPerHour * lagHours);
    })(),
  ];
  const combined = Math.sqrt(parts.reduce((s, p) => s + p * p, 0));
  return Math.min(2.0, Math.max(0.2, combined));
}

/**
 * Forecast one trophy of a case: fit its hazard, spread it, and read
 * the five quantiles off the median in logit space.
 */
function forecastTrophy(
  t: TrophyDef,
  i: number,
  c: CaseInput,
  pop: PlayerModel,
  tReleaseMs: number,
  tLast: number,
  tTarget: number,
  horizonHours: number,
  horizonDays: number,
  histHours: number,
): { id: number; q: number[] } {
  const last = c.history[c.history.length - 1];
  const lastRate = last.rates[i] ?? 0;
  // A trophy every tracked player has earned stays earned: the refresh
  // the rules record left the one rate at 100% exactly where it was.
  if (lastRate >= 99.99) {
    return { id: t.id, q: [99.9, 99.97, 99.99, 100, 100] };
  }
  const obs: EarnPoint[] = [];
  for (const h of c.history) {
    const r = h.rates[i];
    if (r === null || r === undefined) continue;
    obs.push({
      t: spanHours(tReleaseMs, Date.parse(h.at)),
      n: h.players_tracked,
      rate: r,
    });
  }
  if (obs.length === 0) {
    // The case never saw this trophy: a wide band around a small rate.
    return {
      id: t.id,
      q: NORMAL_Z.map((z) => round4dp(clampPct(pctFromLogit(logitPct(0.5) + z * 1.5)))),
    };
  }
  const { lambda, fitResidual, medianPct } = fitHazard(obs, pop, tLast, tTarget);
  const sigma = trophySpread(
    lambda,
    fitResidual,
    medianPct,
    pop,
    horizonHours,
    horizonDays,
    histHours,
  );
  const l = logitPct(medianPct);
  return {
    id: t.id,
    q: NORMAL_Z.map((z) => round4dp(clampPct(pctFromLogit(l + z * sigma)))),
  };
}

/** Forecast every trophy of one case; returns the rows for its .out file. */
function forecastOne(c: CaseInput): Array<{ id: number; q: number[] }> {
  const releaseMs = Date.parse(`${c.game.release_date}T00:00:00Z`);
  const tReleaseMs = Number.isFinite(releaseMs)
    ? releaseMs
    : Date.parse(c.history[0].at);
  const tLastMs = Date.parse(c.history[c.history.length - 1].at);
  const tTargetMs = Date.parse(c.target);
  const tLast = spanHours(tReleaseMs, tLastMs);
  const tTarget = spanHours(tReleaseMs, tTargetMs);
  const horizonHours = Math.max(0, spanHours(tLastMs, tTargetMs));
  const horizonDays = horizonHours / 24;
  const histHours = Math.max(
    0.5,
    spanHours(tReleaseMs, tLastMs) -
      spanHours(tReleaseMs, Date.parse(c.history[0].at)),
  );

  const pop = fitPlayers(c, tReleaseMs, tLastMs, tTargetMs);
  return c.trophies.map((t, i) =>
    forecastTrophy(t, i, c, pop, tReleaseMs, tLast, tTarget, horizonHours, horizonDays, histHours),
  );
}

/** Read every case, forecast it, write its .out file. */
function main(): void {
  mkdirSync(OUTPUT_DIR, { recursive: true });
  // Every case file, in name order, so two runs write the same bytes.
  const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
  for (const file of inputs) {
    const id = file.slice(0, -'.in'.length);
    const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseInput;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    if (c.history.length === 0) {
      // No reads at all: a neutral, wide forecast rather than a crash.
      const trophies = c.trophies.map((t) => ({
        id: t.id,
        q: NORMAL_Z.map((z) => round4dp(pctFromLogit(logitPct(1) + z * 1.5))),
      }));
      writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
      continue;
    }
    const trophies = forecastOne(c);
    writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
  }
}

main();
