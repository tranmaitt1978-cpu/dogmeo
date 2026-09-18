/**
 * hoang.js — Công Nghệ Vip HHOANG 2026
 * Omega Bridge Engine v4 — Tích hợp vào server
 * Developer: HuyHoang
 *
 * Nguồn API: https://sunwin-taixiu-dulieu.onrender.com/data
 *
 * Engine: OMEGA BRIDGE v4
 *  - Multi-pattern (Cầu bệt, 1-1, 2-2, 3-3, 4-4, 5-5)
 *  - Run shape, Staircase, Mirror, Cycle
 *  - Markov 1..6, N-gram, Similarity, Recency
 *  - Momentum, Transition matrix
 *  - Regime detection, Entropy, Conflict detection
 *  - Adaptive evidence weighting
 *  - NO_SIGNAL khi bất định
 *  - KHÔNG Martingale, KHÔNG loss chasing, KHÔNG đảo khi thua
 */

'use strict';

const express = require('express');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;
const API_URL = process.env.API_URL || 'https://sunwin-taixiu-dulieu.onrender.com/data';

const HISTORY_LIMIT = 3000;
const FETCH_INTERVAL_MS = 22000;
const FETCH_TIMEOUT_MS = 15000;
const PREDICTION_TTL_MS = 6 * 60 * 1000;
const LOG_LIMIT = 150;

/* ================================================================
   OMEGA CONFIG
   ================================================================ */

const OMEGA_CONFIG = {
    minHistory: 120,
    maxOrder: 6,
    windows: [8, 12, 16, 20, 30, 50, 80, 120, 250, 500],
    similarityLengths: [5, 6, 7, 8, 10, 12],
    minSupport: 12,
    strongSupport: 30,
    alpha: 1,
    signalThreshold: 0.56,
    strongThreshold: 0.68,
    probabilityFloor: 0.05,
    probabilityCeil: 0.95,
    similarityLimit: 4000,
    conflictPenalty: 0.65,
    noisyPenalty: 0.55,
    probabilityCompression: 0.82
};

/* ================================================================
   OMEGA MATH
   ================================================================ */

const OmegaMath = {
    clamp(v, min, max) { return Math.max(min, Math.min(max, v)); },
    safeNumber(v, fb = 0) { const n = Number(v); return Number.isFinite(n) ? n : fb; },
    probability(t, x, alpha = 1) {
        const total = t + x;
        if (total <= 0) return 0.5;
        return (t + alpha) / (total + alpha * 2);
    },
    edge(p) { return Math.abs(p - 0.5) * 2; },
    entropy(p) {
        p = this.clamp(p, 1e-12, 1 - 1e-12);
        return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
    },
    direction(p) { return p >= 0.5 ? "T" : "X"; },
    opposite(side) { return side === "T" ? "X" : "T"; },
    weightedAverage(items) {
        let num = 0, den = 0;
        for (const item of items) {
            if (!Number.isFinite(item.value)) continue;
            if (!Number.isFinite(item.weight) || item.weight <= 0) continue;
            num += item.value * item.weight;
            den += item.weight;
        }
        return den <= 0 ? 0.5 : num / den;
    },
    average(values) {
        if (!values.length) return 0;
        return values.reduce((a, b) => a + b, 0) / values.length;
    },
    sequenceKey(seq) { return seq.join(""); }
};

/* ================================================================
   NORMALIZER — chuyển dữ liệu API sang side T/X
   ================================================================ */

function normalizeRecord(r) {
    if (!r || typeof r !== "object") return null;

    const d1 = OmegaMath.safeNumber(r.xuc_xac_1);
    const d2 = OmegaMath.safeNumber(r.xuc_xac_2);
    const d3 = OmegaMath.safeNumber(r.xuc_xac_3);
    const total = OmegaMath.safeNumber(r.tong, d1 + d2 + d3);
    const phien = OmegaMath.safeNumber(r.phien, NaN);

    if (!Number.isFinite(phien) || phien <= 0) return null;
    if (d1 < 1 || d1 > 6 || d2 < 1 || d2 > 6 || d3 < 1 || d3 > 6) return null;

    const rawResult = String(r.ket_qua || "").trim().toUpperCase();
    let side;
    if (rawResult === "TÀI" || rawResult === "TAI" || rawResult === "T") side = "T";
    else if (rawResult === "XỈU" || rawResult === "XIU" || rawResult === "X") side = "X";
    else side = total >= 11 ? "T" : "X";

    return {
        phien, d1, d2, d3, total, side,
        time: typeof r.thoi_gian === "string" ? r.thoi_gian : vnNow()
    };
}

function normalizeRecords(records) {
    if (!Array.isArray(records)) return [];
    const result = [];
    const seen = new Set();
    for (const r of records) {
        const rec = normalizeRecord(r);
        if (!rec) continue;
        if (seen.has(rec.phien)) continue;
        seen.add(rec.phien);
        result.push(rec);
    }
    result.sort((a, b) => a.phien - b.phien);
    return result;
}

/* ================================================================
   SEQUENCE
   ================================================================ */

function encodeRuns(seq) {
    if (!seq.length) return [];
    const runs = [];
    let side = seq[0], length = 1;
    for (let i = 1; i < seq.length; i++) {
        if (seq[i] === side) length++;
        else { runs.push({ side, length }); side = seq[i]; length = 1; }
    }
    runs.push({ side, length });
    return runs;
}

/* ================================================================
   PATTERN ANALYSIS
   ================================================================ */

function analyzeEqualBlock(seq) {
    const runs = encodeRuns(seq);
    if (runs.length < 2) return { name: "EQUAL_BLOCK", probability: 0.5, strength: 0, support: 0 };
    const current = runs[runs.length - 1];
    const candidates = [];
    for (let i = 0; i < runs.length - 1; i++) {
        if (runs[i].length === current.length) {
            candidates.push(runs[i + 1] ? runs[i + 1].side : null);
        }
    }
    const valid = candidates.filter(Boolean);
    let t = 0, x = 0;
    for (const side of valid) { if (side === "T") t++; else x++; }
    const p = OmegaMath.probability(t, x, OMEGA_CONFIG.alpha);
    return {
        name: "EQUAL_BLOCK_" + current.length,
        probability: p,
        strength: valid.length >= OMEGA_CONFIG.minSupport ? OmegaMath.edge(p) : 0,
        support: valid.length,
        category: "BET"
    };
}

function analyzeAlternating(seq, window) {
    const data = seq.slice(-window);
    if (data.length < 4) return { name: "ALT_" + window, probability: 0.5, strength: 0, support: 0 };
    let alternations = 0;
    for (let i = 1; i < data.length; i++) {
        if (data[i] !== data[i - 1]) alternations++;
    }
    const ratio = alternations / (data.length - 1);
    const last = data[data.length - 1];
    const predicted = OmegaMath.opposite(last);
    const edge = ratio * 0.42;
    const p = predicted === "T" ? 0.5 + edge : 0.5 - edge;
    return {
        name: "ALT_" + window,
        probability: OmegaMath.clamp(p, 0.05, 0.95),
        strength: ratio * 0.90,
        support: data.length - 1,
        category: "ALTERNATION"
    };
}

function analyzeBlock(seq, blockSize) {
    const runs = encodeRuns(seq);
    if (runs.length < 4) return { name: "BLOCK_" + blockSize, probability: 0.5, strength: 0, support: 0 };
    let t = 0, x = 0;
    for (let i = 0; i < runs.length - 1; i++) {
        if (runs[i].length !== blockSize) continue;
        if (runs[i + 1].side === "T") t++; else x++;
    }
    const p = OmegaMath.probability(t, x, OMEGA_CONFIG.alpha);
    return {
        name: "BLOCK_" + blockSize,
        probability: p,
        strength: (t + x) >= OMEGA_CONFIG.minSupport ? OmegaMath.edge(p) : 0,
        support: t + x,
        category: "BLOCK"
    };
}

function analyzeRunContinuation(seq) {
    const runs = encodeRuns(seq);
    const current = runs[runs.length - 1];
    if (!current) return { name: "RUN_CONTINUATION", probability: 0.5, strength: 0, support: 0 };
    let t = 0, x = 0;
    for (let i = 0; i < runs.length - 1; i++) {
        const r = runs[i];
        if (r.side === current.side && r.length === current.length) {
            if (runs[i + 1].side === "T") t++; else x++;
        }
    }
    const support = t + x;
    const p = OmegaMath.probability(t, x, OMEGA_CONFIG.alpha);
    return {
        name: "RUN_CONTINUATION",
        probability: p,
        strength: support >= OMEGA_CONFIG.minSupport ? OmegaMath.edge(p) : 0,
        support,
        category: "RUN"
    };
}

function analyzeRunShape(seq) {
    const runs = encodeRuns(seq);
    const shapes = [
        [1, 2, 1], [2, 1, 2], [1, 2, 2, 1], [2, 1, 1, 2],
        [1, 3, 1], [3, 1, 3], [2, 2, 1, 2], [2, 1, 2, 2]
    ];
    const signals = [];
    for (const shape of shapes) {
        if (runs.length < shape.length) continue;
        const currentShape = runs.slice(-shape.length).map(r => r.length);
        if (OmegaMath.sequenceKey(currentShape) !== OmegaMath.sequenceKey(shape)) continue;
        let t = 0, x = 0;
        for (let i = shape.length; i < runs.length; i++) {
            const candidate = runs.slice(i - shape.length, i).map(r => r.length);
            if (OmegaMath.sequenceKey(candidate) !== OmegaMath.sequenceKey(shape)) continue;
            if (runs[i].side === "T") t++; else x++;
        }
        const p = OmegaMath.probability(t, x, OMEGA_CONFIG.alpha);
        signals.push({ probability: p, support: t + x });
    }
    if (!signals.length) return { name: "RUN_SHAPE", probability: 0.5, strength: 0, support: 0 };
    const usable = signals.filter(s => s.support > 0);
    if (!usable.length) return { name: "RUN_SHAPE", probability: 0.5, strength: 0, support: 0 };
    const p = OmegaMath.weightedAverage(usable.map(s => ({
        value: s.probability,
        weight: Math.log1p(s.support)
    })));
    return {
        name: "RUN_SHAPE",
        probability: p,
        strength: OmegaMath.edge(p),
        support: usable.reduce((sum, s) => sum + s.support, 0),
        category: "RUN_SHAPE"
    };
}

function analyzeStaircase(seq) {
    const runs = encodeRuns(seq);
    if (runs.length < 3) return { name: "STAIRCASE", probability: 0.5, strength: 0, support: 0 };
    const recent = runs.slice(-5);
    const lengths = recent.map(r => r.length);
    let increasing = 0, decreasing = 0;
    for (let i = 1; i < lengths.length; i++) {
        if (lengths[i] > lengths[i - 1]) increasing++;
        if (lengths[i] < lengths[i - 1]) decreasing++;
    }
    const comparisons = Math.max(1, lengths.length - 1);
    const incRatio = increasing / comparisons;
    const decRatio = decreasing / comparisons;
    const last = recent[recent.length - 1];
    let p = 0.5;
    if (incRatio >= 0.75) p = last.side === "T" ? 0.54 : 0.46;
    else if (decRatio >= 0.75) p = last.side === "T" ? 0.46 : 0.54;
    return {
        name: "STAIRCASE",
        probability: p,
        strength: Math.max(incRatio, decRatio) * 0.25,
        support: comparisons,
        category: "STAIRCASE"
    };
}

function analyzeMirror(seq) {
    const L = 4;
    if (seq.length < L * 2) return { name: "MIRROR", probability: 0.5, strength: 0, support: 0 };
    const left = seq.slice(-L * 2, -L);
    const right = seq.slice(-L);
    let matches = 0;
    for (let i = 0; i < L; i++) if (right[i] === left[L - 1 - i]) matches++;
    const ratio = matches / L;
    if (ratio < 0.75) return { name: "MIRROR", probability: 0.5, strength: 0, support: L };
    const predicted = left[0];
    const p = predicted === "T" ? 0.5 + ratio * 0.18 : 0.5 - ratio * 0.18;
    return {
        name: "MIRROR",
        probability: OmegaMath.clamp(p, 0.05, 0.95),
        strength: ratio * 0.25,
        support: L,
        category: "MIRROR"
    };
}

function analyzeCycle(seq) {
    let best = null;
    const maxPeriod = Math.min(16, Math.floor(seq.length / 4));
    for (let period = 2; period <= maxPeriod; period++) {
        let same = 0, total = 0;
        for (let i = period; i < seq.length; i++) {
            total++;
            if (seq[i] === seq[i - period]) same++;
        }
        if (total <= 0) continue;
        const ratio = same / total;
        if (!best || ratio > best.ratio) best = { period, ratio, support: total };
    }
    if (!best || best.ratio < 0.65) return { name: "CYCLE", probability: 0.5, strength: 0, support: 0 };
    const predicted = seq[seq.length - best.period];
    const p = predicted === "T"
        ? 0.5 + (best.ratio - 0.5) * 0.45
        : 0.5 - (best.ratio - 0.5) * 0.45;
    return {
        name: "CYCLE",
        probability: OmegaMath.clamp(p, 0.05, 0.95),
        strength: (best.ratio - 0.5) * 1.4,
        support: best.support,
        category: "CYCLE"
    };
}

function analyzeMarkov(seq, order) {
    if (seq.length <= order) return { name: "MARKOV_" + order, probability: 0.5, strength: 0, support: 0 };
    const context = seq.slice(-order);
    let t = 0, x = 0;
    for (let i = order; i < seq.length; i++) {
        const previous = seq.slice(i - order, i);
        if (OmegaMath.sequenceKey(previous) !== OmegaMath.sequenceKey(context)) continue;
        if (seq[i] === "T") t++; else x++;
    }
    const support = t + x;
    const p = OmegaMath.probability(t, x, OMEGA_CONFIG.alpha);
    const usable = support >= OMEGA_CONFIG.minSupport;
    return {
        name: "MARKOV_" + order,
        probability: usable ? p : 0.5,
        strength: usable ? OmegaMath.edge(p) : 0,
        support,
        category: "MARKOV"
    };
}

function analyzeSimilarity(seq, length) {
    if (seq.length <= length + 1) return { name: "SIM_" + length, probability: 0.5, strength: 0, support: 0 };
    const target = seq.slice(-length);
    const start = Math.max(length, seq.length - OMEGA_CONFIG.similarityLimit);
    let weightedT = 0, weightedX = 0, matches = 0;
    for (let i = start; i < seq.length; i++) {
        const candidate = seq.slice(i - length, i);
        if (candidate.length !== length) continue;
        const actual = seq[i];
        let d = 0;
        for (let j = 0; j < length; j++) if (target[j] !== candidate[j]) d++;
        const weight = Math.exp(-0.70 * d);
        if (weight < 0.015) continue;
        if (actual === "T") weightedT += weight; else weightedX += weight;
        matches++;
    }
    const total = weightedT + weightedX;
    if (total <= 0) return { name: "SIM_" + length, probability: 0.5, strength: 0, support: 0 };
    const p = weightedT / total;
    return {
        name: "SIM_" + length,
        probability: OmegaMath.clamp(p, 0.05, 0.95),
        strength: OmegaMath.edge(p),
        support: matches,
        category: "SIMILARITY"
    };
}

function analyzeRecency(seq) {
    const windows = [];
    for (const size of OMEGA_CONFIG.windows) {
        if (seq.length < size) continue;
        const data = seq.slice(-size);
        const t = data.filter(x => x === "T").length;
        const p = OmegaMath.probability(t, size - t, OMEGA_CONFIG.alpha);
        const weight = 1 / Math.sqrt(size);
        windows.push({ probability: p, support: size, weight });
    }
    if (!windows.length) return { name: "RECENCY", probability: 0.5, strength: 0, support: 0 };
    const p = OmegaMath.weightedAverage(windows.map(w => ({ value: w.probability, weight: w.weight })));
    return {
        name: "RECENCY",
        probability: p,
        strength: OmegaMath.edge(p),
        support: windows.reduce((sum, w) => sum + w.support, 0),
        category: "RECENCY"
    };
}

function analyzeMomentum(seq) {
    if (seq.length < 30) return { name: "MOMENTUM", probability: 0.5, strength: 0, support: 0 };
    const short = seq.slice(-10), medium = seq.slice(-30), long = seq.slice(-100);
    const ratio = arr => arr.filter(x => x === "T").length / arr.length;
    const pShort = ratio(short), pMedium = ratio(medium), pLong = ratio(long);
    const delta = (pShort * 0.55 + pMedium * 0.30 + pLong * 0.15) - 0.5;
    const p = OmegaMath.clamp(0.5 + delta * 0.75, 0.05, 0.95);
    return {
        name: "MOMENTUM",
        probability: p,
        strength: Math.abs(delta) * 1.5,
        support: short.length + medium.length + long.length,
        category: "MOMENTUM"
    };
}

function analyzeTransition(seq) {
    let TT = 0, TX = 0, XT = 0, XX = 0;
    for (let i = 1; i < seq.length; i++) {
        const a = seq[i - 1], b = seq[i];
        if (a === "T" && b === "T") TT++;
        if (a === "T" && b === "X") TX++;
        if (a === "X" && b === "T") XT++;
        if (a === "X" && b === "X") XX++;
    }
    const current = seq[seq.length - 1];
    let p, support;
    if (current === "T") { p = OmegaMath.probability(TT, TX); support = TT + TX; }
    else { p = OmegaMath.probability(XT, XX); support = XT + XX; }
    return {
        name: "TRANSITION",
        probability: p,
        strength: OmegaMath.edge(p),
        support,
        category: "TRANSITION"
    };
}

function analyzeRegime(seq) {
    const data = seq.slice(-60);
    if (data.length < 20) return { type: "UNKNOWN", confidence: 0 };
    const runs = encodeRuns(data);
    const avgRun = OmegaMath.average(runs.map(r => r.length));
    let transitions = 0;
    for (let i = 1; i < data.length; i++) if (data[i] !== data[i - 1]) transitions++;
    const alternation = transitions / (data.length - 1);
    const t = data.filter(x => x === "T").length;
    const bias = Math.abs(t / data.length - 0.5);
    let type = "BALANCED";
    if (alternation >= 0.82) type = "STRONG_ALTERNATING";
    else if (alternation >= 0.70) type = "ALTERNATING";
    else if (avgRun >= 4) type = "LONG_RUN";
    else if (avgRun >= 2.2) type = "SHORT_RUN";
    const H = OmegaMath.entropy(t / data.length);
    if (H >= 0.995 && alternation > 0.42 && alternation < 0.62) type = "NOISY";
    const confidence = OmegaMath.clamp(Math.max(alternation, 1 - alternation, bias * 2), 0, 1);
    return { type, confidence, averageRun: avgRun, alternation, bias, entropy: H, sample: data.length };
}

function analyzeEntropy(seq) {
    const windows = [20, 50, 100];
    const result = [];
    for (const size of windows) {
        if (seq.length < size) continue;
        const data = seq.slice(-size);
        const p = data.filter(x => x === "T").length / data.length;
        const H = OmegaMath.entropy(p);
        result.push({ entropy: H, predictability: 1 - H });
    }
    if (!result.length) return { entropy: 1, predictability: 0 };
    return {
        entropy: OmegaMath.average(result.map(x => x.entropy)),
        predictability: OmegaMath.average(result.map(x => x.predictability))
    };
}

/* ================================================================
   OMEGA PREDICTION
   ================================================================ */

function omegaPredict(history) {
    if (history.length < OMEGA_CONFIG.minHistory) {
        return {
            status: "INSUFFICIENT_DATA",
            side: null,
            probability: 0.5,
            confidence: 0,
            historySize: history.length,
            signals: [],
            regime: null
        };
    }

    const seq = history.map(x => x.side);
    const signals = [];

    const add = (signal, baseWeight) => {
        if (!signal) return;
        const support = Number(signal.support || 0);
        const supportFactor = OmegaMath.clamp(
            Math.log1p(support) / Math.log1p(OMEGA_CONFIG.strongSupport), 0, 1
        );
        const strength = OmegaMath.clamp(Number(signal.strength || 0), 0, 1);
        const evidence = 0.25 + 0.45 * supportFactor + 0.30 * strength;
        signals.push({ ...signal, effectiveWeight: baseWeight * evidence });
    };

    add(analyzeEqualBlock(seq), 1.15);
    for (const w of [12, 20, 30, 50]) add(analyzeAlternating(seq, w), 0.90);
    for (const bs of [2, 3, 4, 5]) add(analyzeBlock(seq, bs), 0.82);
    add(analyzeRunShape(seq), 1.00);
    add(analyzeStaircase(seq), 0.55);
    add(analyzeMirror(seq), 0.45);
    add(analyzeCycle(seq), 0.65);
    add(analyzeRunContinuation(seq), 0.90);
    for (let order = 1; order <= OMEGA_CONFIG.maxOrder; order++) {
        add(analyzeMarkov(seq, order), 1.00 - (order - 1) * 0.09);
    }
    for (const length of OMEGA_CONFIG.similarityLengths) {
        add(analyzeSimilarity(seq, length), 0.95);
    }
    add(analyzeRecency(seq), 0.65);
    add(analyzeMomentum(seq), 0.48);
    add(analyzeTransition(seq), 0.70);

    const regime = analyzeRegime(seq);
    const entropy = analyzeEntropy(seq);

    let sum = 0, weight = 0;
    for (const s of signals) {
        let w = s.effectiveWeight;
        const edge = OmegaMath.edge(s.probability);
        w *= 0.45 + 0.55 * edge;
        sum += s.probability * w;
        weight += w;
    }
    let p = weight > 0 ? sum / weight : 0.5;

    let taiWeight = 0, xiuWeight = 0;
    for (const s of signals) {
        if (s.effectiveWeight <= 0) continue;
        if (s.probability >= 0.5) taiWeight += s.effectiveWeight;
        else xiuWeight += s.effectiveWeight;
    }
    const totalDir = taiWeight + xiuWeight;
    const agreement = totalDir > 0 ? Math.max(taiWeight, xiuWeight) / totalDir : 0;
    const conflict = 1 - agreement;

    let regimeFactor = 1;
    switch (regime.type) {
        case "STRONG_ALTERNATING": regimeFactor = 0.94; break;
        case "ALTERNATING": regimeFactor = 0.90; break;
        case "LONG_RUN": regimeFactor = 0.88; break;
        case "SHORT_RUN": regimeFactor = 0.82; break;
        case "NOISY": regimeFactor = OMEGA_CONFIG.noisyPenalty; break;
        default: regimeFactor = 0.75;
    }

    p = 0.5 + (p - 0.5) * regimeFactor * OMEGA_CONFIG.probabilityCompression;

    const entropyFactor = OmegaMath.clamp(entropy.predictability * 0.75 + 0.25, 0.25, 1);
    const finalEdge = OmegaMath.edge(p);
    let confidence = finalEdge * (0.42 + 0.38 * agreement + 0.20 * entropyFactor);
    confidence *= 1 - conflict * OMEGA_CONFIG.conflictPenalty;
    confidence = OmegaMath.clamp(confidence, 0, 1);

    let status = "NO_SIGNAL";
    if (confidence >= OMEGA_CONFIG.strongThreshold) status = "STRONG_SIGNAL";
    else if (confidence >= OMEGA_CONFIG.signalThreshold) status = "SIGNAL";

    p = OmegaMath.clamp(p, OMEGA_CONFIG.probabilityFloor, OMEGA_CONFIG.probabilityCeil);

    const side = OmegaMath.direction(p) === "T" ? "TAI" : "XIU";

    const ranked = [...signals].sort((a, b) =>
        (b.effectiveWeight * b.strength) - (a.effectiveWeight * a.strength)
    );

    return {
        status,
        side,
        direction: OmegaMath.direction(p),
        probability: p,
        confidence,
        agreement,
        conflict,
        regime: regime.type,
        regimeConfidence: regime.confidence,
        historySize: history.length,
        signals: ranked.slice(0, 10).map(s => ({
            name: s.name,
            category: s.category,
            direction: OmegaMath.direction(s.probability) === "T" ? "TAI" : "XIU",
            probability: s.probability,
            strength: s.strength,
            support: s.support
        }))
    };
}

/* ================================================================
   TIME
   ================================================================ */

const VN_FMT = new Intl.DateTimeFormat('sv-SE', {
    timeZone: 'Asia/Ho_Chi_Minh',
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
});
function vnNow() {
    return VN_FMT.format(new Date()).replace('T', ' ');
}

/* ================================================================
   ENGINE STATE
   ================================================================ */

const stats = {
    total: 0,
    correct: 0,
    wrong: 0,
    noSignal_total: 0,
    start_time: vnNow()
};

let lastData = [];
let lastPrediction = null;
let predictionLog = [];
let isFetching = false;

/* ================================================================
   FETCH & PREDICT
   ================================================================ */

async function fetchAndAnalyze() {
    if (isFetching) return;
    isFetching = true;
    try {
        const res = await axios.get(API_URL, { timeout: FETCH_TIMEOUT_MS });
        const raw = res.data;
        if (!raw) { console.warn('[WARN] payload rỗng'); return; }

        let recordsRaw = [];
        if (Array.isArray(raw)) recordsRaw = raw;
        else if (Array.isArray(raw.data)) recordsRaw = raw.data;
        else if (Array.isArray(raw.history)) recordsRaw = raw.history;
        else if (Array.isArray(raw.results)) recordsRaw = raw.results;

        const normalized = normalizeRecords(recordsRaw);
        // Sort giảm dần theo phiên (mới nhất ở đầu) cho UI
        const dataDesc = [...normalized].sort((a, b) => b.phien - a.phien).slice(0, HISTORY_LIMIT);
        lastData = dataDesc;

        // Resolve prediction cũ
        if (lastPrediction) {
            const match = dataDesc.find(d => d.phien === lastPrediction.phienDuDoan);
            if (match) {
                const actual = match.side === "T" ? "TAI" : "XIU";
                const isCorrect = lastPrediction.side === actual;

                predictionLog.unshift({
                    phien: match.phien,
                    predict: lastPrediction.side,
                    actual,
                    confidence: lastPrediction.confidence,
                    tag: lastPrediction.tag,
                    correct: isCorrect,
                    fallback: lastPrediction.fallback,
                    time: match.time
                });
                if (predictionLog.length > LOG_LIMIT) predictionLog.pop();

                if (!lastPrediction.noSignal) {
                    stats.total++;
                    if (isCorrect) stats.correct++;
                    else stats.wrong++;
                } else {
                    stats.noSignal_total++;
                }

                console.log(`[RESOLVED] #${match.phien} | ${lastPrediction.side} → ${actual} | ${isCorrect ? 'ĐÚNG' : 'SAI'}`);
                lastPrediction = null;
            } else {
                const age = Date.now() - new Date(lastPrediction.iso).getTime();
                if (age > PREDICTION_TTL_MS) {
                    predictionLog.unshift({
                        phien: lastPrediction.phienDuDoan,
                        predict: lastPrediction.side,
                        actual: null,
                        confidence: lastPrediction.confidence,
                        tag: lastPrediction.tag,
                        correct: false,
                        miss: true,
                        time: vnNow()
                    });
                    if (predictionLog.length > LOG_LIMIT) predictionLog.pop();
                    lastPrediction = null;
                }
            }
        }

        // Prediction mới — dùng toàn bộ history ASC
        if (!lastPrediction && normalized.length >= OMEGA_CONFIG.minHistory) {
            const omega = omegaPredict(normalized);
            const nextPhien = dataDesc[0].phien + 1;

            const side = omega.side;
            const confPct = Math.round(omega.confidence * 100);

            const tagParts = [];
            if (omega.status === "NO_SIGNAL") tagParts.push("NO_SIGNAL");
            else if (omega.status === "STRONG_SIGNAL") tagParts.push("STRONG");
            else tagParts.push("SIGNAL");
            if (omega.regime) tagParts.push(omega.regime);

            const topSig = omega.signals.slice(0, 3)
                .map(s => `${s.name}:${s.direction}(${(s.probability * 100).toFixed(0)}%)`)
                .join(" · ");

            lastPrediction = {
                phienDuDoan: nextPhien,
                side: side || (omega.direction === "T" ? "TAI" : "XIU"),
                confidence: Math.max(1, Math.min(99, confPct)),
                tag: tagParts.join(" · "),
                info: `P(TAI)=${(omega.probability * 100).toFixed(1)}% · ${topSig || 'no strong pattern'}`,
                fallback: omega.status === "NO_SIGNAL",
                noSignal: omega.status === "NO_SIGNAL",
                timestamp: vnNow(),
                iso: new Date().toISOString(),
                raw: omega
            };

            console.log(`[PREDICT] #${nextPhien} → ${lastPrediction.side} (${lastPrediction.confidence}%) | ${lastPrediction.tag}`);
        }
    } catch (err) {
        console.error('[FETCH ERROR]', err.message);
    } finally {
        isFetching = false;
    }
}

process.on('unhandledRejection', r => console.error('[UNHANDLED]', r));
process.on('uncaughtException', e => console.error('[
