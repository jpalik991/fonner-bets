/* fonner.js -- Fonner Park model v4, live engine.
 *
 * Everything the phone page needs, in one file, with no outside libraries:
 *   1. read a Brisnet DRF file            (port of load_db.py v2.2, DRF part)
 *   2. build each runner's inputs          (port of step5_features.py v1.1)
 *   3. add the race-level inputs           (port of step15 build_race_features)
 *   4. run the Layer 1 trees               (the five Step 15 fold models)
 *   5. build the Plan A tickets            (port of build40 in step40, 1.1x + flip)
 *   6. read chart files to score a day     (port of load_db.py chart parts)
 *
 * Works in a browser (window.Fonner) and in Node (require) so the same code
 * that runs at the track is the code that was checked against Python.
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------ helpers
  function parseCSV(text) {
    // Python csv.reader, default dialect. Blank rows dropped like read_csv().
    const rows = [];
    let row = [], field = '', inQ = false, atStart = true;
    const n = text.length;
    for (let i = 0; i < n; i++) {
      const c = text[i];
      if (inQ) {
        if (c === '"') {
          if (text[i + 1] === '"') { field += '"'; i++; } else { inQ = false; }
        } else field += c;
        continue;
      }
      if (c === '"' && atStart) { inQ = true; atStart = false; continue; }
      if (c === ',') { row.push(field); field = ''; atStart = true; continue; }
      if (c === '\r' || c === '\n') {
        if (c === '\r' && text[i + 1] === '\n') i++;
        row.push(field); field = ''; atStart = true;
        if (row.some(v => v.trim())) rows.push(row);
        row = [];
        continue;
      }
      field += c; atStart = false;
    }
    if (field !== '' || row.length) {
      row.push(field);
      if (row.some(v => v.trim())) rows.push(row);
    }
    return rows;
  }

  const NUM_RE = /^[+-]?(\d[\d_]*\.?[\d_]*|\.\d[\d_]*)([eE][+-]?\d+)?$/;
  function get(row, idx) { return (idx > 0 && idx <= row.length) ? row[idx - 1].trim() : ''; }
  function text(row, idx) { const v = get(row, idx); return v ? v : null; }
  function num(row, idx) {
    const v = get(row, idx);
    if (!v || !NUM_RE.test(v)) return null;
    const x = Number(v.replace(/_/g, ''));
    return Number.isFinite(x) ? x : null;
  }
  function whole(row, idx) { const v = num(row, idx); return v === null ? null : Math.trunc(v); }
  function normProgram(v) { return v.trim().toUpperCase().replace(/ /g, ''); }
  function normName(v) {
    return v.toUpperCase().replace(/\((?:[A-Z]{2,3})\)/g, '').replace(/[^A-Z0-9]/g, '');
  }
  function normRunStyle(v) {
    v = v.trim().toUpperCase().replace(/ /g, '');
    if (!v) return null;
    return (v === 'E/P' || v === 'EP') ? 'EP' : v;
  }
  function parseDate(v) {
    v = v.trim();
    if (!/^\d{8}$/.test(v)) return null;
    const y = +v.slice(0, 4), m = +v.slice(4, 6), d = +v.slice(6, 8);
    const t = new Date(Date.UTC(y, m - 1, d));
    if (t.getUTCFullYear() !== y || t.getUTCMonth() !== m - 1 || t.getUTCDate() !== d) return null;
    return v.slice(0, 4) + '-' + v.slice(4, 6) + '-' + v.slice(6, 8);
  }
  function dayNum(iso) { return Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10)) / 86400000; }
  function daysBetween(later, earlier) { return (later && earlier) ? dayNum(later) - dayNum(earlier) : null; }
  const isNum = x => typeof x === 'number' && !Number.isNaN(x);
  const nz = x => (x === null || x === undefined) ? NaN : x;       // null -> NaN

  // numpy's pairwise sum (what ndarray.sum() does), so ticket maths match bit for bit
  function npsum(a) {
    const n = a.length;
    if (n < 8) { let s = 0; for (let i = 0; i < n; i++) s += a[i]; return s; }
    if (n <= 128) {
      const r = a.slice(0, 8);
      let i = 8;
      for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
      let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
      for (; i < n; i++) res += a[i];
      return res;
    }
    let n2 = Math.floor(n / 2); n2 -= n2 % 8;
    return npsum(a.slice(0, n2)) + npsum(a.slice(n2));
  }
  function roundHalfEven(x) {
    const f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return (f % 2 === 0) ? f : f + 1;
  }

  // ---- pandas-exact arithmetic ------------------------------------------------
  // The model was trained on values that went through pandas: group means and sums
  // use Kahan summation, group std uses Welford, and every input was written to
  // features.csv and read back with pandas' own number parser (which drops digits
  // past the 17th). Trees split between float-noise variants of the same number,
  // so the live inputs must go through exactly the same arithmetic.
  function kahanSum(a) {
    let sum = 0, comp = 0, n = 0;
    for (const v of a) {
      if (!isNum(v)) continue;
      n++;
      const y = v - comp, t = sum + y;
      comp = t - sum - y;
      if (comp !== comp) comp = 0;
      sum = t;
    }
    return { sum, n };
  }
  function pdMean(a) { const k = kahanSum(a); return k.n ? k.sum / k.n : NaN; }
  function pdSum(a) { return kahanSum(a).sum; }
  function pdStd(a) {
    let n = 0, mean = 0, m2 = 0;
    for (const v of a) {
      if (!isNum(v)) continue;
      n++;
      const old = mean;
      mean += (v - old) / n;
      m2 += (v - mean) * (v - old);
    }
    return n <= 1 ? NaN : Math.sqrt(m2 / (n - 1));
  }
  // Python repr() of a float, then pandas read_csv's precise_xstrtod on that text
  function pyRepr(x) {
    if (Number.isNaN(x)) return 'nan';
    if (x === Infinity) return 'inf';
    if (x === -Infinity) return '-inf';
    if (x === 0) return (1 / x < 0) ? '-0.0' : '0.0';
    const e = x.toExponential();                 // shortest round-trip digits
    const m = /^(-?)(\d)(?:\.(\d+))?e([+-]\d+)$/.exec(e);
    const sign = m[1], digits = m[2] + (m[3] || ''), exp = +m[4];
    if (exp < -4 || exp >= 16) {
      const mant = digits.length > 1 ? digits[0] + '.' + digits.slice(1) : digits;
      return sign + mant + 'e' + (exp < 0 ? '-' : '+') + String(Math.abs(exp)).padStart(2, '0');
    }
    if (exp < 0) return sign + '0.' + '0'.repeat(-exp - 1) + digits;
    if (digits.length <= exp + 1) return sign + digits + '0'.repeat(exp + 1 - digits.length) + '.0';
    return sign + digits.slice(0, exp + 1) + '.' + digits.slice(exp + 1);
  }
  const POW10 = []; for (let i = 0; i <= 308; i++) POW10.push(Number('1e' + i));
  function pdParse(str) {
    let p = 0; const s = str.trim();
    if (s === 'nan') return NaN;
    if (s === 'inf') return Infinity;
    if (s === '-inf') return -Infinity;
    let neg = false;
    if (s[p] === '-') { neg = true; p++; } else if (s[p] === '+') p++;
    let number = 0, exponent = 0, nd = 0, ndec = 0;
    const dig = c => c >= '0' && c <= '9';
    while (p < s.length && dig(s[p])) {
      if (nd < 17) { number = number * 10 + (s.charCodeAt(p) - 48); nd++; } else exponent++;
      p++;
    }
    if (s[p] === '.') {
      p++;
      while (nd < 17 && p < s.length && dig(s[p])) { number = number * 10 + (s.charCodeAt(p) - 48); p++; nd++; ndec++; }
      if (nd >= 17) while (p < s.length && dig(s[p])) p++;
      exponent -= ndec;
    }
    if (neg) number = -number;
    if (s[p] === 'e' || s[p] === 'E') { p++; exponent += parseInt(s.slice(p), 10); }
    if (exponent > 308) number = number === 0 ? 0 : number < 0 ? -Infinity : Infinity;
    else if (exponent > 0) number *= POW10[exponent];
    else if (exponent < -308) { if (exponent < -616) number = 0; else { number /= POW10[-308 - exponent]; number /= POW10[308]; } }
    else number /= POW10[-exponent];
    return number;
  }
  function csvTrip(x) { return (typeof x === 'number' && Number.isFinite(x)) ? pdParse(pyRepr(x)) : x; }

  // ------------------------------------------------------------ 1. DRF
  const D = {
    TRACK: 1, DATE: 2, RACE: 3, POST: 4, DIST: 6, SURFACE: 7, RACE_TYPE: 9, AGE_SEX: 10, CLASS: 11, PURSE: 12, BREED: 23,
    TRAINER: 28, TRN: 29, JOCKEY: 33, JKY: 35, PROGRAM: 43, ML: 44, HORSE: 45,
    FOAL_YY: 46, SEX: 49, WEIGHT: 51,
    REC_DIST: 65, REC_TRACK: 70, REC_TURF: 75, REC_OFF: 80, REC_AW: 231, REC_FD: 1332,
    LIFE: 97,
    WO_DATE: 102, WO_TIME: 114, WO_OF: 186, WO_RANK: 198,
    RUN_STYLE: 210, QUIRIN: 211, PAR_E1: 214, PAR_E2: 215, PAR_SPEED: 217, PAR_LATE: 218,
    JT: 219, JT_ROI: 223, DAYS_SINCE: 224, PRIME: 251,
    PP_DATE: 256, PP_TRACK: 276, PP_DIST: 316, PP_FIELD: 346, PP_PURSE: 556,
    PP_FINISH: 626, PP_BEATEN: 746, PACE_2F: 766, PACE_4F: 776, LATE: 816, SPEED: 846,
    TRN_RECENT: 1147, TRN_RECENT_ROI: 1151, ANGLE: 1337,
    CLAIM: 13, YTD: 85, PRIOR: 91, POST_TIME: 1374,
  };

  function parseDRF(textIn, fallbackDate) {
    const rows = parseCSV(textIn);
    const races = new Map();
    let raceDate = null, track = null;
    for (const row of rows) {
      const raceNo = whole(row, D.RACE);
      if (raceNo === null) continue;
      if (!raceDate) raceDate = parseDate(get(row, D.DATE)) || fallbackDate || null;
      if (!track) track = text(row, D.TRACK);
      if (!raceDate) throw new Error('Could not read the race date from the file.');
      const raceId = raceDate + '-' + raceNo;
      if (!races.has(raceId)) {
        races.set(raceId, {
          race_id: raceId, race_no: raceNo, race_date: raceDate, track: text(row, D.TRACK),
          surface: text(row, D.SURFACE), distance_yards: num(row, D.DIST), age_sex: text(row, D.AGE_SEX), breed: text(row, D.BREED),
          race_type_code: text(row, D.RACE_TYPE), drf_class: text(row, D.CLASS),
          purse: num(row, D.PURSE), claim: num(row, D.CLAIM),
          post_time: (/\((\d{1,2}:\d{2})\)/.exec(get(row, D.POST_TIME)) || [])[1] || null,
          par_e1: num(row, D.PAR_E1), par_e2: num(row, D.PAR_E2),
          par_late: num(row, D.PAR_LATE), par_speed: num(row, D.PAR_SPEED),
          horses: [],
        });
      }
      const race = races.get(raceId);
      const program = normProgram(get(row, D.PROGRAM));
      const horse = get(row, D.HORSE);
      const foal = whole(row, D.FOAL_YY);
      let age = null;
      if (foal !== null) age = +raceDate.slice(0, 4) - (foal < 50 ? 2000 + foal : 1900 + foal);

      const lines = [];
      for (let i = 0; i < 10; i++) {
        const d = parseDate(get(row, D.PP_DATE + i));
        if (!d) continue;
        const rawFinish = whole(row, D.PP_FINISH + i);
        const dnf = rawFinish === 99;
        const finish = dnf ? null : rawFinish;
        const rawBeaten = num(row, D.PP_BEATEN + i);
        lines.push({
          date: d, track: text(row, D.PP_TRACK + i), distance: num(row, D.PP_DIST + i),
          purse: num(row, D.PP_PURSE + i), field_size: whole(row, D.PP_FIELD + i),
          finish, dnf: dnf ? 1 : 0, beaten: finish === 1 ? 0.0 : rawBeaten,
          pace_2f: num(row, D.PACE_2F + i), pace_4f: num(row, D.PACE_4F + i),
          late: num(row, D.LATE + i), speed: num(row, D.SPEED + i),
        });
      }
      const works = [];
      for (let i = 0; i < 12; i++) {
        const d = parseDate(get(row, D.WO_DATE + i));
        if (!d) continue;
        const t = num(row, D.WO_TIME + i);
        works.push({ date: d, bullet: (t !== null && t < 0) ? 1 : 0,
                     rank: whole(row, D.WO_RANK + i), of: whole(row, D.WO_OF + i) });
      }
      const angles = [];
      for (let g = 0; g < 6; g++) {
        const b = D.ANGLE + g * 5;
        const name = get(row, b);
        if (!name) continue;
        angles.push({ angle: name, starts: num(row, b + 1), win_pct: num(row, b + 2), roi: num(row, b + 4) });
      }
      const rec = b => ({ starts: whole(row, b), wins: whole(row, b + 1), places: whole(row, b + 2),
                          shows: whole(row, b + 3), earnings: num(row, b + 4) });
      const yrec = b => Object.assign({ year: whole(row, b) }, rec(b + 1));
      const lifeStarts = whole(row, D.LIFE);
      race.horses.push({
        entry_id: raceId + '-' + (program || normName(horse).slice(0, 10)),
        program, post: whole(row, D.POST), name: horse,
        trainer: text(row, D.TRAINER), jockey: text(row, D.JOCKEY),
        morning_line: num(row, D.ML), weight: num(row, D.WEIGHT), age,
        sex: text(row, D.SEX), prime_power: num(row, D.PRIME),
        run_style: normRunStyle(get(row, D.RUN_STYLE)), quirin: whole(row, D.QUIRIN),
        days_since: whole(row, D.DAYS_SINCE),
        life_starts: lifeStarts, life_wins: whole(row, D.LIFE + 1), life_earnings: num(row, D.LIFE + 4),
        life: rec(D.LIFE), ytd: yrec(D.YTD), prior: yrec(D.PRIOR),
        dist: rec(D.REC_DIST), trk: rec(D.REC_TRACK), fd: rec(D.REC_FD),
        off: rec(D.REC_OFF), turf: rec(D.REC_TURF), aw: rec(D.REC_AW),
        trn_meet: rec(D.TRN), jky_meet: rec(D.JKY),
        trn_recent: rec(D.TRN_RECENT), trn_recent_roi: num(row, D.TRN_RECENT_ROI),
        jt365: rec(D.JT), jt365_roi: num(row, D.JT_ROI),
        angles, lines, works,
        n_pp_lines: lines.length,
        first_time_starter: (lines.length === 0 && (lifeStarts || 0) === 0) ? 1 : 0,
      });
    }
    const list = [...races.values()].sort((a, b) => a.race_no - b.race_no);
    return { race_date: raceDate, track, races: list };
  }

  // ------------------------------------------------------------ 2. runner inputs
  const QH = new Set([550, 870, 1000]);
  const SHRINK = (w, s) => ((w === null ? 0 : w) + 1) / ((s === null ? 0 : s) + 8);
  function meanNN(a) { return pdMean(a); }
  function maxNN(a) { let m = -Infinity, c = 0; for (const x of a) if (isNum(x)) { if (x > m) m = x; c++; } return c ? m : NaN; }
  function minNN(a) { let m = Infinity, c = 0; for (const x of a) if (isNum(x)) { if (x < m) m = x; c++; } return c ? m : NaN; }
  function firstNN(lines, key) { for (const l of lines) { const v = l[key]; if (v !== null && v !== undefined && !Number.isNaN(v)) return v; } return null; }
  function pyRound(x) { return roundHalfEven(x); }

  const FLAGGED = ['prime_power', 'speed_last', 'speed_best3', 'pace_e1_last', 'pace_e2_last',
    'late_last', 'days_since', 'speed_best_dist', 'speed_best_fon', 'work_days_since', 'purse_last'];
  const RELATIVE = ['prime_power', 'speed_last', 'speed_best3', 'speed_avg3', 'pace_e1_avg3',
    'late_avg3', 'quirin', 'days_since', 'life_win_rate', 'class_move', 'trn_meet_win_rate',
    'jky_meet_win_rate'];

  function horseFeatures(h, race, meetStart) {
    const f = {};
    const today = race.distance_yards;
    f.field_size = NaN;                    // filled per race
    f.distance_yards = nz(today);
    f.is_sprint = (isNum(nz(today)) && today < 1760) ? 1 : 0;
    f.purse = nz(race.purse);
    f.par_e1 = nz(race.par_e1); f.par_e2 = nz(race.par_e2);
    f.par_late = nz(race.par_late); f.par_speed = nz(race.par_speed);
    f.days_into_meet = meetStart ? dayNum(race.race_date) - dayNum(meetStart) : NaN;
    f.month = +race.race_date.slice(5, 7);

    f.morning_line = nz(h.morning_line);
    f.prime_power = nz(h.prime_power);
    f.run_style = h.run_style;
    f.quirin = nz(h.quirin);
    f.post_drf = nz(h.post);
    f.weight = nz(h.weight);
    f.age = nz(h.age);
    f.sex = h.sex;
    f.days_since = nz(h.days_since);
    f.layoff_45 = isNum(f.days_since) ? (f.days_since > 45 ? 1 : 0) : NaN;
    f.layoff_180 = isNum(f.days_since) ? (f.days_since > 180 ? 1 : 0) : NaN;
    f.first_time_starter = h.first_time_starter;
    f.n_pp_lines = h.n_pp_lines;

    f.life_starts = nz(h.life_starts);
    f.life_win_rate = SHRINK(h.life_wins, h.life_starts);
    f.earn_per_start = ((h.life_starts || 0) > 0) ? nz(h.life_earnings) / h.life_starts : NaN;
    for (const p of ['dist', 'trk', 'fd']) {
      f[p + '_starts'] = nz(h[p].starts);
      f[p + '_win_rate'] = SHRINK(h[p].wins, h[p].starts);
    }
    f.off_starts = nz(h.off.starts);
    f.turf_starts = nz(h.turf.starts);
    f.aw_starts = nz(h.aw.starts);
    const dirt = (h.fd.starts || 0) + (h.off.starts || 0);
    f.pct_starts_dirt = ((h.life_starts || 0) > 0) ? dirt / h.life_starts : NaN;

    // form -- pandas groupby.first() takes the first NON-missing value per column
    const L = h.lines;
    const has = L.length > 0;
    const isQH = l => l.distance !== null && QH.has(pyRound(l.distance));
    const P = L.filter(l => !isQH(l));
    const t3 = L.slice(0, 3), p3 = P.slice(0, 3);
    const sp = a => a.map(l => nz(l.speed));
    f.speed_last = nz(firstNN(L, 'speed'));
    f.speed_best3 = maxNN(sp(t3));
    f.speed_avg3 = meanNN(sp(t3));
    f.speed_trend = f.speed_last - meanNN(sp(L.slice(1, 4)));
    const same = L.filter(l => {
      if (l.distance === null || !isNum(nz(today))) return false;
      const r = l.distance / today;
      return r >= 1 - 0.10 && r <= 1 + 0.10;
    });
    f.speed_best_dist = maxNN(sp(same));
    // "at this track" = the track in the file (FON at Fonner, so Fonner is unchanged)
    const home = race.track || 'FON';
    f.speed_best_fon = maxNN(sp(L.filter(l => l.track === home)));
    f.speed_vs_par = f.speed_best3 - f.par_speed;
    f.pace_e1_last = nz(firstNN(P, 'pace_2f'));
    f.pace_e1_avg3 = meanNN(p3.map(l => nz(l.pace_2f)));
    f.pace_e2_last = nz(firstNN(P, 'pace_4f'));
    f.late_last = nz(firstNN(P, 'late'));
    f.late_avg3 = meanNN(p3.map(l => nz(l.late)));
    f.pace_vs_par_e1 = f.pace_e1_avg3 - f.par_e1;
    f.finish_last = nz(firstNN(L, 'finish'));
    f.beaten_last = nz(firstNN(L, 'beaten'));
    f.won_last = isNum(f.finish_last) ? (f.finish_last === 1 ? 1 : 0) : NaN;
    f.itm_last3 = has ? t3.filter(l => l.finish !== null && l.finish <= 3).length : NaN;
    f.dnf_last3 = has ? t3.reduce((s, l) => s + l.dnf, 0) : NaN;
    f.purse_last = nz(firstNN(L, 'purse'));
    f.purse_avg3 = meanNN(t3.map(l => nz(l.purse)));
    const pa = f.purse_avg3 === 0 ? NaN : f.purse_avg3;
    f.class_move = Math.log(f.purse / pa);
    const lastTrk = firstNN(L, 'track');
    f.last_was_fon = lastTrk === null ? NaN : (lastTrk === home ? 1 : 0);
    f.last_dist_ratio = nz(firstNN(L, 'distance')) / f.distance_yards;
    f.field_size_last = nz(firstNN(L, 'field_size'));

    // workouts
    const W = h.works.map(w => Object.assign({ days: dayNum(race.race_date) - dayNum(w.date) }, w));
    const recent = W.filter(w => w.days >= 0 && w.days <= 60);
    f.work_days_since = minNN(W.map(w => w.days));
    f.work_n_60d = recent.length;
    f.work_best_pct = minNN(recent.map(w => (w.rank === null || !w.of) ? NaN : w.rank / w.of));
    f.work_bullet_60d = recent.length ? Math.max(...recent.map(w => w.bullet)) : 0;

    // connections
    f.trn_meet_starts = nz(h.trn_meet.starts);
    f.trn_meet_win_rate = SHRINK(h.trn_meet.wins, h.trn_meet.starts);
    f.trn_recent_starts = nz(h.trn_recent.starts);
    f.trn_recent_win_rate = SHRINK(h.trn_recent.wins, h.trn_recent.starts);
    f.trn_recent_roi = nz(h.trn_recent_roi);
    f.jky_meet_starts = nz(h.jky_meet.starts);
    f.jky_meet_win_rate = SHRINK(h.jky_meet.wins, h.jky_meet.starts);
    f.jt365_starts = nz(h.jt365.starts);
    f.jt365_win_rate = SHRINK(h.jt365.wins, h.jt365.starts);
    f.jt365_roi = nz(h.jt365_roi);
    f.n_angles = h.angles.length;
    const rois = h.angles.filter(a => a.roi !== null && (a.starts || 0) >= 20).map(a => a.roi);
    f.angle_best_roi = rois.length ? Math.max(...rois) : NaN;

    for (const c of FLAGGED) f[c + '_missing'] = isNum(f[c]) ? 0 : 1;
    return f;
  }

  // within-race helpers (pandas skips missing values)
  function rankDesc(vals) {
    return vals.map(v => {
      if (!isNum(v)) return NaN;
      let gt = 0, eq = 0;
      for (const w of vals) if (isNum(w)) { if (w > v) gt++; else if (w === v) eq++; }
      return gt + (eq + 1) / 2;
    });
  }
  function rankMinDesc(vals) {
    return vals.map(v => {
      if (!isNum(v)) return NaN;
      let gt = 0; for (const w of vals) if (isNum(w) && w > v) gt++;
      return gt + 1;
    });
  }
  function stdNN(vals) {
    const a = vals.filter(isNum);
    if (a.length < 2) return NaN;
    const m = a.reduce((s, x) => s + x, 0) / a.length;
    let ss = 0; for (const x of a) ss += (x - m) * (x - m);
    return Math.sqrt(ss / (a.length - 1));
  }
  function medianNN(vals) {
    const a = vals.filter(isNum).sort((x, y) => x - y);
    if (!a.length) return NaN;
    const k = a.length >> 1;
    return a.length % 2 ? a[k] : (a[k - 1] + a[k]) / 2;
  }
  function topTwoGap(vals) {
    const a = vals.filter(isNum).sort((x, y) => x - y);
    return a.length >= 2 ? a[a.length - 1] - a[a.length - 2] : NaN;
  }
  function ownMinusBestOther(vals) {
    if (vals.length < 2 || !vals.some(isNum)) return vals.map(() => NaN);
    const v2 = vals.map(v => isNum(v) ? v : -Infinity);
    const s = [...v2].sort((x, y) => x - y);
    const top = s[s.length - 1], sec = s[s.length - 2];
    const T = top === -Infinity ? NaN : top, S = sec === -Infinity ? NaN : sec;
    return vals.map(v => isNum(v) ? v - (v >= T ? S : T) : NaN);
  }

  // field-level runner inputs: field size, ml_prob, rank + z within the race
  function fieldFeatures(feats) {
    const n = feats.length;
    const ml = feats.map(f => 1 / (f.morning_line + 1));
    const s = pdSum(ml);
    feats.forEach((f, i) => { f.field_size = n; f.ml_prob = ml[i] / s; });
    for (const c of RELATIVE) {
      const v = feats.map(f => f[c]);
      const r = rankDesc(v);
      const m = pdMean(v), sd = pdStd(v);
      feats.forEach((f, i) => {
        f[c + '_rank'] = r[i];
        f[c + '_z'] = (isNum(sd) && sd > 0) ? (v[i] - m) / sd : NaN;
      });
    }
    // features.csv round trip: every number the trees saw was written and read back
    for (const f of feats) for (const k in f) if (typeof f[k] === 'number') f[k] = csvTrip(f[k]);
    return feats;
  }

  // ------------------------------------------------------------ 3. race-level inputs
  function raceFeatures(feats, pmkt) {
    const n = feats.length;
    const out = feats.map(() => ({}));
    const fav = Math.max(...pmkt);
    const rk = rankMinDesc(pmkt);
    const eff = 1 / pdSum(pmkt.map(p => p ** 2));
    const live = pmkt.filter(p => p >= 0.10).length;
    const fgap = topTwoGap(pmkt);
    out.forEach((o, i) => {
      o.rs_fav_prob = fav; o.rs_own_over_fav = pmkt[i] / fav; o.rs_mkt_rank = rk[i];
      o.rs_eff_contenders = eff; o.rs_n_live = live; o.rs_fav_gap = fgap;
      o.rs_is_fav = rk[i] === 1 ? 1 : 0;
    });
    for (const [src, tag] of [['speed_best3', 'sf'], ['pace_e1_avg3', 'e1'], ['late_avg3', 'lp'],
                              ['quirin', 'qu'], ['prime_power', 'pp']]) {
      const v = feats.map(f => f[src]);
      const e = ownMinusBestOther(v), sd = pdStd(v), g = topTwoGap(v);
      out.forEach((o, i) => { o[`rs_${tag}_edge`] = e[i]; o[`rs_${tag}_spread`] = sd; o[`rs_${tag}_field_gap`] = g; });
    }
    const early = feats.map(f => (f.run_style || '').toUpperCase().trim().startsWith('E'));
    const ne = early.filter(Boolean).length;
    out.forEach((o, i) => {
      o.rs_n_early = ne; o.rs_is_early = early[i] ? 1 : 0;
      o.rs_lone_early = (ne === 1 && early[i]) ? 1 : 0; o.rs_early_share = ne / n;
    });
    const pv = feats.map(f => f.purse_avg3);
    const psd = pdStd(pv), pmed = medianNN(pv);
    const nft = feats.reduce((a, f) => a + (f.first_time_starter || 0), 0);
    out.forEach((o, i) => {
      o.rs_class_spread = psd; o.rs_class_vs_field = pv[i] / pmed; o.rs_n_firsttime = nft;
      for (const k in o) if (o[k] === Infinity || o[k] === -Infinity) o[k] = NaN;
    });
    return out;
  }

  // market probability from odds; a coupled entry's price is split evenly among its horses
  function interestOf(program) { return String(program).replace(/[A-Z]+$/, ''); }
  function marketProbs(programs, odds) {
    const cnt = {};
    programs.forEach(p => { const k = interestOf(p); cnt[k] = (cnt[k] || 0) + 1; });
    const raw = odds.map((o, i) => (1 / (o + 1)) / cnt[interestOf(programs[i])]);
    const s = pdSum(raw);
    const y = raw.map(r => csvTrip(r / s));           // step5 y_p_market, via features.csv
    const s2 = pdSum(y);
    return y.map(v => v / s2);                        // step15 renormalises within the race
  }

  function buildX(model, feats, rs, pmkt) {
    const cols = model.columns;
    return feats.map((f, i) => {
      const x = new Float64Array(cols.length);
      for (let j = 0; j < cols.length; j++) {
        const c = cols[j];
        let v;
        if (c === '_p_mkt') v = pmkt[i];
        else if (c in rs[i]) v = rs[i][c];
        else if (c.startsWith('run_style_')) v = f.run_style === c.slice(10) ? 1 : 0;
        else if (c.startsWith('sex_')) v = f.sex === c.slice(4) ? 1 : 0;
        else if (c.startsWith('race_type_')) v = 0;          // chart-only field; no tree uses it
        else v = f[c];
        x[j] = (v === undefined || v === null) ? NaN : v;
      }
      return x;
    });
  }

  // ------------------------------------------------------------ 4. trees
  function evalTree(node, x) {
    while (node.leaf_value === undefined) {
      let v = x[node.split_feature];
      const mt = node.missing_type;
      if (Number.isNaN(v) && mt !== 'NaN') v = 0.0;
      if ((mt === 'Zero' && Math.abs(v) <= 1e-35) || (mt === 'NaN' && Number.isNaN(v)))
        node = node.default_left ? node.left_child : node.right_child;
      else node = v <= node.threshold ? node.left_child : node.right_child;
    }
    return node.leaf_value;
  }
  function foldRaw(fm, X, pmkt) {
    return X.map((x, i) => {
      let s = 0;
      for (const t of fm.trees) s += evalTree(t, x);
      return fm.alpha * Math.log(Math.max(pmkt[i], 1e-12)) + s;
    });
  }
  function softmax(u) {
    const mx = Math.max(...u);
    const e = u.map(v => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }
  // live: the five fold models averaged on the log scale
  function predict(model, X, pmkt, onlyFold) {
    const folds = onlyFold === undefined ? model.folds : model.folds.filter(f => f.fold === onlyFold);
    const raw = new Array(X.length).fill(0);
    for (const fm of folds) foldRaw(fm, X, pmkt).forEach((v, i) => { raw[i] += v / folds.length; });
    return softmax(raw);
  }

  // full pipeline for one race: runners = horses not scratched, odds in the same order
  function scoreRace(model, race, runners, odds, meetStart, onlyFold) {
    const feats = fieldFeatures(runners.map(h => horseFeatures(h, race, meetStart)));
    const programs = runners.map(h => h.program);
    const pmkt = marketProbs(programs, odds);
    const rs = raceFeatures(feats, pmkt);
    const X = buildX(model, feats, rs, pmkt);
    const p = predict(model, X, pmkt, onlyFold);
    return { feats, rs, X, pmkt, p };
  }

  // ------------------------------------------------------------ 5. Plan A tickets
  const BUTTONS = [2, 3, 5, 10, 15, 20];
  const TAKE = 0.78, TOP_N = 8;

  function buttonSplit(money, k) {
    if (k === 0) return [];
    k = Math.min(k, Math.floor(money / BUTTONS[0]));
    if (k === 0) return [];
    const share = money / k;
    const under = BUTTONS.filter(b => b <= share);
    const base = under.length ? Math.max(...under) : BUTTONS[0];
    const st = new Array(k).fill(base);
    let left = money - st.reduce((a, b) => a + b, 0);
    let moved = true;
    while (moved) {
      moved = false;
      for (let j = 0; j < k; j++) {
        const nxt = BUTTONS.filter(b => b > st[j]);
        if (nxt.length && nxt[0] - st[j] <= left) { left -= nxt[0] - st[j]; st[j] = nxt[0]; moved = true; }
      }
    }
    return st;
  }

  // every exacta among our top 8, in the same order step40 builds them
  function exactaCandidates(programs, pOurs, odds, a2) {
    const s0 = npsum(pOurs);
    const po = pOurs.map(v => v / s0);
    const i0 = odds.map(o => 1 / (o + 1));
    const si = npsum(i0);
    const pp = i0.map(v => v / si);
    const order = po.map((v, i) => i).sort((x, y) => (po[y] - po[x]) || (x - y));
    const keep = order.slice(0, TOP_N);
    const q2 = po.map(v => Math.pow(Math.max(v, 1e-12), a2));
    const Q2 = npsum(q2);
    const ex = [];
    for (let x = 0; x < keep.length; x++) for (let y = 0; y < keep.length; y++) {
      if (x === y) continue;
      const I = keep[x], J = keep[y];
      const ours = po[I] * q2[J] / (Q2 - q2[I]);
      const pool = pp[I] * pp[J] / (1 - pp[I]);
      const est1 = TAKE / Math.max(pool, 1e-9);
      ex.push({ a: programs[I], b: programs[J], combo: programs[I] + '-' + programs[J],
                ours, pool, gap: ours - pool, est1, ia: I, ib: J });
    }
    return { ex, po, pp };
  }

  function argmaxGap(ex) { let b = 0; for (let i = 1; i < ex.length; i++) if (ex[i].gap > ex[b].gap) b = i; return b; }

  // build40(rid, (1.1, 'flip'), ...) -- one key, up to 5 under, flip, buttons, 1.1x after buttons
  function planA(ex, raceMoney, mult) {
    mult = mult === undefined ? 1.1 : mult;
    if (!ex.length) return null;
    const score = new Map();
    for (const e of ex) if (e.gap > 0) score.set(e.a, (score.get(e.a) || 0) + e.gap);
    if (!score.size) score.set(ex[argmaxGap(ex)].a, 1.0);
    const ranked = [...score.keys()].map((k, i) => [k, score.get(k), i])
      .sort((u, v) => (v[1] - u[1]) || (u[2] - v[2])).map(u => u[0]);
    const key = ranked[0];
    const cap = 5;
    const idx = [];
    ex.forEach((e, i) => { if (e.a === key && e.gap > 0) idx.push(i); });
    idx.sort((u, v) => (ex[v].gap - ex[u].gap) || (u - v));
    let order = idx.slice(0, cap + 3);
    let flipIdx = null;
    if (idx.length) {
      const i0 = idx[0];
      const fl = ex.findIndex(e => e.a === ex[i0].b && e.b === ex[i0].a);
      if (fl >= 0 && ex[fl].gap > 0) { order.splice(1, 0, fl); flipIdx = fl; }
    }
    if (!order.length) order = [argmaxGap(ex)];
    const perKey = new Map();
    let chosen = [];
    for (const i of order) {
      const k = ex[i].a;
      const lim = cap + (k !== key ? 1 : 0);
      if ((perKey.get(k) || 0) < lim) { chosen.push(i); perKey.set(k, (perKey.get(k) || 0) + 1); }
    }
    chosen = chosen.slice(0, 6);
    let dropped = [];
    for (let it = 0; it < 10; it++) {
      const st = buttonSplit(raceMoney, chosen.length);
      chosen = chosen.slice(0, st.length);
      const total = st.reduce((a, b) => a + b, 0);
      const bad = [];
      chosen.forEach((i, j) => { if (ex[i].est1 * st[j] < mult * total) bad.push(j); });
      if (!bad.length || chosen.length === 1) break;
      dropped = dropped.concat(bad.map(j => chosen[j]));
      chosen = chosen.filter((i, j) => !bad.includes(j));
    }
    const st = buttonSplit(raceMoney, chosen.length);
    const tickets = chosen.map((i, j) => ({
      combo: ex[i].combo, a: ex[i].a, b: ex[i].b, stake: st[j], est: ex[i].est1 * st[j],
      gap: ex[i].gap, ours: ex[i].ours, pool: ex[i].pool, flip: i === flipIdx,
    }));
    return { key, tickets, total: st.reduce((a, b) => a + b, 0), dropped: dropped.map(i => ex[i].combo) };
  }

  // race money: $200 / races on the card, nudged by race strength, $10-$50, whole dollars
  function raceMoney(maxGap, racesOnCard, strengthNorm, dayTarget) {
    const strength = Math.max(maxGap, 1e-4);
    const factor = Math.sqrt(strength) / strengthNorm;
    const base = (dayTarget || 200) / racesOnCard;
    return roundHalfEven(Math.min(Math.max(base * factor, 10), 50));
  }

  // live: coupled entries (1, 1A) become one betting interest before the exacta maths
  function collapseInterests(programs, p, odds) {
    const map = new Map();
    programs.forEach((pg, i) => {
      const k = interestOf(pg);
      if (!map.has(k)) map.set(k, { program: k, p: 0, odds: odds[i], members: [] });
      const m = map.get(k); m.p += p[i]; m.members.push(pg);
    });
    return [...map.values()];
  }

  // ------------------------------------------------------------ 6. charts (for scoring a day)
  function parseChart2(textIn) {
    const out = [];
    for (const row of parseCSV(textIn)) {
      const raceNo = whole(row, 3);
      if (raceNo === null) continue;
      const program = normProgram(get(row, 9));
      const official = whole(row, 61), post = whole(row, 8);
      const scr = ['SCR', 'SCRATCH', ''].includes(program) || official === 0 || official === null || (post !== null && post >= 90);
      out.push({ race_no: raceNo, program, name: get(row, 5), finish: official,
                 final_odds: num(row, 31), win_pay: num(row, 51), scratch: scr });
    }
    return out;
  }
  function parseChart4(textIn) {
    const out = [];
    for (const row of parseCSV(textIn)) {
      const raceNo = whole(row, 3);
      if (raceNo === null) continue;
      out.push({ race_no: raceNo, bet_type: text(row, 5), base: num(row, 6),
                 payout: num(row, 7), combo: text(row, 9), pool: num(row, 10) });
    }
    return out;
  }

  // ------------------------------------------------------------ 6b. tracks other than Fonner
  // Fonner's opening days come from model.json. Other meets are listed here.
  const OTHER_MEETS = [
    { track: 'EUR', name: 'Eureka Downs', short: 'Eureka', start: '2026-10-17', end: '2026-11-01' },
  ];
  function trackInfo(model, track, date) {
    const y = String(date || '').slice(0, 4);
    if (!track || track === 'FON')
      return { track: 'FON', name: 'Fonner Park', short: 'Fonner',
               meet_start: (model && model.meet_start && model.meet_start[y]) || (y + '-02-14') };
    const d = String(date || '');
    const inWin = m => d >= m.start && d <= m.end;
    // match on the track code; an unknown code during a listed meet's dates is taken to be that meet
    const m = OTHER_MEETS.find(x => x.track === track) ||
              (OTHER_MEETS.some(x => x.track === track) ? null : OTHER_MEETS.find(inWin));
    if (m) return { track, name: m.name, short: m.short, meet_start: m.start };
    return { track, name: track, short: track, meet_start: null };   // unknown meet: days into meet left blank
  }
  // What the Fonner data covered: 4f to 1 1/8m, dirt, 3-year-olds and up (1,112 races, 2023-2026).
  // Not a thoroughbred race (breed field is TB in every Fonner row), or under 4f: no numbers.
  // The rest still get numbers, with a marker when they fall outside what Fonner had.
  function raceScope(race) {
    const y = race.distance_yards;
    if (race.breed && race.breed.toUpperCase() !== 'TB') return { modeled: false, outside: true, why: 'breed' };
    if (y !== null && y !== undefined && y < 880) return { modeled: false, outside: true, why: 'distance' };
    const twoYO = (race.age_sex && race.age_sex.charAt(0) === 'A') ||
                  (race.horses.length > 0 && race.horses.every(h => h.age !== null && h.age <= 2));
    const why = [];
    if (twoYO) why.push('2yo');
    if (y !== null && y !== undefined && y > 1980) why.push('distance');
    if (race.surface && race.surface.toUpperCase() !== 'D') why.push('surface');
    return { modeled: true, outside: why.length > 0, why: why.join(',') };
  }

  const api = {
    parseCSV, parseDRF, horseFeatures, fieldFeatures, raceFeatures, marketProbs, buildX,
    evalTree, foldRaw, predict, softmax, scoreRace, exactaCandidates, planA, buttonSplit,
    raceMoney, collapseInterests, interestOf, parseChart2, parseChart4, npsum, roundHalfEven,
    pyRepr, pdParse, csvTrip, pdSum, pdMean, pdStd,
    normProgram, normName, BUTTONS, trackInfo, raceScope, OTHER_MEETS,
  };

  // ------------------------------------------------------------ 7. one race, end to end (live)
  // scratched: Set of program numbers; oddsOf(interest) -> odds to 1; meetStart: 'YYYY-MM-DD'
  function runRace(model, racesOnCard, race, scratched, oddsOf, meetStart) {
    const runners = race.horses.filter(h => !scratched.has(h.program));
    if (runners.length < 2) return { error: 'Fewer than two horses left, so there is no exacta to bet.' };
    const odds = runners.map(h => oddsOf(interestOf(h.program)));
    if (odds.some(o => !(o > 0))) return { error: 'Enter odds for every horse first.' };
    const sc = scoreRace(model, race, runners, odds, meetStart);
    const ints = collapseInterests(runners.map(h => h.program), sc.p, odds);
    const { ex, pp } = exactaCandidates(ints.map(x => x.program), ints.map(x => x.p), ints.map(x => x.odds), model.a2);
    ints.forEach((x, i) => { x.board = pp[i]; x.name = runners.find(h => interestOf(h.program) === x.program).name; });
    const maxGap = ex.length ? Math.max(...ex.map(e => e.gap)) : 0;
    const money = raceMoney(maxGap, racesOnCard, model.strength_norm, model.day_target);
    const plan = planA(ex, money, model.mult);
    return { runners, interests: ints, plan, money, maxGap, horseP: sc.p, pmkt: sc.pmkt };
  }

  function parseOdds(t) {
    t = String(t || '').trim().replace('/', '-');
    const m = /^(\d+(?:\.\d+)?)(?:-(\d+(?:\.\d+)?))?$/.exec(t);
    if (!m) return null;
    const v = m[2] ? (+m[1]) / (+m[2]) : +m[1];
    return v > 0 ? v : null;
  }
  function fmtOdds(v) {
    const ladder = [[1/10,'1-10'],[1/9,'1-9'],[1/5,'1-5'],[2/5,'2-5'],[1/2,'1-2'],[3/5,'3-5'],[4/5,'4-5'],[1,'1-1'],
      [6/5,'6-5'],[7/5,'7-5'],[3/2,'3-2'],[8/5,'8-5'],[9/5,'9-5'],[2,'2-1'],[5/2,'5-2'],[3,'3-1'],[7/2,'7-2'],[4,'4-1'],[9/2,'9-2']];
    for (const [x, s] of ladder) if (Math.abs(v - x) < 1e-9) return s;
    if (v >= 5 && Math.abs(v - Math.round(v)) < 1e-9) return Math.round(v) + '-1';
    return v < 10 ? v.toFixed(1) + '-1' : Math.round(v) + '-1';
  }
  function fairOdds(p) { return p > 0 ? (1 - p) / p : Infinity; }

  api.runRace = runRace; api.parseOdds = parseOdds; api.fmtOdds = fmtOdds; api.fairOdds = fairOdds;

  // ------------------------------------------------------------ 8. DRF-only line (Step 6 S3, no odds)
  // Shown before live odds are entered. Never used to pick bets.
  function drfLine(model, feats) {
    const L = model.drf_line;
    const u = feats.map(f => {
      let s = 0;
      for (let j = 0; j < L.cols.length; j++) {
        const c = L.cols[j];
        let v;
        if (c.startsWith('rs_')) v = (c === 'rs_NA') ? 0 : (f.run_style === c.slice(3) ? 1 : 0);
        else v = f[c];
        let z = (v === null || v === undefined || Number.isNaN(v)) ? NaN : (v - L.mu[j]) / L.sd[j];
        if (Number.isNaN(z)) z = 0;
        s += z * L.coef[j];
      }
      return s;
    });
    return softmax(u);
  }
  // DRF line for a race with the current scratches (runners in DRF order)
  function drfLineRace(model, race, scratched, meetStart) {
    const runners = race.horses.filter(h => !scratched.has(h.program));
    if (runners.length < 2) return { runners, p: runners.map(() => 1) };
    const feats = fieldFeatures(runners.map(h => horseFeatures(h, race, meetStart)));
    return { runners, feats, p: drfLine(model, feats) };
  }
  api.drfLine = drfLine; api.drfLineRace = drfLineRace;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Fonner = api;
})(typeof window !== 'undefined' ? window : this);
