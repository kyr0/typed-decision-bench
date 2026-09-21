#!/usr/bin/env python3
"""/ Scorer: joins predictions against the golden responses and emits two artifacts.

    output/<run>_stats.jsonl — one flat self-describing line per capability plus
    'micro' (accuracy, calibration, transport errors, latency) so lines from
    different models concatenate for comparison. output/<run>_cases.jsonl — one
    line per case with correctness + per-case calibration/latency for error
    analysis.

Two prediction sources: a responses/ directory (line-aligned with the gold
files, single-run mode) or the combined run log itself, which embeds every
successful response and supports --all-logs backfill. Every request_id maps to
(capability, line); the LAST record per case wins so resumed runs recover
failures instead of double-counting them. Calibration (nll/brier/soft_accuracy)
uses the FULL gold distribution, not the argmax one-hot.
"""
import argparse,atexit,json,math,re,sys
from pathlib import Path

try: from .gpqa_zip import cleanup_unlocked, unlock_paths  # imported as scripts.score (run_eval hook)
except ImportError: from gpqa_zip import cleanup_unlocked, unlock_paths  # run directly as scripts/score.py
CASE_ID_PREFIX='typed-decisions-bench-v1'
EPS=1e-12

def rows(p):
    """/ Parse a JSONL file into list[dict], skipping blank lines."""
    with open(p,encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]

def all_logs(root):
    """/ Every combined run log in output/, oldest first (empty list when none).
    Excludes this tool's own *_stats.jsonl / *_cases.jsonl outputs so re-runs
    never mistake a report for a run log."""
    d=root/'output'
    logs=[p for p in d.glob('*.jsonl') if not p.stem.endswith(('_stats','_cases'))] if d.is_dir() else []
    return sorted(logs,key=lambda p:p.stat().st_mtime)

def newest_log(root):
    """/ Newest combined run log in output/ (highest mtime), or None when none exist."""
    logs=all_logs(root)
    return logs[-1] if logs else None

def last_records(log_path):
    """/ {(capability, line_no): record} keeping only the LAST record per case:
    a resumed run appends retries after failures, so later lines override their
    earlier tries for errors, latency and predictions alike."""
    last={}
    with open(log_path,encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            cap,_,no=r.get('request_id','').rpartition('-')
            if no.isdigit(): last[(cap,int(no))]=r
    return last

def run_stats(log_path):
    """/ Per-capability aggregates + per-line latency from a combined run log;
    request_id is '<cap>-<line:03d>' and the LAST record per case wins, so a
    recovered case no longer counts as an error. Only successful records
    contribute durations — a failed request's time is not serving latency."""
    stats={}
    for (cap,no),r in last_records(log_path).items():
        s=stats.setdefault(cap,{'errors':0,'durations':[],'per_line':{}})
        if r.get('error_type'): s['errors']+=1
        elif r.get('duration_ms'):
            s['durations'].append(float(r['duration_ms']))
            s['per_line'][no]=float(r['duration_ms'])
    return stats

def distributions(ga,pa):
    """/ (labels, y, p): gold distribution y and normalized prediction p over labels.

    Calibration metrics must use the FULL gold distribution (not a one-hot
    argmax), so soft agreement and cross-entropy stay meaningful when gold
    itself spreads mass — noul maps to the [false, true] pair."""
    if ga['type']=='noul':
        labels=('false','true')
        y=(1.0-float(ga['noul']),float(ga['noul']))
        p=(1.0-float(pa['noul']),float(pa['noul']))
    else:
        labels=tuple(ga['probabilities'])
        y=tuple(float(ga['probabilities'][k]) for k in labels)
        p=tuple(float(pa.get('probabilities',{}).get(k,0.0)) for k in labels)
    s=sum(p) or 1.0
    return labels,y,tuple(x/s for x in p)

def case_record(cap,i,ga,pa,duration_ms):
    """/ One per-case JSONL record (output/<run>_cases.jsonl): correctness,
    calibration (nll/brier/soft against the full gold distribution) and the
    joined per-case latency — self-describing so cases from different models
    can simply be concatenated and compared."""
    labels,y,p=distributions(ga,pa)
    gold_i=max(range(len(y)),key=y.__getitem__); pred_i=max(range(len(p)),key=p.__getitem__)
    rec={'case_id':f'{CASE_ID_PREFIX}:{cap}:{i:04d}','family':cap,'qtype':ga['type'],
         'correct':int(pred_i==gold_i),'confidence':max(p),
         'nll':-sum(yy*math.log(max(EPS,pp)) for yy,pp in zip(y,p))+0.0,  # +0.0 avoids IEEE -0.0 in JSON
         'brier':sum((pp-yy)**2 for pp,yy in zip(p,y)),
         'soft_accuracy':sum(pp*yy for pp,yy in zip(p,y)),
         'prediction_label':labels[pred_i],'gold_label':labels[gold_i],
         'latency_ms':duration_ms}
    if ga['type']=='score':
        rec['score_error']=abs(sum(j*pp for j,pp in enumerate(p))-sum(j*yy for j,yy in enumerate(y)))
    return rec

def ece(records,bins=15):
    """/ Expected calibration error over equal-width confidence bins."""
    n=len(records); value=0.0
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins
        idx=[k for k,r in enumerate(records) if lo<=r['confidence']<hi or (b==bins-1 and r['confidence']>=lo)]
        if not idx: continue
        mc=sum(records[k]['confidence'] for k in idx)/len(idx)
        ma=sum(records[k]['correct'] for k in idx)/len(idx)
        value+=len(idx)/n*abs(mc-ma)
    return value

def mean_of(records,key):
    """/ Mean of a numeric per-case field; None when no case carries it."""
    xs=[r[key] for r in records if r.get(key) is not None]
    return round(sum(xs)/len(xs),4) if xs else None

def latency_fields(durations):
    """/ Nearest-rank mean/p50/p95 latency in ms; all None when nothing was measured."""
    if not durations: return {'latency_ms_mean':None,'latency_ms_p50':None,'latency_ms_p95':None}
    ds=sorted(durations); n=len(ds)
    return {'latency_ms_mean':round(sum(ds)/n,1),
            'latency_ms_p50':ds[min(n-1,math.ceil(.5*n)-1)],
            'latency_ms_p95':ds[min(n-1,math.ceil(.95*n)-1)]}

def stat_line(run,cap,records,errors,durations):
    """/ One flat, self-describing record per capability (or 'micro') so stats lines
    from different models/runs can simply be concatenated and compared."""
    scores=[r for r in records if 'score_error' in r]
    return {'run':run,'capability':cap,'n':len(records),
            'accuracy':mean_of(records,'correct'),  # mean_of rounds; None when a run covered no case
            'soft_accuracy':mean_of(records,'soft_accuracy'),
            'nll':mean_of(records,'nll'),'brier':mean_of(records,'brier'),
            'confidence':mean_of(records,'confidence'),'ece_15':round(ece(records),4) if records else None,
            'score_mae':mean_of(scores,'score_error') if scores else None,
            'score_within_one':round(sum(r['score_error']<=1 for r in scores)/len(scores),4) if scores else None,
            'errors':errors,'error_rate':round(errors/len(records),4) if records else None,
            **latency_fields(durations)}

def run_name(log_path):
    """/ output/<run>_<YYYY-MM-DD_HH_MM>.jsonl -> <run>; explicit names (e2e.jsonl) stay as-is."""
    return re.sub(r'_\d{4}-\d{2}-\d{2}_\d{2}_\d{2}$','',log_path.stem)

def prediction_lookup(respdir,man,root):
    """/ {(capability,line): response row} from a responses directory, line-aligned
    with the golden files; refuses misaligned directories loudly (a truncated
    responses dir would silently undercount accuracy)."""
    preds={}
    for cap,e in man['capabilities'].items():
        g=rows(root/e['responses']); p=rows(respdir/f'{cap}.jsonl')
        if len(g)!=len(p): raise SystemExit(f'{cap}: expected {len(g)} responses, got {len(p)}')
        for i,P in enumerate(p,1): preds[(cap,i)]=P
    return preds

def log_predictions(log_path):
    """/ {(capability,line): response payload} from a combined run log, taking
    the LAST record per case (a retry appended by a resumed run overrides its
    failed try). The log embeds every successful response, so each log can be
    scored standalone — this is what makes --all-logs work without a responses dir."""
    return {k:r['response'] for k,r in last_records(log_path).items()
            if not r.get('error_type') and isinstance(r.get('response'),dict)}

def build_report(name,root,man,preds,stats):
    """/ (stat lines, case records) for one run: a flat line per manifest
    capability plus 'micro'. Cases exist only where a usable prediction is
    present — failed-request placeholders (error rows from the log or the
    responses dir) are skipped, not crashed on, and still count as errors."""
    lines=[]; cases=[]
    for cap,e in man['capabilities'].items():
        s=stats.get(cap,{'errors':0,'durations':[],'per_line':{}})
        cap_cases=[]
        for i,G in enumerate(rows(root/e['responses']),1):
            P=preds.get((cap,i))
            answers=P.get('answers') if isinstance(P,dict) else None
            if not answers: continue  # error placeholder / failed request: no case, but counted as error
            qid=next(iter(G['answers']))
            if qid not in answers: continue
            cap_cases.append(case_record(cap,i,G['answers'][qid],answers[qid],s['per_line'].get(i)))
        cases+=cap_cases
        lines.append(stat_line(name,cap,cap_cases,s['errors'],s['durations']))
    all_err=sum(s['errors'] for s in stats.values())
    lines.append(stat_line(name,'micro',cases,all_err,
                           sorted(d for s in stats.values() for d in s['durations'])))
    return lines,cases

def write_report(out,lines,cases):
    """/ Writes the stats file plus its sibling <stem>_cases.jsonl; returns the cases path."""
    base=out.stem[:-len('_stats')] if out.stem.endswith('_stats') else out.stem
    cases_path=out.with_name(base+'_cases.jsonl')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text('\n'.join(json.dumps(l,ensure_ascii=False) for l in lines)+'\n')
    cases_path.write_text('\n'.join(json.dumps(c,ensure_ascii=False) for c in cases)+'\n')
    return cases_path

def score_log(root,man,log_path):
    """/ Scores one combined run log standalone (predictions from the log itself)
    into output/<run>_stats.jsonl + output/<run>_cases.jsonl; returns (stats,cases) paths."""
    name=run_name(log_path)
    lines,cases=build_report(name,root,man,log_predictions(log_path),run_stats(log_path))
    out=root/'output'/f'{name}_stats.jsonl'
    cases_path=write_report(out,lines,cases)
    print(f'wrote {len(lines)-1} capabilities + micro -> {out}; {len(cases)} cases -> {cases_path} '
          f'(log: {log_path})',file=sys.stderr)
    return out,cases_path

def main():
    """/ CLI entry: single-run mode (responses dir + newest/explicit log) or --all-logs
    backfill of every unscored run log; prints the stats lines to stdout."""
    ap=argparse.ArgumentParser(description='Score responses against gold; writes output/<run>_stats.jsonl '
                                           '(per-capability accuracy/errors/latency/calibration) and '
                                           'output/<run>_cases.jsonl (per-case records).')
    ap.add_argument('--benchmark',default='.')
    ap.add_argument('--responses',default=None,help='Directory containing <capability>.jsonl '
                                                    '(single-run mode; not needed with --all-logs)')
    ap.add_argument('--log',default=None,help='Combined run log for latency/errors; default: newest output/*.jsonl')
    ap.add_argument('--out',default=None,help='Stats JSONL path; default: output/<run>_stats.jsonl '
                                              '(cases go next to it as <stem>_cases.jsonl)')
    ap.add_argument('--all-logs',action='store_true',help='Score every output/*.jsonl run log that has no '
                                                          '<run>_stats.jsonl yet, using the responses embedded '
                                                          'in each log; skips already-scored runs')
    a=ap.parse_args(); root=Path(a.benchmark); man=json.loads((root/'manifest.json').read_text())
    # golden response files may be password-protected zips at rest (gpqa_diamond);
    # decrypt for this run, remove/re-encrypt at exit
    atexit.register(cleanup_unlocked, unlock_paths(
        [root/e['responses'] for e in man['capabilities'].values()]))
    if a.all_logs:
        if a.log or a.out or a.responses: ap.error('--all-logs takes no --log/--out/--responses')
        logs=all_logs(root)
        if not logs: print('no run logs found in output/',file=sys.stderr)
        for log_path in logs:
            if (root/'output'/f'{run_name(log_path)}_stats.jsonl').exists():
                print(f'skip {log_path.name} (already scored)',file=sys.stderr)
                continue
            score_log(root,man,log_path)
        return
    if not a.responses: ap.error('--responses is required without --all-logs')
    log_path=Path(a.log) if a.log else newest_log(root)
    stats=run_stats(log_path) if log_path and log_path.exists() else {}
    name=run_name(log_path) if log_path else 'score'
    lines,cases=build_report(name,root,man,prediction_lookup(Path(a.responses),man,root),stats)
    out=Path(a.out) if a.out else root/'output'/f'{name}_stats.jsonl'
    cases_path=write_report(out,lines,cases)
    print(f'wrote {len(lines)-1} capabilities + micro -> {out}; {len(cases)} cases -> {cases_path} '
          f'(log: {log_path})',file=sys.stderr)
    print(out.read_text(),end='')
if __name__=='__main__': main()
