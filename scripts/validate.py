#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema>=4"]
# ///
import argparse, atexit, json, math, sys
from pathlib import Path
from jsonschema import Draft202012Validator

from gpqa_zip import cleanup_unlocked, unlock_paths

def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def validators(spec):
    """Builds (request, response) validators whose internal $refs resolve against the full spec.

    Schema fragments under components/schemas reference siblings via JSON pointers
    (e.g. ``#/components/schemas/Answer``) that only exist in the root document, so each
    validator wraps the fragments as a self-contained root document entered via
    ``$ref`` -- replacing the deprecated ``RefResolver.from_schema``.
    """
    return tuple(Draft202012Validator({'$ref': f'#/components/schemas/{name}', 'components': spec['components']})
                 for name in ('SystemOneRequest', 'SystemOneResponse'))

def finite01(x): return isinstance(x,(int,float)) and math.isfinite(x) and 0 <= x <= 1

def semantic_pair(req,resp):
    assert set(resp['answers']) == set(req['questions']), 'answer/question IDs differ'
    for qid,q in req['questions'].items():
        a=resp['answers'][qid]
        assert a['type']==q['type'], f'{qid}: answer type mismatch'
        if q['type']=='choice':
            keys=set(q['criteria']); probs=a['probabilities']
            assert a['choice'] in keys, f'{qid}: invalid choice'
            assert set(probs)==keys, f'{qid}: probability keys differ from criteria'
            assert all(finite01(v) for v in probs.values()), f'{qid}: invalid probability'
            assert abs(sum(probs.values())-1)<=1e-5, f'{qid}: probabilities do not sum to 1'
            assert finite01(a['confidence']), f'{qid}: invalid confidence'
        elif q['type']=='score':
            n=len(q['criteria']); expected={str(i) for i in range(n)}
            assert n>=2, f'{qid}: benchmark requires >=2 score levels'
            assert set(a['legend'])==expected and set(a['probabilities'])==expected, f'{qid}: score level keys mismatch'
            assert [a['legend'][str(i)] for i in range(n)]==q['criteria'], f'{qid}: legend differs from request rubric'
            assert all(finite01(v) for v in a['probabilities'].values()), f'{qid}: invalid probability'
            assert abs(sum(a['probabilities'].values())-1)<=1e-5, f'{qid}: probabilities do not sum to 1'
            expected_score=sum(i*a['probabilities'][str(i)] for i in range(n))
            assert abs(a['score']-expected_score)<=1e-5, f'{qid}: score is not probability-weighted expectation'
            assert finite01(a['confidence']), f'{qid}: invalid confidence'
        elif q['type']=='noul':
            assert finite01(a['noul']), f'{qid}: invalid noul'

def validate_e2e(root, sv):
    """Schema-checks the most recent e2e run log (output/e2e.jsonl) against the response schema.

    Transport failures and aborted cases are counted, not validated; only
    records that actually carry a response are asserted on, so a partial or
    failed e2e run still validates cleanly for the cases that succeeded.
    """
    p=root/'output'/'e2e.jsonl'
    if not p.exists(): return None
    recs=load_jsonl(p)
    out={'log':'output/e2e.jsonl','records':len(recs),'responses_ok':0,'failed':0,'aborted':0}
    for i,rec in enumerate(recs,1):
        et=rec.get('error_type')
        if et=='aborted': out['aborted']+=1; continue
        if et: out['failed']+=1; continue
        errs=list(sv.iter_errors(rec['response']))
        assert not errs, f"e2e:{rec.get('request_id',i)} response schema: {errs[0].message}" if errs else ''
        out['responses_ok']+=1
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',default='.')
    args=ap.parse_args(); root=Path(args.benchmark)
    spec=json.loads((root/'openapi/typesafe-systemone-openapi-v0.2.0.json').read_text())
    rv,sv=validators(spec)
    manifest=json.loads((root/'manifest.json').read_text())
    # gated files (gpqa_diamond) live as password-protected zips at rest; decrypt
    # for this run, remove/re-encrypt at exit (stderr logging keeps stdout JSON clean)
    atexit.register(cleanup_unlocked, unlock_paths(
        [root/e[k] for e in manifest['capabilities'].values() for k in ('requests','responses')]))
    total=0
    all_request_hashes=set()
    for cap,e in manifest['capabilities'].items():
        reqs=load_jsonl(root/e['requests']); golds=load_jsonl(root/e['responses'])
        assert len(reqs)==len(golds)==e['cases'], f'{cap}: line count mismatch'
        for i,(req,gold) in enumerate(zip(reqs,golds),1):
            errs=list(rv.iter_errors(req)); assert not errs, f'{cap}:{i} request schema: {errs[0].message}' if errs else ''
            errs=list(sv.iter_errors(gold)); assert not errs, f'{cap}:{i} response schema: {errs[0].message}' if errs else ''
            semantic_pair(req,gold)
            h=json.dumps(req, sort_keys=True, ensure_ascii=False, separators=(',',':'))
            import hashlib
            hh=hashlib.sha256(h.encode()).hexdigest()
            assert hh not in all_request_hashes, f'{cap}:{i} duplicate request body'
            all_request_hashes.add(hh)
            total+=1
    e2e=validate_e2e(root, sv)
    result={'ok':True,'cases':total,'capabilities':len(manifest['capabilities'])}
    if e2e is not None: result['e2e']=e2e
    print(json.dumps(result))
if __name__=='__main__': main()
