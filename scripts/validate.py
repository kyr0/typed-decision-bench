#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["jsonschema>=4"]
# ///
"""/ Two-layer validator for the whole benchmark.

Layer 1 is structural: every request, golden response and metadata line must
satisfy the SystemOne OpenAPI schemas, fetched from the canonical live spec
(https://api.typesafe.ai/openapi.json — no local copy to drift out of sync;
`--openapi` accepts a local path for offline runs). Layer 2 is semantic —
schema-valid payloads can still be wrong (answers not matching question IDs,
probabilities not summing to 1, a score that is not the probability-weighted
expectation, a score legend differing from the request rubric). The status
document is printed as JSON to stdout; `make validate` redirects it to
validation_status.json. The most recent e2e run log (output/e2e.jsonl) is
response-schema-checked too, counting — not validating — transport failures
and aborted cases.
"""
import argparse, atexit, json, math, sys, urllib.request
from pathlib import Path
from jsonschema import Draft202012Validator

# canonical wire contract lives with the API it describes; the http:// form 301s here
OPENAPI_URL = 'https://api.typesafe.ai/openapi.json'
OPENAPI_TIMEOUT_S = 15

from gpqa_zip import cleanup_unlocked, unlock_paths
from splits import VALID_SPLITS

def load_jsonl(path):
    """/ Parse a JSONL file into list[dict], skipping blank lines."""
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
    """/ Semantic checks for one (request, response) pair that JSON Schema cannot
    express; raises AssertionError with the question ID and the violated rule."""
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
    """/ Validates every manifest capability (requests + gold responses aligned with
    metadata, semantic pairs included) plus the latest e2e log; prints the status
    document to stdout and exits non-zero on any violation."""
    ap=argparse.ArgumentParser()
    ap.add_argument('--benchmark',default='.')
    ap.add_argument('--openapi',default=OPENAPI_URL,
                    help='OpenAPI spec: URL (default: the live spec) or local file path (offline runs)')
    args=ap.parse_args(); root=Path(args.benchmark)
    if args.openapi.startswith(('http://','https://')):
        # urlopen follows the http->https redirect; fail loudly, never skip validation
        with urllib.request.urlopen(args.openapi, timeout=OPENAPI_TIMEOUT_S) as r:
            spec=json.load(r)
    else:
        spec=json.loads(Path(args.openapi).read_text(encoding='utf-8'))
    rv,sv=validators(spec)
    manifest=json.loads((root/'manifest.json').read_text())
    # gated files (gpqa_diamond) live as password-protected zips at rest; decrypt
    # for this run, remove/re-encrypt at exit (stderr logging keeps stdout JSON clean)
    atexit.register(cleanup_unlocked, unlock_paths(
        [root/e[k] for e in manifest['capabilities'].values() for k in ('requests','responses','metadata') if e.get(k)]))
    total=0
    split_totals={name:0 for name in VALID_SPLITS}
    all_request_hashes=set()
    for cap,e in manifest['capabilities'].items():
        assert e.get('metadata'), f'{cap}: manifest metadata path is required'
        reqs=load_jsonl(root/e['requests']); golds=load_jsonl(root/e['responses']); metas=load_jsonl(root/e['metadata'])
        assert len(reqs)==len(golds)==len(metas)==e['cases'], f'{cap}: request/response/metadata line count mismatch'
        cap_splits={name:0 for name in VALID_SPLITS}
        for i,(req,gold,meta) in enumerate(zip(reqs,golds,metas),1):
            assert isinstance(meta,dict), f'{cap}:{i} metadata row must be an object'
            split=meta.get('split')
            assert split in VALID_SPLITS, f'{cap}:{i} metadata split must be one of {VALID_SPLITS}, got {split!r}'
            cap_splits[split]+=1; split_totals[split]+=1
            errs=list(rv.iter_errors(req)); assert not errs, f'{cap}:{i} request schema: {errs[0].message}' if errs else ''
            errs=list(sv.iter_errors(gold)); assert not errs, f'{cap}:{i} response schema: {errs[0].message}' if errs else ''
            semantic_pair(req,gold)
            h=json.dumps(req, sort_keys=True, ensure_ascii=False, separators=(',',':'))
            import hashlib
            hh=hashlib.sha256(h.encode()).hexdigest()
            assert hh not in all_request_hashes, f'{cap}:{i} duplicate request body'
            all_request_hashes.add(hh)
            total+=1
        assert cap_splits['test']>0, f'{cap}: requires at least one split=test row'
        assert cap_splits['calibrate']>0, f'{cap}: requires at least one split=calibrate row'
    e2e=validate_e2e(root, sv)
    result={'ok':True,'cases':total,'capabilities':len(manifest['capabilities']),'splits':split_totals}
    if e2e is not None: result['e2e']=e2e
    print(json.dumps(result))
if __name__=='__main__': main()
