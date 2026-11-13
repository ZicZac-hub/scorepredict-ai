"""Frozen league-opponent challenger, walk-forward 2024/25 public-data holdout."""
import argparse, csv, io, json, math, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import requests
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from engine import predict, _grid
from league_opponent_model import expected_goals
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--download', action='store_true', help='Allow downloading public season CSVs')
parser.add_argument('--data-dir', type=Path, default=Path('work/model_benchmark_data'))
parser.add_argument('--output', type=Path, default=Path('work/model_repair_benchmark_results.json'))
args=parser.parse_args()
BASE=args.data_dir;BASE.mkdir(parents=True,exist_ok=True)
LEAGUES={'E0':'PL','D1':'BL1'}
SEASONS=['2223','2324','2425']
def get(pair):
 season, league=pair;path=BASE/f'{season}_{league}.csv'
 if not path.exists():
  if not args.download:raise RuntimeError(f'Missing {path}; pass --download to fetch public data')
  r=requests.get(f'https://www.football-data.co.uk/mmz4281/{season}/{league}.csv',timeout=20);r.raise_for_status();path.write_bytes(r.content)
 return season,league,path.read_text(encoding='utf-8-sig')
allrows=[]
for season,league,txt in ThreadPoolExecutor(max_workers=6).map(get,[(s,l) for s in SEASONS for l in LEAGUES]):
 for row in csv.DictReader(io.StringIO(txt)):
  if not row.get('Date') or not row.get('FTHG') or not row.get('FTAG'):continue
  d=datetime.strptime(row['Date'],'%d/%m/%Y').replace(tzinfo=timezone.utc)
  # Dates are sufficient: all matches on the same day excluded from inputs.
  allrows.append({'id':f'{league}-{season}-{row["Date"]}-{row["HomeTeam"]}-{row["AwayTeam"]}', 'provider':'football-data.co.uk','utcDate':d.isoformat(),'status':'FINISHED','homeTeam':{'id':row['HomeTeam'],'name':row['HomeTeam']},'awayTeam':{'id':row['AwayTeam'],'name':row['AwayTeam']},'competition':{'code':LEAGUES[league]},'score':{'fullTime':{'home':int(row['FTHG']),'away':int(row['FTAG'])}},'dataset_season':season})
start=time.monotonic();results={};paired_differences=[]
for code in LEAGUES.values():
 rows=sorted([r for r in allrows if r['competition']['code']==code],key=lambda r:r['utcDate']);out={n:{'n':0,'exact_hits':0,'logloss':0.,'brier':0.,'modes':Counter()} for n in ['baseline','league_opponent']};paired=0
 for i,f in enumerate(rows):
  if f['dataset_season']!='2425':continue
  past=[r for r in rows[:i] if r['utcDate']<f['utcDate']]
  baseline=predict(f,past)
  candidate=expected_goals(f,past)
  if not baseline or candidate is None:continue
  paired+=1
  matrices={'baseline':baseline['matrix'],'league_opponent':_grid(*candidate)}
  hg=f['score']['fullTime']['home'];ag=f['score']['fullTime']['away'];winner=0 if hg>ag else 1 if hg==ag else 2
  pair_metrics={}
  for name,m in matrices.items():
   s=out[name];s['n']+=1;top=np.unravel_index(m.argmax(),m.shape);s['exact_hits']+=int(top==(hg,ag));s['modes'][f'{top[0]}-{top[1]}']+=1
   p=float(m[hg,ag]) if hg<m.shape[0] and ag<m.shape[1] else 0.;s['logloss']-=math.log(max(p,1e-12))
   probs=[float(np.tril(m,-1).sum()),float(np.trace(m)),float(np.triu(m,1).sum())];brier=sum((p-int(k==winner))**2 for k,p in enumerate(probs));s['brier']+=brier
   pair_metrics[name]=(-math.log(max(p,1e-12)),brier,int(top==(hg,ag)))
  paired_differences.append({'league':code,'logloss':pair_metrics['league_opponent'][0]-pair_metrics['baseline'][0],'brier':pair_metrics['league_opponent'][1]-pair_metrics['baseline'][1],'exact':pair_metrics['league_opponent'][2]-pair_metrics['baseline'][2]})
 for s in out.values():
  n=s['n'];s['exact_rate']=s['exact_hits']/n;s['score_logloss']=s.pop('logloss')/n;s['outcome_brier']=s.pop('brier')/n;s['modes']=dict(s['modes'])
 results[code]=out
 print(code,json.dumps(out),flush=True)
uncertainty={}
for key in ['logloss','brier','exact']:
 vals=np.array([r[key] for r in paired_differences]);mean=float(vals.mean());se=float(vals.std(ddof=1)/np.sqrt(len(vals)));uncertainty[key]={'candidate_minus_baseline':mean,'approx_95_interval':[mean-1.96*se,mean+1.96*se]}
report={'source':'https://www.football-data.co.uk/data.php','training_seasons':['2022/23','2023/24'],'holdout_season':'2024/25','method':'Frozen existing league_opponent hyperparameters, expanding chronological past, excludes all same-day matches, paired forecasts only. No hyperparameter selection on holdout. Baseline is the current corrected competition-scoped engine. Goal-rate ceilings removed; score grids include their tails. Exact accuracy is reported separately from distribution quality.','leagues':results,'elapsed_seconds':time.monotonic()-start,'paired_difference_uncertainty':uncertainty,'uncertainty_note':'Normal approximation treats matches as independent; repeated teams and shared histories can reduce effective sample size.'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print('Saved',args.output,flush=True)
