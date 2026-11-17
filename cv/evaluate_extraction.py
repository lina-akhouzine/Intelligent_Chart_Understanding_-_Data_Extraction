"""Évaluer des prédictions déjà produites ; les CSV ne sont lus qu'après extraction."""
from __future__ import annotations
import argparse
import csv
import json
import random
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd
from bar_extractor import parse_number
from chart_data_extractor import ROOT,ChartDataExtractor,save_extraction


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC',str(text)).casefold().strip().rstrip('*').split())


def text_match(value,choices):
    value=normalized(value)
    exact=[c for c in choices if normalized(c)==value]
    if len(exact)==1:
        return exact[0],'exact'
    matches=sorted([(SequenceMatcher(None,value,normalized(c)).ratio(),c) for c in choices],reverse=True)
    if matches and matches[0][0]>=.92 and (len(matches)==1 or matches[0][0]-matches[1][0]>.04):
        return matches[0][1],'fuzzy_text_only'
    return None,None


def evaluate_table(result,reference_path):
    reference=pd.read_csv(reference_path,dtype=str,keep_default_na=False)
    if len(reference.columns)<2:
        return {'error':'CSV sans colonne numérique','matched':[],'missing':[],'extra':[]}
    category_column=reference.columns[0]; series=list(reference.columns[1:])
    categories=reference[category_column].tolist()
    expected={}; ambiguous=set()
    for _,row in reference.iterrows():
        for name in series:
            parsed=parse_number(row[name])
            if parsed is None:
                continue
            key=(row[category_column],name)
            if key in expected:
                ambiguous.add(key)
            expected[key]=parsed['value']
    for key in ambiguous:
        expected.pop(key,None)
    predictions={}; extra=[]; category_matches=set()
    extracted_ids={r['series_id'] for r in result['rows']}
    for row in result['rows']:
        category,method=text_match(row['category'],categories) if row.get('category') is not None else (None,None)
        if category is not None:
            category_matches.add(category)
        if row.get('series'):
            name,_=text_match(row['series'],series)
        elif len(series)==1 and len(extracted_ids)==1:
            name=series[0]  # une seule colonne possible, aucune décision basée sur les valeurs.
        else:
            name=None
        if row.get('value') is None:
            continue
        record={'predicted_category':row.get('category'),'predicted_series':row.get('series') or row['series_id'],
                'prediction':row['value'],'category':category,'series':name,'category_matching':method}
        key=(category,name)
        if key not in expected:
            extra.append({**record,'reason':'category_or_series_not_uniquely_matched'})
        else:
            predictions.setdefault(key,[]).append(record)
    matched=[]; missing=[]
    for key,value in expected.items():
        candidates=predictions.get(key,[])
        if len(candidates)==1:
            matched.append({**candidates[0],'reference':value,'error':candidates[0]['prediction']-value})
        else:
            missing.append({'category':key[0],'series':key[1],'reference':value,
                            'reason':'no_prediction' if not candidates else 'ambiguous_multiple_predictions'})
            extra.extend({**c,'reason':'ambiguous_multiple_predictions'} for c in candidates)
    errors=np.array([m['error'] for m in matched],dtype=float)
    nonzero=[m for m in matched if m['reference']!=0]
    numerical={'MAE':float(np.mean(np.abs(errors))) if len(errors) else None,
               'RMSE':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
               'MAPE_nonzero_percent':float(np.mean([abs(m['error']/m['reference']) for m in nonzero])*100) if nonzero else None}
    return {'reference_file':str(reference_path),'reference_values':len(expected),'numerical_metrics':numerical,
            'reference_categories':len(set(categories)),'matched_categories':len(category_matches),
            'matched':matched,'missing':missing,'extra':extra,'ambiguous_reference_cells_excluded':len(ambiguous),
            'single_series_fallback':len(series)==1 and len(extracted_ids)==1}


def aggregate(records):
    matched=[m for r in records for m in r['evaluation']['matched']]
    errors=np.array([m['error'] for m in matched],dtype=float)
    nonzero=[m for m in matched if m['reference']!=0]
    total=sum(r['evaluation'].get('reference_values',0) for r in records)
    categories=sum(r['evaluation'].get('reference_categories',0) for r in records)
    cat_matched=sum(r['evaluation'].get('matched_categories',0) for r in records)
    return {'images':len(records),'matched_values':len(matched),'reference_values':total,
            'missing_values':sum(len(r['evaluation']['missing']) for r in records),
            'extra_values':sum(len(r['evaluation']['extra']) for r in records),
            'MAE':float(np.mean(np.abs(errors))) if len(errors) else None,
            'RMSE':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            'MAPE_nonzero_percent':float(np.mean([abs(m['error']/m['reference']) for m in nonzero])*100) if nonzero else None,
            'MAPE_nonzero_pairs':len(nonzero),'category_match_rate':cat_matched/categories if categories else None,
            'value_coverage':len(matched)/total if total else None,
            'images_with_any_extracted_value':sum(r['extracted_values']>0 for r in records),
            'images_with_paired_values':sum(bool(r['evaluation']['matched']) for r in records),
            'any_extraction_rate':sum(r['extracted_values']>0 for r in records)/len(records) if records else None,
            'paired_extraction_rate':sum(bool(r['evaluation']['matched']) for r in records)/len(records) if records else None,
            'complete_reference_coverage_rate':sum(bool(r['evaluation']['matched']) and not r['evaluation']['missing'] and
                not r['evaluation']['extra'] for r in records)/len(records) if records else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=20,help='Nombre de barres verticales (dont 5 pilotes réutilisés)')
    parser.add_argument('--horizontal',action='store_true',help='Ajouter cinq h_bar en évaluation exploratoire')
    parser.add_argument('--reuse-run',type=Path,help='Évaluer les JSON déjà générés sans relancer la CV/OCR')
    args=parser.parse_args()
    output=ROOT/'results/cv4'/datetime.now().strftime('evaluation_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    if args.reuse_run:
        manifest=json.loads((args.reuse_run/'sample_manifest.json').read_text(encoding='utf-8'))
        extractor=None
    else:
        with (ROOT/'results/cv2/official_labels.csv').open(encoding='utf-8') as handle:
            rows=list(csv.DictReader(handle))
        cv3_manifest=json.loads((ROOT/'results/cv3/evaluation_20261117_013720_001076/sample_manifest.json').read_text())
        pilot=[r for r in cv3_manifest if r['label']=='v_bar']
        names={r['image'] for r in pilot}
        extra=[r for r in rows if r['split']=='test' and r['label']=='v_bar' and r['image'] not in names]
        rng=random.Random(74)
        if args.count<5 or args.count>len(pilot)+len(extra):
            parser.error('--count doit être compris entre 5 et la disponibilité des v_bar')
        manifest=[{**r,'sample_role':'development_pilot'} for r in pilot]
        manifest += [{**r,'sample_role':'additional_random_test_images'} for r in rng.sample(extra,args.count-5)]
        if args.horizontal:
            manifest += [{**r,'sample_role':'horizontal_exploratory'} for r in cv3_manifest if r['label']=='h_bar']
        extractor=ChartDataExtractor(horizontal=args.horizontal)
    (output/'sample_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    records=[]
    for index,row in enumerate(manifest):
        path=ROOT/row['image']
        if extractor is None:
            result=json.loads((args.reuse_run/path.stem/'extraction.json').read_text(encoding='utf-8'))
            prediction_folder=args.reuse_run/path.stem
        else:
            try:
                result,rgb=extractor.extract(path)
                prediction_folder=save_extraction(result,rgb,output)
            except (OSError,ValueError,RuntimeError) as error:
                result={'image':path.name,'chart_type':None,'status':'failed','rows':[],'errors':[str(error)]}
                prediction_folder=None
        # Première lecture de la table de référence, APRÈS la prédiction sauvegardée.
        reference=ROOT/'ChartQA Dataset'/row['split']/'tables'/(path.stem+'.csv')
        evaluation=evaluate_table(result,reference)
        record={'image':path.name,'sample_role':row['sample_role'],'reference_type':row['label'],
                'predicted_type':result['chart_type'],'status':result['status'],'errors':result['errors'],
                'extracted_values':sum(r['value'] is not None for r in result['rows']),
                'prediction_folder':str(prediction_folder) if prediction_folder else None,'evaluation':evaluation}
        records.append(record)
        print(f"{index+1}/{len(manifest)} {path.name}: {result['status']}, {record['extracted_values']} extraites, {len(evaluation['matched'])} appariées",flush=True)
    report={'protocol':{'prediction_inputs':'pixels + CV2 prediction + OCR ONLY; no annotations/tables',
             'sampling':'5 development pilot v_bar + seeded random v_bar; optional 5 exploratory h_bar',
             'matching':'normalized exact text or unique similarity >=0.92; no numerical matching',
             'warning':'Exploratory development evaluation: these images were inspected to refine rejection controls; NOT an untouched held-out benchmark. Pooled MAE/RMSE mix native units.',
             'MAPE':'matched pairs only, references !=0; percentage points remain native percentage units'},
             'overall':aggregate(records),'by_type':{t:aggregate([r for r in records if r['reference_type']==t])
                 for t in sorted({r['reference_type'] for r in records})},
             'additional_random_v_bar':aggregate([r for r in records if r['sample_role']=='additional_random_test_images']),
             'per_image':records}
    (output/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(output),'metrics':report['overall']},indent=2),flush=True)


if __name__=='__main__':
    main()
