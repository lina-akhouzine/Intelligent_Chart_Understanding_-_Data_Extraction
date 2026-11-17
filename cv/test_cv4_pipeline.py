"""Tests CV4 : géométrie connue synthétique + contrôle des vraies sorties ChartQA."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from bar_extractor import calibrate,extract_bars,parse_number
from chart_data_extractor import ROOT,dataframes
from evaluate_extraction import evaluate_table,aggregate


def item(text,x,y,w=30,h=12):
    return {'text':text,'bbox':[x-w/2,y-h/2,x+w/2,y+h/2],'confidence':1.}


def synthetic_cases():
    image=np.full((400,500,3),255,dtype=np.uint8)
    image[175:300,140:180]=[30,100,210]
    image[225:300,260:300]=[30,100,210]
    labels=[item('A',160,320),item('B',280,320)]
    ticks=[item(str(v),80,y) for y,v in [(100,100),(200,50),(300,0)]]
    result=extract_bars(image,ticks+labels,'v_bar')
    values={r['category']:r['value'] for r in result['rows']}
    assert abs(values['A']-62.5)<1 and abs(values['B']-37.5)<1,values
    nonzero=extract_bars(image,[item(str(v),80,y) for y,v in [(100,200),(200,150),(300,100)]]+labels,'v_bar')
    assert abs(nonzero['baseline_value']-100)<1
    assert min(r['value'] for r in nonzero['rows'])>100
    negative=np.full_like(image,255)
    negative[150:200,140:180]=[30,100,210]
    negative[200:280,260:300]=[30,100,210]
    neg=extract_bars(negative,[item(str(v)+'%',80,y) for y,v in [(100,100),(200,0),(300,-100)]]+labels,'v_bar')
    negative_values={r['category']:r['value'] for r in neg['rows']}
    assert abs(negative_values['A']-50)<1 and abs(negative_values['B']+80)<1,negative_values
    assert all(r['unit']=='percent' for r in neg['rows'])
    grouped=np.full((450,500,3),255,dtype=np.uint8)
    blue,red=[30,100,210],[220,40,50]
    for x,y,color in [(125,175,blue),(165,225,red),(245,200,blue),(285,250,red)]:
        grouped[y:300,x:x+25]=color
    grouped[356:364,140:148]=blue; grouped[356:364,240:248]=red
    multi=extract_bars(grouped,ticks+[item('A',157,320),item('B',277,320),
          item('Blue',182,360,50),item('Red',277,360,40)],'v_bar')
    assert len(multi['rows'])==4 and all(r['value'] is not None for r in multi['rows'])
    assert {r['series'] for r in multi['rows']}=={'Blue','Red'},multi['series']
    # Même géométrie transposée : vérifie la conversion horizontale.
    horizontal=np.transpose(image,(1,0,2)).copy()
    ht=[item(str(v),y,80) for y,v in [(100,100),(200,50),(300,0)]]
    # L'axe horizontal doit être croissant vers la droite, changer les valeurs.
    ht=[item(str(v),x,320) for x,v in [(100,0),(200,50),(300,100)]]
    horizontal=np.full((400,500,3),255,dtype=np.uint8)
    horizontal[140:175,100:225]=blue; horizontal[240:275,100:275]=blue
    h=extract_bars(horizontal,ht+[item('A',70,157,20),item('B',70,257,20)],'h_bar')
    assert len(h['rows'])==2 and all(r['value'] is not None for r in h['rows']),h
    nonlinear=[{'position':p,'value':v,'text':str(v),'unit':None,'bbox':[0,p,10,p+5]}
               for p,v in [(100,1000),(200,100),(300,10),(400,1)]]
    assert calibrate(nonlinear,'v_bar')[0] is None
    blank=extract_bars(np.full_like(image,255),[],'v_bar')
    assert not blank['rows'] and blank['errors']
    stacked=np.full_like(image,255)
    stacked[150:225,140:180]=blue; stacked[225:300,140:180]=red
    stack=extract_bars(stacked,ticks+labels,'v_bar')
    assert not stack['rows'] and any('empilées' in e for e in stack['errors'])
    assert parse_number('−12.5%')['value']==-12.5
    assert parse_number('1,234.5')['value']==1234.5
    assert parse_number('4-5%') is None and parse_number('not a number') is None
    return {'linear_geometry':'PASS','nonzero_origin':'PASS','negative_values':'PASS',
            'percent_units':'PASS','grouped_color_legend':'PASS','horizontal_geometry':'PASS',
            'nonlinear_rejected':'PASS','stacked_rejected':'PASS','blank_rejected':'PASS','strict_numeric_parser':'PASS'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path)
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args()
    protected=[p for p in (ROOT/'cv').glob('*') if p.is_file() and p.name not in
               {'bar_extractor.py','chart_data_extractor.py','evaluate_extraction.py','test_cv4_pipeline.py','README_CV4.md'}]
    protected += [ROOT/'run_all.py',ROOT/'models_cv2/resnet18_20261117_010609_069660.pt']
    protected=[p for p in protected if p.is_file()]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    report=synthetic_cases()
    destination=ROOT/'results/cv4'/datetime.now().strftime('tests_%Y%m%d_%H%M%S_%f')
    destination.mkdir(parents=True)
    # Référence SYNTHÉTIQUE indépendante : jamais une table ChartQA produite en sortie.
    reference=destination/'synthetic_evaluation_reference.csv'
    pd.DataFrame({'category':['A','B'],'value':[1,0]}).to_csv(reference,index=False)
    fixture={'rows':[{'category':c,'series':None,'series_id':'series_1','value':v}
                     for c,v in [('A',1),('B',0),('EXTRA',1000)]]}
    scored=evaluate_table(fixture,reference)
    stats=aggregate([{'evaluation':scored,'extracted_values':3}])
    assert stats['MAE']==0 and stats['RMSE']==0 and stats['extra_values']==1
    assert stats['MAPE_nonzero_pairs']==1 and stats['MAPE_nonzero_percent']==0
    report['unmatched_not_used_for_numeric_error']='PASS'
    report['MAPE_zero_exclusion']='PASS'
    if not args.synthetic_only:
        folder=args.evaluation or sorted((ROOT/'results/cv4').glob('evaluation_*/metrics.json'))[-1].parent
        metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        assert metrics['overall']['images']>=20
        checked=0; populated=0
        for entry in metrics['per_image']:
            if not entry['prediction_folder']:
                continue
            path=Path(entry['prediction_folder'])
            result=json.loads((path/'extraction.json').read_text(encoding='utf-8'))
            assert result['reference_data_used_for_prediction'] is False
            assert hashlib.sha256(Path(result['source']).read_bytes()).hexdigest()==result['source_sha256']
            data=pd.read_csv(path/'data.csv')
            assert len(data)==len(result['rows'])
            assert int(pd.to_numeric(data['value'],errors='coerce').notna().sum())==entry['extracted_values']
            if result['calibration'] is None:
                assert all(r['value'] is None for r in result['rows'])
            with Image.open(path/'detected_elements.png') as image:
                assert image.width>0 and image.height>0
            checked+=1; populated+=int(entry['extracted_values']>0)
        assert checked>=20
        report.update(real_images_verified=checked,images_with_values=populated,evaluation=str(folder))
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before==after
    report.update(status='PASS',protected_files_unchanged=before)
    (destination/'test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='protected_files_unchanged'},indent=2),flush=True)


if __name__=='__main__':
    main()
