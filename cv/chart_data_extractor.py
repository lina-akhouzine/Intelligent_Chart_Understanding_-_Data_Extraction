"""CV4 : image -> CV2/OCR/OpenCV -> tableaux. Aucune lecture des références."""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw
from bar_extractor import extract_bars,detect_rectangles,group_baseline
from chart_classifier import ROOT,load_checkpoint,load_image
from ocr_extraction import DEFAULT_CHECKPOINT,OCRExtractor,create_reader


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def cached_texts(path):
    sha=file_hash(path)
    candidates=list((ROOT/'results/cv3').glob(f'*/images/{path.stem}.json'))
    candidates+=list((ROOT/'results/cv3').glob(f'ocr_*/{path.stem}.json'))
    candidates+=list((ROOT/'results/pipeline').glob(f'run_*/cv3/{path.stem}.json'))
    candidates+=list((ROOT/'results/cv4').glob(f'*/{path.stem}/extraction.json'))
    for candidate in sorted(candidates,key=lambda p:p.stat().st_mtime_ns,reverse=True):
        try:
            data=json.loads(candidate.read_text(encoding='utf-8'))
            if data.get('source_sha256')==sha and (not data.get('errors') or 'ocr_texts' in data):
                # Ne reprendre ni rôles assistés, ni champ d'annotation, ni type de référence.
                source=data.get('texts',data.get('ocr_texts'))
                if source is not None:
                    return [{k:t[k] for k in ('text','bbox','confidence')} for t in source],str(candidate)
        except (OSError,ValueError,KeyError):
            continue
    return None,None


class ChartDataExtractor:
    def __init__(self,checkpoint=DEFAULT_CHECKPOINT,horizontal=False):
        torch.set_num_threads(4)
        self.model,self.metadata=load_checkpoint(checkpoint)
        self.checkpoint=str(checkpoint)
        self.reader=create_reader(download=False)
        self.ocr=OCRExtractor(reader=self.reader,checkpoint=None)
        self.ocr.classifier,self.ocr.classes=self.model,self.metadata['classes']
        self.horizontal=horizontal

    def extract(self,path):
        path=Path(path).resolve(); sha=file_hash(path)
        with Image.open(path) as image:
            rgb=np.asarray(image.convert('RGB'))
        with torch.inference_mode():
            probabilities=self.model(load_image(path).unsqueeze(0)).softmax(1)[0]
        chart_type=self.metadata['classes'][probabilities.argmax().item()]
        original_type=chart_type
        orientation_note=None
        if chart_type in ('v_bar','h_bar'):
            rectangles=detect_rectangles(rgb)
            vertical,_=group_baseline(rectangles,'v_bar')
            horizontal,_=group_baseline(rectangles,'h_bar')
            if chart_type=='h_bar' and len(vertical)>=max(3,len(horizontal)+3):
                chart_type='v_bar'
                orientation_note='CV2 h_bar corrigé en v_bar : davantage de rectangles partagent une base Y commune.'
            elif chart_type=='v_bar' and len(horizontal)>=max(3,len(vertical)+3):
                chart_type='h_bar'
                orientation_note='CV2 v_bar corrigé en h_bar : davantage de rectangles partagent une base X commune.'
        result={'image':path.name,'source':str(path),'source_sha256':sha,'chart_type':chart_type,
                'cv2_chart_type':original_type,'orientation_correction':orientation_note,
                'chart_probabilities':dict(zip(self.metadata['classes'],probabilities.tolist())),
                'checkpoint':self.checkpoint,'prediction_inputs':['image pixels','CV2 trained checkpoint','EasyOCR visible text'],
                'reference_data_used_for_prediction':False}
        if chart_type not in (('v_bar','h_bar') if self.horizontal else ('v_bar',)):
            result.update(status='unsupported',rows=[],bars=[],errors=[f'{chart_type}: extraction non activée/implémentée.'],uncertainties=[],calibration=None)
            return result,rgb
        texts,cache=cached_texts(path)
        if texts is None:
            print(f'OCR CV3 original : {path.name}',flush=True)
            ocr_result,_=self.ocr.extract(path,variants=('original',))
            if ocr_result['errors']:
                result.update(status='failed',rows=[],bars=[],calibration=None,errors=ocr_result['errors'],uncertainties=[])
                return result,rgb
            texts=[{k:t[k] for k in ('text','bbox','confidence')} for t in ocr_result['texts']]
        result['ocr_cache']=cache
        result['ocr_texts']=texts
        extraction=extract_bars(rgb,texts,chart_type,self.reader)
        result.update(extraction)
        if orientation_note:
            result['uncertainties'].append(orientation_note)
        valid=sum(r['value'] is not None for r in result['rows'])
        result['status']='success' if valid and valid==len(result['rows']) and not result['errors'] else 'partial' if valid else 'failed'
        result['uncertainties'].append('Rectangle plein seulement ; ombres, petites barres et objets non rectangulaires peuvent être manqués.')
        assert sha==file_hash(path),'Image source modifiée'
        return result,rgb


def dataframes(result):
    columns=['category','series','series_id','value','unit','error']
    long=pd.DataFrame(result.get('rows',[]))
    if long.empty:
        return pd.DataFrame(columns=columns),pd.DataFrame()
    long=long.reindex(columns=columns)
    known=long[long['category'].notna()].drop_duplicates(['category','series_id'])
    wide=known.pivot(index='category',columns='series_id',values='value') if not known.empty else pd.DataFrame()
    mapping=result.get('series',{})
    wide=wide.rename(columns={k:f'{v} [{k}]' for k,v in mapping.items()})
    return long,wide


def save_extraction(result,rgb,directory):
    directory=Path(directory)/Path(result['image']).stem
    directory.mkdir(parents=True,exist_ok=False)
    long,wide=dataframes(result)
    long.to_csv(directory/'data.csv',index=False,encoding='utf-8-sig')
    wide.to_csv(directory/'table.csv',encoding='utf-8-sig')
    (directory/'extraction.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    image=Image.fromarray(rgb.copy()); draw=ImageDraw.Draw(image)
    for i,bar in enumerate(result.get('bars',[])):
        box=bar['bbox']; row=result['rows'][i] if i<len(result['rows']) else None
        color='green' if row and row['value'] is not None else 'orange'
        draw.rectangle(box,outline=color,width=2)
        value=f"{row['value']:.3g}" if row and row['value'] is not None else '?'
        draw.text((box[0],max(0,box[1]-12)),f'{i}: {value}',fill='black')
    calibration=result.get('calibration')
    if calibration:
        for tick in calibration['ticks']:
            b=tick['bbox']; draw.rectangle(b,outline='blue',width=2)
        baseline=result['baseline_pixel']
        if result['chart_type']=='v_bar':
            draw.line([(0,baseline),(rgb.shape[1],baseline)],fill='purple',width=1)
        else:
            draw.line([(baseline,0),(baseline,rgb.shape[0])],fill='purple',width=1)
    image.save(directory/'detected_elements.png')
    return directory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,nargs='+',required=True)
    parser.add_argument('--horizontal',action='store_true',help='Activer aussi les barres horizontales')
    args=parser.parse_args()
    output=ROOT/'results/cv4'/datetime.now().strftime('extraction_%Y%m%d_%H%M%S_%f')
    extractor=ChartDataExtractor(horizontal=args.horizontal)
    for path in args.image:
        result,rgb=extractor.extract(path)
        folder=save_extraction(result,rgb,output)
        print(f"{path.name}: {result['chart_type']} {result['status']}, {sum(r['value'] is not None for r in result['rows'])} valeurs -> {folder}",flush=True)


if __name__=='__main__':
    main()
