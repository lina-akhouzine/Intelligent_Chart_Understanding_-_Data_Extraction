"""Extraction de rectangles et calibration depuis les pixels/OCR, sans référence ChartQA."""
from __future__ import annotations
import itertools
import re
import unicodedata
import cv2
import numpy as np
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


def parse_number(text):
    """Lecture stricte ; aucune correction devinée d'un chiffre OCR ambigu."""
    text = unicodedata.normalize('NFKC', str(text)).strip().replace('−','-').replace('–','-')
    text = text.replace('\u00a0',' ').replace(' ','')
    match = re.fullmatch(r'([$€£]?)([+-]?[\d,.]+)(%|k|K|million|billion|thousand)?',text)
    if not match:
        return None
    raw, suffix = match[2], match[3] or ''
    if ',' in raw:
        if re.fullmatch(r'[+-]?\d{1,3}(,\d{3})+(\.\d+)?',raw):
            raw = raw.replace(',','')
        elif raw.count(',')==1 and '.' not in raw and len(raw.split(',')[1]) in (1,2):
            raw = raw.replace(',','.')
        else:
            return None
    try:
        value = float(raw)
    except ValueError:
        return None
    factor = {'k':1e3,'K':1e3,'thousand':1e3,'million':1e6,'billion':1e9}.get(suffix,1)
    return {'value':value*factor,'unit':'percent' if suffix=='%' else match[1] or None}


def detect_rectangles(rgb):
    """Composantes de couleurs pleines : rectangles colorés ou gris, hors petites icônes."""
    height,width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    eligible = ((hsv[:,:,1]>45)&(hsv[:,:,2]>25)) | ((hsv[:,:,1]<=45)&(hsv[:,:,2]>20)&(hsv[:,:,2]<205))
    quantized = (rgb//16).astype(np.int32)
    codes = quantized[:,:,0]*256+quantized[:,:,1]*16+quantized[:,:,2]
    unique, counts = np.unique(codes[eligible],return_counts=True)
    rectangles=[]
    for code in unique[np.argsort(counts)[::-1]][:30]:
        mask = np.uint8((codes==code)&eligible)*255
        mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
        for index in range(1,count):
            x,y,w,h,area = map(int,stats[index])
            if (w<4 or h<4 or max(w,h)<23 or area<max(70,width*height*.00015) or
                area/(w*h)<.82 or w>width*.8 or h>height*.88 or
                x<width*.025 or x+w>width*.96 or y<height*.02 or y+h>height*.90):
                continue
            color = np.median(rgb[labels==index],axis=0).astype(int).tolist()
            rectangles.append({'bbox':[x,y,x+w,y+h],'color':color,'fill_ratio':float(area/(w*h))})
    return rectangles


def group_baseline(rectangles, orientation):
    dimension = 1 if orientation=='v_bar' else 0
    best=[]; baseline=None
    for rectangle in rectangles:
        for edge in (rectangle['bbox'][dimension],rectangle['bbox'][dimension+2]):
            members=[r for r in rectangles if min(abs(r['bbox'][dimension]-edge),abs(r['bbox'][dimension+2]-edge))<=3]
            if len(members)>len(best):
                best=members; baseline=float(edge)
    return best,baseline


def has_stacks(rectangles,orientation):
    d=0 if orientation=='v_bar' else 1
    v=1-d
    for a,b in itertools.combinations(rectangles,2):
        aa,bb=a['bbox'],b['bbox']
        overlap=max(0,min(aa[d+2],bb[d+2])-max(aa[d],bb[d]))
        fraction=overlap/max(1,min(aa[d+2]-aa[d],bb[d+2]-bb[d]))
        if fraction>.8 and min(abs(aa[v+2]-bb[v]),abs(bb[v+2]-aa[v]))<=3:
            return True
    return False


def axis_candidates(texts,bars,orientation):
    left=min(r['bbox'][0] for r in bars); right=max(r['bbox'][2] for r in bars)
    top=min(r['bbox'][1] for r in bars); bottom=max(r['bbox'][3] for r in bars)
    candidates=[]
    for item in texts:
        box=item['bbox']; cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
        number=parse_number(item['text'])
        if number is None or item.get('confidence',0)<.35:
            continue
        if orientation=='v_bar' and box[2]<=left-2 and 0<=cy<=bottom+25:
            candidates.append({**number,'position':cy,'text':item['text'],'bbox':box})
        elif orientation=='h_bar' and box[1]>=bottom+2 and left-40<=cx<=right+50:
            candidates.append({**number,'position':cx,'text':item['text'],'bbox':box})
    # Garder la colonne/la rangée de graduations la plus cohérente spatialement.
    if orientation=='v_bar' and candidates:
        best=max(([t for t in candidates if abs(t['bbox'][2]-a['bbox'][2])<12] for a in candidates),key=len)
        candidates=best
    return candidates


def calibrate(ticks,orientation):
    """Consensus linéaire >=3 valeurs distinctes. Rejette non-linéarité/ambiguïté."""
    unique=[]
    for tick in sorted(ticks,key=lambda t:t['position']):
        if not any(abs(tick['position']-t['position'])<4 for t in unique):
            unique.append(tick)
    if len(unique)<3:
        return None,'Moins de trois graduations numériques distinctes localisées.'
    p=np.array([t['position'] for t in unique]); values=np.array([t['value'] for t in unique])
    hypotheses=[]
    for i,j in itertools.combinations(range(len(unique)),2):
        if abs(p[i]-p[j])<15 or values[i]==values[j]:
            continue
        slope=(values[i]-values[j])/(p[i]-p[j])
        if (orientation=='v_bar' and slope>=0) or (orientation=='h_bar' and slope<=0):
            continue
        intercept=values[i]-slope*p[i]
        tolerance=max(abs(values[i]-values[j])*.025,abs(slope)*2,1e-7)
        mask=np.abs(slope*p+intercept-values)<=tolerance
        if mask.sum()<3 or len(set(values[mask]))<3 or np.ptp(p[mask])<.4*np.ptp(p):
            continue
        hypotheses.append((int(mask.sum()),slope,intercept,mask))
    if not hypotheses:
        return None,'Échelle non linéaire, chiffres OCR incohérents ou calibration ambiguë.'
    hypotheses.sort(key=lambda h:h[0],reverse=True)
    support,_,_,mask=hypotheses[0]
    if support/max(1,len(unique))<.75:
        return None,'Trop de graduations incompatibles : échelle non linéaire ou OCR insuffisant.'
    slope,intercept=np.polyfit(p[mask],values[mask],1)
    competitors=[h for h in hypotheses if h[0]>=support and abs(h[1]-slope)>abs(slope)*.1]
    if competitors:
        return None,'Plusieurs calibrations linéaires concurrentes.'
    units={unique[i]['unit'] for i in range(len(unique)) if mask[i] and unique[i]['unit']}
    if len(units)>1:
        return None,'Unités de graduations contradictoires.'
    rmse=float(np.sqrt(np.mean((slope*p[mask]+intercept-values[mask])**2)))
    return {'slope':float(slope),'intercept':float(intercept),'unit':next(iter(units),None),
            'rmse_ticks':rmse,'ticks':[unique[i] for i in range(len(unique)) if mask[i]],
            'rejected_ticks':[unique[i] for i in range(len(unique)) if not mask[i]],
            'mapping':'value = slope * pixel_coordinate + intercept','scale':'linear'},None


def roi_ocr(reader,rgb,box,numeric_only=False):
    x1,y1,x2,y2=map(int,box)
    height,width=rgb.shape[:2]
    x1,y1,x2,y2=max(0,x1),max(0,y1),min(width,x2),min(height,y2)
    if x2-x1<8 or y2-y1<8:
        return []
    options={'detail':1,'paragraph':False,'workers':0,'min_size':3,'mag_ratio':2.,'rotation_info':None}
    if numeric_only:
        options['allowlist']='0123456789.,%$€£-+kKmillionbthousad '
    raw=reader.readtext(cv2.cvtColor(rgb[y1:y2,x1:x2],cv2.COLOR_RGB2BGR),**options)
    texts=[]
    for polygon,text,confidence in raw:
        points=[[float(x+x1),float(y+y1)] for x,y in polygon]
        texts.append({'text':str(text),'confidence':float(confidence),
                      'bbox':[min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points)],
                      'source':'CV4 targeted OCR, no reference text'})
    return texts


def category_labels(texts,bars,orientation):
    left=min(r['bbox'][0] for r in bars); right=max(r['bbox'][2] for r in bars)
    top=min(r['bbox'][1] for r in bars); bottom=max(r['bbox'][3] for r in bars)
    candidates=[]
    for item in texts:
        b=item['bbox']; x=(b[0]+b[2])/2; y=(b[1]+b[3])/2
        if item.get('confidence',0)<.45 or any(s in item['text'].casefold() for s in ('statista','show source','additional information')):
            continue
        if orientation=='v_bar' and bottom+2<=y<=bottom+95 and left-35<=x<=right+35:
            candidates.append({**item,'position':x,'row':y})
        elif orientation=='h_bar' and b[2]<=left-2 and top-12<=y<=bottom+12:
            candidates.append({**item,'position':y,'row':x})
    if orientation=='v_bar' and candidates:
        first=min(t['row'] for t in candidates)
        candidates=[t for t in candidates if t['row']<=first+22]
    # Même catégorie sur plusieurs lignes : regrouper seulement si les centres se superposent.
    groups=[]
    for item in sorted(candidates,key=lambda t:(t['row'],t['position'])):
        group=next((g for g in groups if abs(g['position']-item['position'])<12),None)
        if group:
            if item['text'] not in group['text']:
                group['text']+=' '+item['text']
        else:
            groups.append(dict(item))
    return sorted(groups,key=lambda t:t['position'])


def legend_series(rgb,texts,bars):
    """Associer un nom uniquement si un marqueur visible a la couleur de la série."""
    height,width=rgb.shape[:2]
    top=min(b['bbox'][1] for b in bars); bottom=max(b['bbox'][3] for b in bars)
    colors=[]
    for bar in bars:
        color=np.array(bar['color'])
        index=next((i for i,c in enumerate(colors) if np.linalg.norm(color-c)<25),None)
        if index is None:
            colors.append(color)
            index=len(colors)-1
        bar['series_id']=f'series_{index+1}'
    names={}
    for index,color in enumerate(colors):
        mask=np.uint8(np.linalg.norm(rgb.astype(float)-color,axis=2)<30)*255
        n,_,stats,_=cv2.connectedComponentsWithStats(mask,8)
        pairs=[]
        for x,y,w,h,area in stats[1:]:
            if not (4<=w<=27 and 4<=h<=27 and area>12 and y<height*.88 and (y>bottom+8 or y+h<top-8)):
                continue
            for item in texts:
                b=item['bbox']
                if x+w-3<=b[0]<=x+w+45 and abs((b[1]+b[3])/2-(y+h/2))<12 and item.get('confidence',0)>.5:
                    pairs.append((abs(b[0]-x-w),item['text']))
        if pairs:
            names[f'series_{index+1}']=min(pairs)[1]
    return names


def percent_glyph_evidence(rgb,ticks):
    """Contrôle de rejet uniquement : gabarits du glyphe %, jamais correction de chiffres."""
    font_path=Path('C:/Windows/Fonts/arial.ttf')
    if not font_path.is_file():
        return []
    gray=255-cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    templates=[]
    for size in range(8,19):
        font=ImageFont.truetype(str(font_path),size)
        b=font.getbbox('%')
        tile=Image.new('L',(b[2]-b[0]+2,b[3]-b[1]+2))
        ImageDraw.Draw(tile).text((1-b[0],1-b[1]),'%',font=font,fill=255)
        templates.append(np.asarray(tile))
    evidence=[]
    for tick in ticks:
        x1,y1,x2,y2=map(int,tick['bbox'])
        patch=gray[max(0,y1-2):y2+2,max(0,x1-2):x2+2]
        best=-1.
        for template in templates:
            if template.shape[0]<=patch.shape[0] and template.shape[1]<=patch.shape[1]:
                matched=cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED)
                suffix=matched[:,max(0,patch.shape[1]//2-template.shape[1]//2):]
                if suffix.size:
                    best=max(best,float(suffix.max()))
        evidence.append({'ocr_text':tick['text'],'percent_template_correlation':best})
    return evidence


def extract_bars(rgb,texts,orientation,reader=None):
    warnings=[]; errors=[]; refinements=[]
    rectangles=detect_rectangles(rgb)
    if not rectangles:
        return {'rows':[],'bars':[],'errors':['Aucun rectangle de barre fiable.'],'uncertainties':warnings,'calibration':None}
    bars,baseline=group_baseline(rectangles,orientation)
    if not bars:
        return {'rows':[],'bars':[],'errors':['Aucune base de barres cohérente.'],'uncertainties':warnings,'calibration':None}
    if has_stacks(rectangles,orientation):
        return {'rows':[],'bars':bars,'errors':['Barres empilées détectées : extraction refusée, pas de décomposition implémentée.'],
                'uncertainties':warnings,'calibration':None}
    ticks=axis_candidates(texts,bars,orientation)
    calibration,reason=calibrate(ticks,orientation)
    height,width=rgb.shape[:2]
    left=min(b['bbox'][0] for b in bars); right=max(b['bbox'][2] for b in bars)
    top=min(b['bbox'][1] for b in bars); bottom=max(b['bbox'][3] for b in bars)
    if reader is not None:
        axis_box=[max(0,left-110),max(0,top-40),left-3,bottom+25] if orientation=='v_bar' else [left-40,bottom+3,right+55,min(height,bottom+50)]
        refined=roi_ocr(reader,rgb,axis_box,True)
        refinements.extend(refined)
        refined_ticks=axis_candidates(refined,bars,orientation)
        new_calibration,new_reason=calibrate(refined_ticks,orientation)
        if new_calibration is not None:
            calibration,reason=new_calibration,None
        elif calibration is None:
            reason=new_reason
        label_box=[left-40,bottom+3,right+40,min(height*.9,bottom+85)] if orientation=='v_bar' else [max(0,left-300),top-15,left-3,bottom+15]
        refined_labels=roi_ocr(reader,rgb,label_box,False)
        refinements.extend(refined_labels)
    else:
        refined_labels=[]
    labels=category_labels(refined_labels or texts,bars,orientation)
    series=legend_series(rgb,texts,bars)
    # Contrôle indépendant depuis des étiquettes de valeur visibles, sans en faire
    # des vérités terrain : le symbole % est souvent lu « 70 » par EasyOCR.
    if calibration is not None:
        percent_checks=[]
        for item in texts:
            number=parse_number(item['text'])
            if number is None or number['unit']!='percent' or item.get('confidence',0)<.4:
                continue
            box=item['bbox']; cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
            for bar in bars:
                b=bar['bbox']
                if orientation=='v_bar' and b[0]-8<=cx<=b[2]+8 and 0<=b[1]-cy<=28:
                    predicted=calibration['slope']*b[1]+calibration['intercept']
                    percent_checks.append(abs(predicted-number['value'])/max(abs(number['value']),1))
        if len(percent_checks)>=2:
            if float(np.median(percent_checks))>.35:
                calibration=None
                reason='Calibration en contradiction avec plusieurs pourcentages visibles ; chiffres/symbole % OCR ambigus.'
            else:
                calibration['unit']='percent'
        if calibration is not None and calibration['unit'] is None:
            glyphs=percent_glyph_evidence(rgb,calibration['ticks'])
            suspicious=sum(g['percent_template_correlation']>=.50 for g in glyphs)
            if suspicious>=3 and suspicious>=.6*len(glyphs):
                calibration=None
                reason='Symboles % visuellement probables mais non reconnus par OCR : calibration refusée, aucun chiffre corrigé.'
                warnings.append('Contrôle conservateur par gabarit Arial % (Windows), susceptible de faux rejets.')
    if calibration is None:
        errors.append(reason)
    if not labels:
        errors.append('Aucune catégorie OCR spatialement associable.')
    if len({b['series_id'] for b in bars})>1 and len(series)<len({b['series_id'] for b in bars}):
        warnings.append('Noms de séries incomplets : identifiants de couleur conservés sans nom inventé.')
    rows=[]
    d=1 if orientation=='v_bar' else 0
    cat_dim=1-d
    positions=[t['position'] for t in labels]
    spacing=float(np.median(np.diff(positions))) if len(positions)>1 else (right-left if orientation=='v_bar' else bottom-top)
    for bar in bars:
        b=bar['bbox']; center=(b[cat_dim]+b[cat_dim+2])/2
        label=min(labels,key=lambda t:abs(t['position']-center)) if labels else None
        if label is not None and abs(label['position']-center)>max(25,spacing*.48):
            label=None
        endpoint=float(b[d] if abs(b[d]-baseline)>abs(b[d+2]-baseline) else b[d+2])
        value=None; issue=None
        if calibration is None:
            issue='calibration_failed'
        elif label is None:
            issue='category_missing'
        else:
            low=min(t['position'] for t in calibration['ticks']); high=max(t['position'] for t in calibration['ticks'])
            allowance=max(8,(high-low)*.12)
            if not low-allowance<=endpoint<=high+allowance:
                issue='endpoint_outside_validated_scale'
            else:
                value=float(calibration['slope']*endpoint+calibration['intercept'])
        if issue:
            warnings.append(f'{bar["series_id"]} {label["text"] if label else "catégorie inconnue"}: {issue}')
        rows.append({'category':label['text'] if label else None,'series':series.get(bar['series_id']),
                     'series_id':bar['series_id'],'value':value,'unit':calibration['unit'] if calibration else None,
                     'bar_bbox':b,'color':bar['color'],'endpoint_pixel':endpoint,'error':issue,
                     'pixel_value_uncertainty':abs(calibration['slope']) if calibration else None})
    # Deux rectangles attribués à la même cellule sont ambigus, jamais additionnés.
    keys={}
    for row in rows:
        key=(row['category'],row['series_id'])
        keys.setdefault(key,[]).append(row)
    for group in keys.values():
        if len(group)>1:
            for row in group:
                row['value']=None; row['error']='ambiguous_multiple_bars_for_cell'
            warnings.append('Plusieurs rectangles pour une même catégorie/série : valeurs laissées manquantes.')
    return {'rows':rows,'bars':bars,'categories':labels,'series':series,'calibration':calibration,
            'baseline_pixel':baseline,'baseline_value':float(calibration['slope']*baseline+calibration['intercept']) if calibration else None,
            'refined_ocr':refinements,'errors':errors,'uncertainties':warnings}
