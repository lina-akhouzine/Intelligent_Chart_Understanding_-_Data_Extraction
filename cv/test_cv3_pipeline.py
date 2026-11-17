"""Vérifications CV3 réelles : exports, boîte/confiance, image blanche et image invalide."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
from chart_text_parser import annotation_references
from ocr_extraction import ROOT, OCRExtractor, create_reader, deduplicate, save_result
from evaluate_ocr import edit_distance, score_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path)
    args = parser.parse_args()
    folders = sorted((ROOT/'results/cv3').glob('evaluation_*/metrics.json'))
    folder = args.evaluation or folders[-1].parent
    metrics = json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
    protected = [ROOT/'cv/preprocessing.py', ROOT/'cv/README_CV1.md', ROOT/'cv/README_CV2.md',
                 ROOT/'models_cv2/resnet18_20261117_010609_069660.pt']
    protected += [ROOT/'cv'/name for name in ('chart_classifier.py','prepare_cv2.py','train_classifier.py',
                  'evaluate_classifier.py','test_cv2_pipeline.py','prepare_official_cv2.py',
                  'fetch_official_annotations.py','inspect_cv2_results.py','report_cv2.py')]
    before = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    exports = sorted((folder/'images').glob('*.json'))
    assert len(exports) == metrics['images'] >= 20
    total_texts = 0
    for path in exports:
        result = json.loads(path.read_text(encoding='utf-8'))
        assert not result['errors'], result['errors']
        assert result['chart_type'] in ('v_bar','h_bar','line','pie')
        assert hashlib.sha256(Path(result['source']).read_bytes()).hexdigest() == result['source_sha256']
        with Image.open(path.with_name(path.stem+'_boxes.png')) as image:
            assert image.size == (result['width'],result['height'])
        for text in result['texts']:
            box = text['bbox']
            assert len(box)==4 and all(math.isfinite(v) for v in box)
            assert 0 <= box[0] < box[2] <= result['width']
            assert 0 <= box[1] < box[3] <= result['height']
            assert 0 <= text['confidence'] <= 1 and text['text'].strip()
            assert text['role_source'] == 'geometry'
            if result['chart_type'] == 'pie':
                assert text['role'] not in ('x_tick','y_tick','x_category','y_category','x_axis_title','y_axis_title')
        total_texts += len(result['texts'])
    sample = {'text':'2010','confidence':.9,'bbox':[0,0,20,10],
              'polygon':[[0,0],[20,0],[20,10],[0,10]],'variant':'original'}
    combined = deduplicate([sample,{**sample,'confidence':.8,'variant':'clahe'},
                            {**sample,'bbox':[30,0,50,10]}])
    assert len(combined)==2 and len(combined[0]['variants'])==2
    refs, issues = annotation_references({'type':'v_bar','general_figure_info':{
        'y_axis':{'major_labels':{'bboxes':[{'x':1,'y':1,'w':10,'h':10}]}}},
        'models':[{'y':[999]}]})
    assert len(refs)==1 and refs[0]['text'] is None and issues
    assert edit_distance('abc','adc')==1 and edit_distance(['one','two'],['one'])==1
    counts,_,_ = score_image([{'text':'abc','bbox':[0,0,20,10],'role':'title','field':'fixture'}],[],100,100)
    assert counts['character_edits']==3 and counts['word_edits']==1
    destination = ROOT/'results/cv3'/datetime.now().strftime('tests_%Y%m%d_%H%M%S_%f')
    destination.mkdir()
    blank = destination/'blank.png'
    Image.fromarray(np.full((300,400,3),255,dtype=np.uint8)).save(blank)
    extractor = OCRExtractor(reader=create_reader(),checkpoint=None)
    result, original = extractor.extract(blank)
    assert result['texts']==[] and not result['errors']
    save_result(result,original,destination/'blank_output')
    invalid = destination/'invalid.png'
    invalid.write_bytes(b'CV3 invalid image test')
    try:
        extractor.extract(invalid)
    except (OSError, ValueError):
        pass
    else:
        raise AssertionError('Image invalide acceptée')
    after = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    assert before==after
    report = {'status':'PASS','evaluation':str(folder),'images_verified':len(exports),
              'ocr_texts_verified':total_texts,'blank_image_real_engine':'PASS',
              'invalid_image_rejected':'PASS','deduplication':'PASS',
              'missing_transcripts_not_invented':'PASS','edit_distance_checks':'PASS',
              'protected_files_unchanged_during_tests':before}
    (destination/'test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='protected_files_unchanged_during_tests'},indent=2))


if __name__ == '__main__':
    main()
