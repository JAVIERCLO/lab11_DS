import argparse, json, hashlib, sys, datetime, pathlib
import numpy as np
import pandas as pd

def validate(df, cfg, source):
    checks=[]
    def check(name, condition, detail):
        checks.append({'control':name,'status':'PASS' if condition else 'FAIL','detail':detail})
    dc,vc=cfg['date_column'],cfg['value_column']
    check('columnas',set([dc,vc]).issubset(df.columns),'Columnas requeridas: Fecha, Viajero')
    if checks[-1]['status']=='FAIL': return checks,None
    dates=pd.to_datetime(df[dc],errors='coerce'); vals=pd.to_numeric(df[vc],errors='coerce')
    check('tipos',dates.notna().all() and (vals.notna() | df[vc].isna()).all(),'Fechas parseables y valores numericos')
    check('fechas_y_cobertura',dates.notna().all() and set(pd.date_range(cfg['expected_start'],cfg['expected_end'],freq='MS')).issubset(set(dates.dropna())) and dates.dropna().dt.is_month_start.all(),'Cobertura mensual configurada y fechas validas')
    check('unicidad',not dates.duplicated().any(),'Una fila por mes')
    check('faltantes',not vals.isna().any(),'Sin valores ausentes en meses requeridos')
    check('valores_y_denominadores',bool(np.isfinite(vals.dropna()).all() and (vals.dropna()>=0).all() and cfg['distribution_factor']>0),'Conteos no negativos y factor de reparto positivo')
    check('regla_metodologica',pathlib.Path(source).name=='serie_total_mensual.csv' and len(df)==len(pd.date_range(cfg['expected_start'],cfg['expected_end'],freq='MS')),'Fuente mensual declarada; no sumar presencias diarias')
    return checks,pd.DataFrame({'Fecha':dates,'Viajero':vals}).sort_values('Fecha')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='config/config.json');ap.add_argument('--source',default=None);a=ap.parse_args()
    root=pathlib.Path(__file__).resolve().parents[1];cfg=json.loads((root/a.config).read_text());src=root/(a.source or cfg['source']);out=root/'outputs';out.mkdir(exist_ok=True)
    run=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');report={'run_id':run,'source':str(src.relative_to(root)),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        df=pd.read_csv(src);report['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest();checks,clean=validate(df,cfg,src)
        report['checks']=checks
        if any(c['status']=='FAIL' for c in checks):raise ValueError('Controles criticos fallidos')
        s=clean.set_index('Fecha')['Viajero'];ev=s.loc[cfg['evaluation_start']:cfg['evaluation_end']];train=s.loc[:cfg['train_end']]
        if ev.empty or len(train)<12:raise ValueError('Periodo de evaluacion o ajuste insuficiente')
        if ev.index.min()<=train.index.max():raise ValueError('Fuga temporal: periodos solapados')
        # Backtesting rolling origin: cada mes solo usa meses anteriores
        pred=[];base=[]
        for dt in ev.index:
            prior=s.loc[s.index<dt];prev=dt-pd.DateOffset(years=1)
            if prev not in prior.index:raise ValueError('Falta mes comparable previo')
            pred.append(float(prior.loc[prev]));base.append(float(prior.iloc[-1]))
        mae=float(np.mean(np.abs(ev.to_numpy()-pred)));baseline=float(np.mean(np.abs(ev.to_numpy()-base)))
        report['metrics']={'mae':mae,'baseline_mae':baseline,'evaluated_months':len(ev)}
        last=s.index.max();future=pd.date_range(last+pd.offsets.MonthBegin(1),periods=cfg['forecast_months'],freq='MS')
        records=[]
        for dt in future:
            prev=dt-pd.DateOffset(years=1)
            if prev not in s.index:raise ValueError('No hay dato estacional para '+str(dt))
            records.append({'destino':cfg['destination'],'periodo_estimado':dt.strftime('%Y-%m-%d'),'valor':float(s.loc[prev])*cfg['distribution_factor'],'unidad':cfg['unit'],'run_id':run,'fecha_generacion':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version_procedimiento':cfg['method']})
        candidate=out/('candidate_'+run+'.csv');pd.DataFrame(records).to_csv(candidate,index=False)
        accepted=mae<=cfg['max_mae'] and mae<=baseline*cfg['max_ratio_to_baseline'] and all(np.isfinite(r['valor']) and r['valor']>=0 for r in records)
        report['accepted']=bool(accepted)
        if accepted:
            # Solo reemplazar al aprobar, con respaldo de la version anterior
            approved=out/'approved.csv'
            if approved.exists():approved.replace(out/('approved_previous_'+run+'.csv'))
            candidate.replace(approved);report['publication']='approved.csv actualizado'
        else:report['publication']='Rechazado; se conserva approved.csv si existe'
    except Exception as e:
        report['error']=str(e);report['accepted']=False;report['publication']='No publicado'
    report['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out/('report_'+run+'.json')).write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False));return 0 if report['accepted'] else 1
if __name__=='__main__':sys.exit(main())
