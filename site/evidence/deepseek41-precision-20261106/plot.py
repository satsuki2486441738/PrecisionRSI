from pathlib import Path
import argparse,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
parser=argparse.ArgumentParser(description='Render allowlisted core measurements; does not replay raw numerical captures');parser.add_argument('--measurements',type=Path,default=Path(__file__).with_name('core-measurements.json'));parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True);source=args.measurements;d=json.loads(source.read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue='#245A81';teal='#288C85';gold='#D89A32';red='#BD5046';grey='#D9E0E5';ink='#1D2A36'
fig,axs=plt.subplots(2,2,figsize=(13,9.5));fig.subplots_adjust(left=.10,right=.91,top=.84,bottom=.13,wspace=.45,hspace=.53)
fig.suptitle('DeepSeek-V4.1-Flash-1017 | selected numerical diagnostics',x=.10,ha='left',y=.965,fontsize=18,fontweight='bold',color=ink)
fig.text(.10,.915,'Pinned release 1bae0e10 · GPU original mixed-format arithmetic vs NPU adapted BF16, TP8 / EP8 / offload48',fontsize=10,color=ink)
fig.text(.10,.891,'Descriptive results; configuration effects remain mixed. Numerical acceptance thresholds were not supplied.',fontsize=10,color='#53616C')
a=axs[0,0];rows=d['coverage'];labels=[x['stage'].replace('fixed-decode','Fixed teacher32').replace('repeatability','Repeat controls').replace('inference','Inference').replace('structured','Structured').replace('padding','Masked padding').replace('blocks','mHC layers')for x in rows];y=np.arange(len(rows));h=.22
for offset,key,color,label in [(-h,'gpu_missing',blue,'GPU retained'),(0,'npu_missing',teal,'NPU retained'),(h,'paired',gold,'Paired')]:
 counts=[x['expected']-x[key] if key!='paired' else x[key] for x in rows];fractions=[n/x['expected'] for n,x in zip(counts,rows)]
 a.barh(y+offset,fractions,height=h,color=color,label=label)
 for yi,f,n,row in zip(y+offset,fractions,counts,rows):a.text(max(.02,f+.015),yi,str(n)+'/'+str(row['expected']),va='center',fontsize=8,color=ink)
a.set_yticks(y,labels);a.invert_yaxis();a.set_xlim(0,1.31);a.set_xticks([0,.5,1],['0%','50%','100%']);a.set_title('A  Strict tensor inventory (275 cells)',loc='left',fontweight='bold',pad=12);a.set_xlabel('Retained fraction of expected inventory cells');a.legend(ncol=3,loc='upper left',bbox_to_anchor=(-.03,-.19),frameon=False,fontsize=8,columnspacing=1)
a=axs[0,1];selected=[next(x for x in d['selected_position_agreement']if (x['stage'],x['mode'])==key)for key in [('inference','full'),('inference','token'),('fixed-decode','teacher32'),('structured','full')]]
names=['Full forward','Token cache','Fixed teacher32','Structured full'];left=np.zeros(4)
for key,color,label in [('top1_equal',blue,'Same top-1 ID'),('disagreements_with_tie_intersection',gold,'Different ID, tied-set overlap'),('disagreements_with_disjoint_tied_sets',red,'Disjoint tied sets')]:
 vals=np.array([x[key]/x['positions']for x in selected]);a.barh(np.arange(4),vals,left=left,height=.54,color=color,label=label);left+=vals
for i,x in enumerate(selected):a.text(1.02,i,str(x['top1_equal'])+'/'+str(x['positions']),va='center',fontsize=9,color=ink)
a.set_yticks(range(4),names);a.invert_yaxis();a.set_xlim(0,1.27);a.set_xticks([0,.5,1],['0%','50%','100%']);a.set_title('B  Selected positions: ties change interpretation',loc='left',fontweight='bold',pad=12);a.set_xlabel('Selected positions; repeated tokens are correlated');a.legend(loc='upper left',bbox_to_anchor=(-.02,-.19),frameon=False,fontsize=8)
a=axs[1,0];groups=[d['target_groups'][k]for k in ['quality','long-targets','structured']];x=np.arange(3);width=.22
for dx,key,color,label in [(-width,'mae_nat',teal,'MAE'),(0,'p99_nat',gold,'P99'),(width,'max_abs_nat',red,'Maximum')]:
 vals=[g[key]for g in groups];a.bar(x+dx,vals,width=width,color=color,label=label)
 for xx,v in zip(x+dx,vals):a.text(xx,v+.14,f'{v:.2f}',ha='center',fontsize=8)
a.set_xticks(x,['Quality\nn=2,304','32K targets\nn=288','Structured\nn=576']);a.set_ylim(0,12);a.set_ylabel('Absolute target log-probability difference (nat)');a.set_title('C  Full-vocabulary score errors and tails',loc='left',fontweight='bold',pad=12);a.legend(frameon=False,fontsize=9,ncol=3,loc='upper left');a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
a=axs[1,1];layers=d['mhc_layers'];xs=[v['layer']for v in layers];line1=a.plot(xs,[v['rmse']for v in layers],color=teal,lw=2,label='RMSE');a.set_yscale('log');a.set_xlabel('Native block index');a.set_ylabel('RMSE across 4 positions × 4 streams × 5,120',color=teal);a.tick_params(axis='y',labelcolor=teal);a.set_xticks([0,10,20,30,39]);a.grid(axis='y',alpha=.2)
b=a.twinx();b.spines['right'].set_visible(True);line2=b.plot(xs,[v['cosine']for v in layers],color=blue,lw=1.7,ls='--',label='Cosine');b.set_ylabel('Cosine similarity',color=blue);b.tick_params(axis='y',labelcolor=blue);b.set_ylim(.5,1.025);a.set_title('D  Actual post-FFN mHC stream boundary',loc='left',fontweight='bold',pad=12);a.legend(line1+line2,[z.get_label()for z in line1+line2],loc='lower right',frameon=False,fontsize=9)
fig.text(.10,.052,'191/275 paired cells; 3,168 target scores. Missing GPU chunk/padding cells remain missing. Selected top-1 is not task accuracy.',fontsize=9,color=ink)
fig.text(.10,.030,'Core diagnostics only. Separate head controls, timing boundaries and explicit numerical gaps are detailed in the report.',fontsize=9,color=ink)
for extension in ['svg','png']:fig.savefig(out/('deepseek41-core-diagnostics.'+extension),dpi=170,facecolor='white')
plt.close(fig)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();receipt=dict(status='rendered-aggregate-diagnostics',source_sha256=sha(source),numerical_acceptance=False,files={p.name:sha(p)for p in out.glob('deepseek41-core-diagnostics.*')},measurement_sha256=sha(source))
(out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
