"""Evaluate causal five-second quantitative features on single-condition videos."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_models import CACHE_VERSION, feature_value, read_csv


BASES = {"H2": ["flame_L", "flame_a", "flame_b"], "RH": ["drop_L", "drop_a", "drop_b"]}


def add_temporal(rows, window=5.0):
    by_video = defaultdict(list)
    for row in rows: by_video[str(row["video"])].append(row)
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: float(row["time"]))
        times = np.asarray([float(row["time"]) for row in video_rows])
        names = sorted(set(BASES["H2"] + BASES["RH"]))
        values = {name: np.asarray([feature_value(row, name) for row in video_rows]) for name in names}
        for index, row in enumerate(video_rows):
            start = int(np.searchsorted(times, times[index]-window, side="left"))
            dt = max(times[index]-times[start], 1e-6)
            for name in names:
                row[name+"_med5"] = float(np.median(values[name][start:index+1]))
                row[name+"_slope5"] = float((values[name][index]-values[name][start])/dt) if index>start else 0.0
            row["history_seconds"] = float(times[index]-times[start])
        for label in ("h2_value", "rh_value"):
            segment_start = None
            previous = object()
            for row in video_rows:
                value = row.get(label)
                if value is None:
                    segment_start = None
                    previous = object()
                    row[label+"_stable_seconds"] = 0.0
                    continue
                if segment_start is None or value != previous:
                    segment_start = float(row["time"])
                row[label+"_stable_seconds"] = float(row["time"]) - segment_start
                previous = value


def models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                             random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=220, learning_rate=.04,
                                                             max_leaf_nodes=15, min_samples_leaf=15,
                                                             l2_regularization=2, random_state=42),
    }


def feature_sets(task):
    base = BASES[task]
    return {
        "static": base,
        "median5": [name+"_med5" for name in base],
        "temporal5": base + [name+"_med5" for name in base] + [name+"_slope5" for name in base],
    }


def evaluate(rows, label, features, estimator):
    x=np.asarray([[float(row[name]) for name in features] for row in rows]); y=np.asarray([float(row[label]) for row in rows])
    groups=np.asarray([str(row["group"]) for row in rows]); pred=np.full(len(y),np.nan)
    for group in sorted(set(groups)):
        test=groups==group; train=~test
        fitted=clone(estimator).fit(x[train],y[train]); pred[test]=np.asarray(fitted.predict(x[test])).reshape(-1)
    per=[]
    for group in sorted(set(groups)):
        use=groups==group; per.append({"group":group,"mae":float(mean_absolute_error(y[use],pred[use])),"n":int(use.sum())})
    return {"video_macro_mae":float(np.mean([r["mae"] for r in per])),"frame_mae":float(mean_absolute_error(y,pred)),
            "frame_r2":float(r2_score(y,pred)),"n_frames":len(y),"n_videos":len(per),"per_video":per},y,pred,groups


def make_figure(path, predictions, range_predictions, report):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.2),constrained_layout=True)
    panels=[(axes[0,0],predictions,"H2",(0,4),"H₂ full range"),
            (axes[0,1],predictions,"RH",(20,90),"RH full range"),
            (axes[1,0],range_predictions,"H2 3-4%",(3,4),"H₂ validated range"),
            (axes[1,1],range_predictions,"RH 70-90%",(70,90),"RH validated range")]
    colors={"H2":"#D55E00","RH":"#0072B2"}
    for label,(ax,data,task,limits,title) in zip("ABCD",panels):
        use=[row for row in data if row["task"]==task]
        base="H2" if task.startswith("H2") else "RH"
        ax.scatter([row["reference"] for row in use],[row["prediction"] for row in use],
                   s=8,alpha=.22,color=colors[base],edgecolor="none")
        ax.plot(limits,limits,"--",color="#333",lw=1)
        pad=(limits[1]-limits[0])*.07
        ax.set(xlim=(limits[0]-pad,limits[1]+pad),ylim=(limits[0]-pad,limits[1]+pad),
               xlabel="Reference",ylabel="Held-out-video prediction",title=title)
        mae=np.mean([abs(row["prediction"]-row["reference"]) for row in use])
        ax.text(.04,.94,f"Frame MAE={mae:.2f}",transform=ax.transAxes,va="top")
        ax.text(.96,.05,label,transform=ax.transAxes,ha="right",fontweight="bold",fontsize=11)
    fig.suptitle("Single-condition quantitative validation (causal 5 s, video-wise holdout)",fontweight="bold")
    fig.savefig(path.with_suffix(".png"),dpi=600,bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"),bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"),bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,default=Path(f"training/cache/{CACHE_VERSION}/features.csv"));
    parser.add_argument("--output",type=Path,default=Path("training/output/temporal_quantitation")); args=parser.parse_args()
    rows=read_csv(args.cache); add_temporal(rows)
    tasks={"H2":([r for r in rows if r["kind"]=="h2_only" and r.get("h2_value") is not None],"h2_value"),
           "RH":([r for r in rows if r["kind"]=="rh_only" and r.get("rh_value") is not None],"rh_value")}
    comparison={"H2":{},"RH":{}}; outputs={}; retained_rows={}
    for task,(task_rows,label) in tasks.items():
        # A quantitative reading requires a complete causal observation window.
        label = "h2_value" if task == "H2" else "rh_value"
        task_rows=[r for r in task_rows if float(r["history_seconds"])>=4.5
                   and float(r[label+"_stable_seconds"])>=4.5]
        retained_rows[task]=(task_rows,label)
        for feature_name,features in feature_sets(task).items():
            for model_name,model in models().items():
                key=feature_name+"+"+model_name; metric,y,pred,groups=evaluate(task_rows,label,features,model)
                comparison[task][key]=metric; outputs[(task,key)]=(task_rows,y,pred,groups)
    selected={task:min(comparison[task],key=lambda k:comparison[task][k]["video_macro_mae"]) for task in tasks}
    predictions=[]; levels={}
    for task,key in selected.items():
        task_rows,y,pred,groups=outputs[(task,key)]
        for row,t,e,g in zip(task_rows,y,pred,groups): predictions.append({"task":task,"model":key,"video":row["video"],"group":g,"time":row["time"],"reference":float(t),"prediction":float(e),"residual":float(e-t)})
        level_rows=[]
        for level in sorted(set(y)):
            use=y==level; level_rows.append({"reference":float(level),"mae":float(mean_absolute_error(y[use],pred[use])),
                                              "n_frames":int(use.sum()),"n_videos":len(set(groups[use]))})
        levels[task]=level_rows
    range_definitions={"H2":{"3-4%":(3,4)},"RH":{"70-90%":(70,90),"80-90%":(80,90)}}
    range_models={"H2":{},"RH":{}}; range_predictions=[]
    for task, definitions in range_definitions.items():
        task_rows,label=retained_rows[task]
        for range_name,(low,high) in definitions.items():
            use=[row for row in task_rows if low <= float(row[label]) <= high]
            candidates={}; candidate_outputs={}
            for feature_name,features in feature_sets(task).items():
                for model_name,model in models().items():
                    key=feature_name+"+"+model_name
                    metric,y,pred,groups=evaluate(use,label,features,model); candidates[key]=metric
                    candidate_outputs[key]=(y,pred,groups)
            best=min(candidates,key=lambda key:candidates[key]["video_macro_mae"])
            range_models[task][range_name]={"selected":best,"metrics":candidates[best],"models":candidates}
            y,pred,groups=candidate_outputs[best]
            for truth,estimate,group in zip(y,pred,groups):
                range_predictions.append({"task":task+" "+range_name,"group":str(group),
                                          "reference":float(truth),"prediction":float(estimate)})
    report={"policy":"single condition; causal prior 5 s; leave-one-video-group-out","selected":selected,
            "models":comparison,"level_metrics":levels,"validated_ranges":range_models}
    args.output.mkdir(parents=True,exist_ok=True); (args.output/"metrics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    with (args.output/"predictions.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
    with (args.output/"validated_range_predictions.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(range_predictions[0]));w.writeheader();w.writerows(range_predictions)
    make_figure(args.output/"single_condition_quantitative_validation",predictions,range_predictions,report)
    print(json.dumps({"selected":selected,"metrics":{t:comparison[t][selected[t]] for t in selected},
                      "levels":levels,"validated_ranges":{task:{name:{"selected":value["selected"],"metrics":value["metrics"]}
                      for name,value in ranges.items()} for task,ranges in range_models.items()}},indent=2))


if __name__=="__main__": main()
