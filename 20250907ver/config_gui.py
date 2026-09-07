"""AI Mobility 전체 설정 GUI. GUI의 모든 실험/시뮬레이션 파라미터는 config.json으로 저장됩니다."""
import json, os, subprocess, sys, tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from region_autocomplete import RegionSearchEntry
from config_loader import DEFAULT_CONFIG

ROOT=os.path.dirname(os.path.abspath(__file__)); PRESET_DIR=os.path.join(ROOT,"presets"); os.makedirs(PRESET_DIR,exist_ok=True)
CONFIG_PATH=os.path.join(ROOT,"config.json")
FREQ_OPTIONS=["1min","5min","10min","15min","30min"]
STRATEGIES=["patrol","prepositioned"]

# key, label, min, max, decimals, integer
NUMERIC_GROUPS={
"Module 1 — 맵/시뮬레이션":[
("grid_x","그리드 가로 블록 수",2,20,0,1),("grid_y","그리드 세로 블록 수",2,20,0,1),("grid_length","블록 한 변 길이(m)",10,1000,0,1),
("num_normal_cars","일반 차량 대수",0,500,0,1),("num_taxis","택시 대수",0,500,0,1),("num_auto_cars","자율주행차 대수",0,100,0,1),("num_obstacles","장애물 개수",0,100,0,1),("num_passengers","기본 승객 수",0,10000,0,1),
("sim_start_hour","시뮬레이션 시작 시각",0,23,0,1),("sim_end_hour","시뮬레이션 종료 시각",1,24,0,1),("passenger_wait_timeout","승객 최대 대기시간(초)",1,7200,0,1),("depart_jitter_sec","출발시간 분산 표준편차(초)",0,1800,1,0),("sumo_time_to_teleport","SUMO teleport 기준(초)",0,7200,0,1),
("taxi_remaining_edges_threshold","택시 재타겟팅 잔여 edge 기준",0,20,0,1),("taxi_target_pick_attempts","목적지 탐색 시도 횟수",1,100,0,1),("taxi_fail_threshold","택시 고립 판정 연속 실패 횟수",1,100,0,1),
("school_pop_base","학교 기본 인구 모수",0,10000,0,1),("company_pop_base","회사 기본 인구 모수",0,10000,0,1),("residential_base_pop_per_edge","주거 edge당 기본 인구",0,10000,0,1),("residential_schedule_scale","주거 스케줄 전체 배율",0,1,3,0),("residential_taxi_probability","기본승객수 독립가챠 택시확률",0,1,3,0),
("residential_out_hour","주거 외출 시각",0,24,2,0),("residential_out_probability","주거 외출 확률",0,1,3,0),("residential_evening_in_hour","주거 저녁 귀가 시각",0,24,2,0),("residential_evening_in_probability","주거 저녁 귀가 확률",0,1,3,0),("residential_late_in_hour","주거 늦은 귀가 시각",0,24,2,0),("residential_late_in_probability","주거 늦은 귀가 확률",0,1,3,0),
("school_start_hour","학교 등교 시작",0,23.99,2,0),("school_end_hour","학교 등교 종료",0,24,2,0),("school_taxi_peak_hour","학교 택시확률 peak",0,24,2,0),("school_afternoon_start_hour","학교 하교 시작",0,24,2,0),("school_afternoon_end_hour","학교 하교 종료",0,24,2,0),("school_afternoon_fraction","학교 하교 비율",0,1,3,0),
("company_start_hour","회사 출근 시작",0,24,2,0),("company_end_hour","회사 출근 종료",0,24,2,0),("company_taxi_peak_hour","회사 택시확률 peak",0,24,2,0),
("lunch_start_hour","점심 시작",0,24,2,0),("lunch_end_hour","점심 종료",0,24,2,0),("lunch_company_release_fraction","회사 점심 이탈 비율",0,1,3,0),("lunch_taxi_fraction","점심 택시 비율",0,1,3,0),
("evening_start_hour","퇴근 시작",0,24,2,0),("evening_end_hour","퇴근 주요시간 종료",0,24,2,0),("evening_taxi_fraction","퇴근 택시 비율",0,1,3,0),("late_evening_start_hour","늦은 퇴근 시작",0,24,2,0),("late_evening_end_hour","늦은 퇴근 종료",0,24,2,0),("evening_residential_fraction","퇴근 목적지 주거 비율",0,1,3,0),("restaurant_stay_sec","음식점 체류시간(초)",0,86400,0,1),("restaurant_pop_base","음식점 건물당 인구",0,1000,0,1),("restaurant_window1_start","음식점 시간대1 시작",0,24,2,0),("restaurant_window1_end","음식점 시간대1 종료",0,24,2,0),("restaurant_window2_start","음식점 시간대2 시작",0,24,2,0),("restaurant_window2_end","음식점 시간대2 종료",0,24,2,0),("restaurant_civilian_taxi_probability","음식점 시민 택시확률",0,1,3,0),
],
"Module 2 — 전처리":[("h3_resolution","H3 resolution",5,15,0,1),("max_lag","Max lag",1,100,0,1),("rolling_short","단기 rolling window",1,100,0,1),("rolling_long","장기 rolling window",1,200,0,1)],
"Module 3 — 예측모델":[("xgb_n_estimators","XGB n_estimators",10,1000,0,1),("xgb_max_depth","XGB max_depth",1,30,0,1),("xgb_learning_rate","XGB learning_rate",0.001,1,4,0),("test_size","Test size",0.01,0.5,3,0),("cnn_hidden_dim","CNN hidden_dim",8,512,0,1),("cnn_num_layers","CNN num_layers",1,8,0,1),("cnn_kernel_size","CNN kernel_size",1,15,0,1),("cnn_epochs","CNN epochs",1,1000,0,1),("cnn_batch_size","CNN batch_size",1,512,0,1),("cnn_lr","CNN learning rate",0.00001,0.1,6,0)],
"Module 4 — 배차/가격":[("base_fare","기본요금(원)",0,50000,0,1),("min_multiplier","최소 할증배수",0.1,5,2,0),("max_multiplier","최대 할증배수",0.1,10,2,0),("surge_coefficient","할증 증가계수",0,5,3,0),("mock_available_taxis_min","테스트 가용택시 최소",0,500,0,1),("mock_available_taxis_max","테스트 가용택시 최대",0,500,0,1),("mock_taxi_count","배차 테스트 택시 수",1,500,0,1),("mock_passenger_count","배차 테스트 승객 수",1,500,0,1)],
"공통/실험":[("demo_data_rows","학습 데모 데이터 행 수",100,100000,0,1),("demo_data_minutes","추론 데모 데이터 길이",10,100000,0,1),("xgb_random_state","XGB random state",0,2147483647,0,1),("train_random_state","학습 random state",0,2147483647,0,1)]}

BOOLS=[("use_real_map","실제 OSM 지도 사용"),("dispatch_use_euclidean","Module 4 테스트 거리: 유클리드 사용")]
CHOICES=[("taxi_strategy","택시 전략",STRATEGIES),("taxi_dispatch_algorithm","SUMO taxi dispatch algorithm",["greedy","routeExtension","traci"]),("taxi_idle_algorithm","SUMO taxi idle algorithm",["randomCircling","stopOnRoad","taxiStop"])]

class ConfigGUI:
 def __init__(self,root):
  self.root=root; root.title("AI Mobility — CONFIG GUI (전체 설정)"); root.geometry("780x820"); self.vars={}; self._build() ; self.load_config_file(show=False)
 def _build(self):
  ttk.Label(self.root,text="CONFIG GUI — 프로젝트 전체 실험/시뮬레이션 설정",font=("맑은 고딕",14,"bold")).pack(pady=10)
  nb=ttk.Notebook(self.root); nb.pack(fill="both",expand=True,padx=12,pady=8)
  for tab,items in NUMERIC_GROUPS.items(): self._numeric_tab(nb,tab,items)
  self._special_tab(nb)
  pf=ttk.LabelFrame(self.root,text="프리셋",padding=8); pf.pack(fill="x",padx=12,pady=4)
  self.preset_combo=ttk.Combobox(pf,state="readonly"); self.preset_combo.pack(side="left",fill="x",expand=True,padx=5); ttk.Button(pf,text="불러오기",command=self.load_preset).pack(side="left"); ttk.Button(pf,text="저장",command=self.save_preset).pack(side="left"); ttk.Button(pf,text="삭제",command=self.delete_preset).pack(side="left"); self.refresh_presets()
  bf=ttk.Frame(self.root); bf.pack(fill="x",padx=12,pady=8)
  ttk.Button(bf,text="현재 설정 저장",command=self.save_config_only).pack(side="left",fill="x",expand=True,padx=3); ttk.Button(bf,text="▶ 플레이",command=self.run_main).pack(side="left",fill="x",expand=True,padx=3); ttk.Button(bf,text="학습만 실행",command=self.run_training).pack(side="left",fill="x",expand=True,padx=3); ttk.Button(bf,text="🧪 동일 설정 A/B 비교",command=self.run_headless_compare).pack(side="left",fill="x",expand=True,padx=3)
  self.status=ttk.Label(self.root,text="현재 config.json을 불러오는 중..."); self.status.pack(anchor="w",padx=12,pady=(0,8))
 def _numeric_tab(self,nb,tab,items):
  frame=ttk.Frame(nb,padding=12); nb.add(frame,text=tab.split("—")[0].strip() if "—" in tab else tab)
  canvas=tk.Canvas(frame,highlightthickness=0); sb=ttk.Scrollbar(frame,orient="vertical",command=canvas.yview); inner=ttk.Frame(canvas); inner.bind("<Configure>",lambda e:canvas.configure(scrollregion=canvas.bbox("all"))); canvas.create_window((0,0),window=inner,anchor="nw"); canvas.configure(yscrollcommand=sb.set); canvas.pack(side="left",fill="both",expand=True); sb.pack(side="right",fill="y")
  for key,label,lo,hi,dec,isint in items:
   row=ttk.Frame(inner); row.pack(fill="x",pady=3); ttk.Label(row,text=label,width=34).pack(side="left"); var=tk.DoubleVar(value=DEFAULT_CONFIG.get(key,0)); self.vars[key]=(var,dec,isint); ttk.Entry(row,textvariable=var,width=12).pack(side="right",padx=4); ttk.Scale(row,from_=lo,to=hi,variable=var,length=300).pack(side="right",fill="x",expand=True)
 def _special_tab(self,nb):
  f=ttk.Frame(nb,padding=12); nb.add(f,text="공통/선택")
  r=ttk.Frame(f); r.pack(fill="x",pady=5); ttk.Label(r,text="지역",width=30).pack(side="left"); self.region_var=tk.StringVar(); self.region=RegionSearchEntry(r,default="홍대입구",width=24); self.region.pack(side="right")
  r=ttk.Frame(f); r.pack(fill="x",pady=5); ttk.Label(r,text="freq",width=30).pack(side="left"); self.freq_var=tk.StringVar(); ttk.Combobox(r,textvariable=self.freq_var,values=FREQ_OPTIONS,state="readonly",width=20).pack(side="right")
  self.bool_vars={}
  for key,label in BOOLS:
   v=tk.BooleanVar(value=bool(DEFAULT_CONFIG.get(key,False))); self.bool_vars[key]=v; ttk.Checkbutton(f,text=label,variable=v).pack(anchor="w",pady=4)
  self.choice_vars={}
  for key,label,vals in CHOICES:
   r=ttk.Frame(f); r.pack(fill="x",pady=5); ttk.Label(r,text=label,width=30).pack(side="left"); v=tk.StringVar(); self.choice_vars[key]=v; ttk.Combobox(r,textvariable=v,values=vals,state="readonly",width=20).pack(side="right")
  r=ttk.Frame(f); r.pack(fill="x",pady=8); ttk.Label(r,text="Passenger seed (비우면 랜덤)",width=30).pack(side="left"); self.seed_var=tk.StringVar(); ttk.Entry(r,textvariable=self.seed_var,width=20).pack(side="right")
 def current(self):
  c={"region":self.region.get() or "홍대입구","freq":self.freq_var.get() or DEFAULT_CONFIG["freq"],"passenger_seed":None,"taxi_strategy":self.choice_vars["taxi_strategy"].get() or DEFAULT_CONFIG["taxi_strategy"]}
  s=self.seed_var.get().strip(); c["passenger_seed"]=int(s) if s.lstrip("-").isdigit() else None
  c.update({k:v.get() for k,v in self.bool_vars.items()}); c.update({k:v.get() for k,v in self.choice_vars.items()})
  for k,(v,dec,isint) in self.vars.items(): c[k]=int(v.get()) if isint else round(float(v.get()),dec)
  return c
 def set_values(self,c):
  self.region.set(c.get("region","홍대입구")); self.freq_var.set(c.get("freq",DEFAULT_CONFIG["freq"])); self.seed_var.set("" if c.get("passenger_seed") is None else str(c.get("passenger_seed")))
  for k,v in self.bool_vars.items(): v.set(bool(c.get(k,DEFAULT_CONFIG.get(k,False))))
  for k,v in self.choice_vars.items(): v.set(c.get(k,DEFAULT_CONFIG.get(k,"")))
  for k,(v,dec,isint) in self.vars.items():
   if k in c: v.set(int(c[k]) if isint else float(c[k]))
 def load_config_file(self,show=True):
  try:
   with open(CONFIG_PATH,"r",encoding="utf-8") as f: c=json.load(f)
  except Exception: c=DEFAULT_CONFIG.copy()
  merged={**DEFAULT_CONFIG,**c}; self.set_values(merged); self.status.config(text="현재 config.json을 GUI에 불러왔습니다.") if hasattr(self,'status') else None
 def save_config_only(self):
  c=self.current()
  with open(CONFIG_PATH,"w",encoding="utf-8") as f:
   json.dump(c,f,ensure_ascii=False,indent=2)
  self.status.config(text="config.json 저장 완료")
 def refresh_presets(self):
  files=[f[:-5] for f in os.listdir(PRESET_DIR) if f.endswith(".json")]; self.preset_combo["values"]=files;
  if files:self.preset_combo.current(0)
 def save_preset(self):
  name=simpledialog.askstring("프리셋","이름:");
  if name:
   with open(os.path.join(PRESET_DIR,name+".json"),"w",encoding="utf-8") as f:json.dump(self.current(),f,ensure_ascii=False,indent=2)
   self.refresh_presets(); self.status.config(text=f"프리셋 '{name}' 저장 완료")
 def load_preset(self):
  n=self.preset_combo.get();
  if not n:return
  with open(os.path.join(PRESET_DIR,n+".json"),encoding="utf-8") as f:self.set_values({**DEFAULT_CONFIG,**json.load(f)})
  self.status.config(text=f"프리셋 '{n}' 불러오기 완료")
 def delete_preset(self):
  n=self.preset_combo.get();
  if n and messagebox.askyesno("확인",f"'{n}' 삭제?"):os.remove(os.path.join(PRESET_DIR,n+".json"));self.refresh_presets()
 def _env(self):
  e=os.environ.copy();e["PYTHONIOENCODING"]="utf-8";e["PYTHONUTF8"]="1";return e
 def _run(self,script,args=(),blocking=False):
  path=os.path.join(ROOT,script)
  try:
   if blocking:return subprocess.run([sys.executable,path,*args],cwd=ROOT,env=self._env(),check=True)
   if os.name=="nt":
    # 새 콘솔에서 실행 후 끝나도 창이 자동으로 안 닫히게 cmd /k로 감쌈 (결과 다 보고 사용자가 직접 X 눌러야 닫힘)
    cmd_args=["cmd","/k",sys.executable,path,*args]
    subprocess.Popen(cmd_args,cwd=ROOT,env=self._env(),creationflags=subprocess.CREATE_NEW_CONSOLE);return True
   subprocess.Popen([sys.executable,path,*args],cwd=ROOT,env=self._env());return True
  except Exception as e:messagebox.showerror("실행 오류",str(e));return False
 def run_main(self):
  self.save_config_only();
  if self._run(os.path.join("module1_simulation","build_env.py"),blocking=True):self._run("main.py")
 def run_training(self):self.save_config_only();self._run("train.py")
 def run_headless_compare(self):self.save_config_only();self._run("measure_wait_time.py",["compare"])

if __name__=="__main__":
 root=tk.Tk();ConfigGUI(root);root.mainloop()