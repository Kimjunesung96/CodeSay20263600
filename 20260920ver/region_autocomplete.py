"""
config_gui.py에서 쓰는 지역명 검색 + 자동완성 위젯.

동작 방식:
- 타이핑 즉시: REGION_PRESETS(로컬 프리셋) 이름 중 일치하는 것부터 바로 후보로 표시 (네트워크 대기 없음)
- 2글자 이상 입력 & 450ms 동안 추가 입력 없을 때: geo_lookup.search_suggestions()로
  Nominatim 실제 검색 결과도 백그라운드 스레드에서 가져와 후보에 합침 (GUI 멈춤 방지)
- 후보 목록에서 클릭 또는 Enter로 확정. 확정된 텍스트가 곧 config.json의 "region" 값이 됨
  (실제 좌표 확정은 config_loader.py의 geo_lookup.lookup_region()이 담당 — 여기선 이름만 고름)
"""

import threading
import tkinter as tk
from tkinter import ttk

from config_loader import REGION_PRESETS
from geo_lookup import search_suggestions


class RegionSearchEntry(ttk.Frame):
    DEBOUNCE_MS = 450

    def __init__(self, parent, default="홍대입구", width=20):
        super().__init__(parent)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self, textvariable=self.var, width=width)
        self.entry.pack(fill="x")

        self._popup = None
        self._listbox = None
        self._debounce_id = None
        self._search_generation = 0

        self.entry.bind("<KeyRelease>", self._on_key_release)
        self.entry.bind("<FocusOut>", self._on_focus_out)
        self.entry.bind("<Down>", self._focus_listbox)
        self.entry.bind("<Return>", lambda e: self._hide_popup())

    # ---------- 외부 인터페이스 (config_gui.py에서 사용) ----------
    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(value)

    # ---------- 내부 로직 ----------
    def _on_key_release(self, event):
        if event.keysym in ("Down", "Up", "Return", "Escape"):
            return
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._debounce_id = self.after(self.DEBOUNCE_MS, self._trigger_search)

    def _trigger_search(self):
        query = self.get()
        local_matches = [name for name in REGION_PRESETS.keys() if query and query in name]

        if len(query) < 2:
            self._show_suggestions(local_matches)
            return

        # 로컬 프리셋 매칭은 API 응답을 기다리지 않고 즉시 표시
        self._show_suggestions(local_matches)

        self._search_generation += 1
        gen = self._search_generation

        def worker():
            try:
                remote = search_suggestions(query, limit=5)
                remote_names = [r["display_name"] for r in remote if r["display_name"] not in local_matches]
            except Exception:
                remote_names = []
            combined = local_matches + remote_names
            self.after(0, lambda: self._apply_search_result(gen, combined))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_search_result(self, gen, names):
        if gen != self._search_generation:
            return  # 이미 더 최신 검색이 진행 중 -> 오래된 결과는 버림
        self._show_suggestions(names)

    def _show_suggestions(self, names):
        if not names:
            self._hide_popup()
            return

        if self._popup is None:
            self._popup = tk.Toplevel(self)
            self._popup.wm_overrideredirect(True)
            self._listbox = tk.Listbox(self._popup, height=min(6, max(1, len(names))))
            self._listbox.pack(fill="both", expand=True)
            self._listbox.bind("<<ListboxSelect>>", self._on_select)
            self._listbox.bind("<Return>", self._on_select)

        self._listbox.delete(0, tk.END)
        for name in names[:8]:
            self._listbox.insert(tk.END, name)

        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        self._popup.wm_geometry(f"+{x}+{y}")
        self._popup.deiconify()
        self._popup.lift()

    def _hide_popup(self):
        if self._popup is not None:
            self._popup.destroy()
            self._popup = None
            self._listbox = None

    def _on_select(self, event):
        if not self._listbox:
            return
        sel = self._listbox.curselection()
        if sel:
            self.var.set(self._listbox.get(sel[0]))
        self._hide_popup()
        self.entry.focus_set()

    def _focus_listbox(self, event):
        if self._listbox is not None:
            self._listbox.focus_set()
            if self._listbox.size() > 0:
                self._listbox.selection_set(0)
        return "break"

    def _on_focus_out(self, event):
        # 리스트박스 클릭이 처리될 시간을 준 뒤 닫기 (즉시 닫으면 클릭 이벤트가 씹힘)
        self.after(150, self._hide_popup)