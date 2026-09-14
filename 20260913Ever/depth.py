import os
import re
import ctypes
from ctypes import wintypes
from collections import defaultdict
import tkinter as tk
from tkinter import ttk, messagebox
import shutil
import subprocess

# ============================================================
# Project Code Analyzer
# 프로젝트 최상단에 이 파일을 두고 실행하세요.
# ============================================================

# ------------------------------------------------------------
# 프로젝트 위치
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# 복사될 목적지 폴더 경로
# ------------------------------------------------------------
TARGET_COPY_PATH = r"C:\Users\skrkt\Downloads\CopiedBlueprints"

# ------------------------------------------------------------
# 무시할 폴더 (가상환경 관련 폴더 Lib, Scripts, venv 등 추가)
# ------------------------------------------------------------
IGNORE_DIRS = {
    "Library", "Logs", "Packages", "ProjectSettings", "obj", "bin", ".git", ".vs",
    ".idea", "Temp", "UserSettings", "__pycache__", "node_modules", "Build", "Builds",
    "Lib", "lib", "Scripts", "scripts", "site-packages", "venv", ".venv", "env", ".env"
}

# ------------------------------------------------------------
# 지원 확장자
# ------------------------------------------------------------
FILE_TYPES = {
    ".cs": "C#", ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".jsx": "React JSX", ".tsx": "React TSX", ".java": "Java", ".cpp": "C++",
    ".c": "C", ".h": "C/C++ Header", ".hpp": "C++ Header", ".kt": "Kotlin",
    ".go": "Go", ".rs": "Rust", ".html": "HTML", ".css": "CSS",
    ".scss": "SCSS", ".json": "JSON", ".vue": "Vue",
}

# ------------------------------------------------------------
# 전역 분석 데이터
# ------------------------------------------------------------
all_files = {}
name_index = defaultdict(set)
forward_graph = defaultdict(set)
reverse_graph = defaultdict(set)

# ============================================================
# 기본 유틸
# ============================================================
def normalize_path(path):
    return os.path.normpath(os.path.abspath(path))

def relative_path(path):
    try:
        return os.path.relpath(path, BASE_DIR)
    except ValueError:
        return path

def get_extension(path):
    return os.path.splitext(path)[1].lower()

def read_file(path):
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, OSError):
            continue
    return ""

# ============================================================
# 심볼 추출
# ============================================================
def extract_symbols(content, path):
    extension = get_extension(path)
    symbols = set()

    filename = os.path.splitext(os.path.basename(path))[0]
    if filename:
        symbols.add(filename)

    if extension == ".cs":
        patterns = [
            r"\bclass\s+([A-Za-z_]\w*)", r"\bstruct\s+([A-Za-z_]\w*)",
            r"\binterface\s+([A-Za-z_]\w*)", r"\benum\s+([A-Za-z_]\w*)",
            r"\bdelegate\s+\w+\s+([A-Za-z_]\w*)",
        ]
        for pattern in patterns:
            symbols.update(re.findall(pattern, content))

    elif extension == ".py":
        patterns = [r"\bclass\s+([A-Za-z_]\w*)", r"\bdef\s+([A-Za-z_]\w*)"]
        for pattern in patterns:
            symbols.update(re.findall(pattern, content))

    elif extension in {".java", ".kt"}:
        patterns = [
            r"\bclass\s+([A-Za-z_]\w*)", r"\binterface\s+([A-Za-z_]\w*)",
            r"\benum\s+([A-Za-z_]\w*)", r"\bobject\s+([A-Za-z_]\w*)",
        ]
        for pattern in patterns:
            symbols.update(re.findall(pattern, content))

    elif extension in {".js", ".ts", ".jsx", ".tsx"}:
        patterns = [
            r"\bclass\s+([A-Za-z_]\w*)", r"\bfunction\s+([A-Za-z_]\w*)",
            r"\b(?:const|let|var)\s+([A-Za-z_]\w*)",
        ]
        for pattern in patterns:
            symbols.update(re.findall(pattern, content))

    elif extension in {".c", ".cpp", ".h", ".hpp"}:
        patterns = [
            r"\bclass\s+([A-Za-z_]\w*)", r"\bstruct\s+([A-Za-z_]\w*)", r"\benum\s+([A-Za-z_]\w*)",
        ]
        for pattern in patterns:
            symbols.update(re.findall(pattern, content))

    return symbols

# ============================================================
# 프로젝트 스캔
# ============================================================
def scan_project(selected_extensions, status_callback=None):
    global all_files, name_index, forward_graph, reverse_graph

    all_files = {}
    name_index = defaultdict(set)
    forward_graph = defaultdict(set)
    reverse_graph = defaultdict(set)

    file_paths = []

    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for filename in files:
            extension = get_extension(filename)
            if extension not in selected_extensions:
                continue

            full_path = normalize_path(os.path.join(root, filename))
            if full_path == normalize_path(__file__):
                continue

            file_paths.append(full_path)

    file_paths.sort(key=lambda p: relative_path(p).lower())
    total = len(file_paths)

    for index, path in enumerate(file_paths, start=1):
        if status_callback:
            status_callback(f"파일 분석 중... {index}/{total}\n{relative_path(path)}")

        content = read_file(path)
        symbols = extract_symbols(content, path)

        all_files[path] = {
            "name": os.path.basename(path),
            "path": path,
            "relative": relative_path(path),
            "extension": get_extension(path),
            "content": content,
            "symbols": symbols,
        }

        for symbol in symbols:
            name_index[symbol].add(path)

    total_files = len(all_files)
    for index, (current_path, info) in enumerate(all_files.items(), start=1):
        if status_callback:
            status_callback(f"참조 관계 분석 중... {index}/{total_files}\n{info['relative']}")

        content = info["content"]
        words = set(re.findall(r"\b[A-Za-z_]\w*\b", content))

        for word in words:
            targets = name_index.get(word)
            if not targets:
                continue

            for target_path in targets:
                if target_path == current_path:
                    continue
                forward_graph[current_path].add(target_path)
                reverse_graph[target_path].add(current_path)

    if status_callback:
        status_callback(f"스캔 완료\n총 {len(all_files)}개 파일")

# ============================================================
# Windows 실제 파일 클립보드 복사
# ============================================================
CF_HDROP = 15
GMEM_MOVEABLE = 0x0002
GMEM_ZEROINIT = 0x0040

class DROPFILES(ctypes.Structure):
    _fields_ = [
        ("pFiles", wintypes.DWORD),
        ("pt", wintypes.POINT),
        ("fNC", wintypes.BOOL),
        ("fWide", wintypes.BOOL),
    ]

def copy_files_to_windows_clipboard(paths):
    """여러 실제 파일을 Windows 파일 클립보드(CF_HDROP)에 넣습니다."""
    if os.name != "nt":
        raise RuntimeError("여러 파일 클립보드는 Windows에서만 지원됩니다.")

    paths = [os.path.abspath(path) for path in paths]

    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    # 여러 파일 경로를 Windows CF_HDROP 형식으로 구성
    file_list = "".join(path + "\0" for path in paths) + "\0"
    data = file_list.encode("utf-16-le")

    dropfiles_size = ctypes.sizeof(DROPFILES)
    total_size = dropfiles_size + len(data)

    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL

    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL

    user32.SetClipboardData.argtypes = [
        wintypes.UINT,
        wintypes.HANDLE
    ]
    user32.SetClipboardData.restype = wintypes.HANDLE

    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL

    kernel32.GlobalAlloc.argtypes = [
        wintypes.UINT,
        ctypes.c_size_t
    ]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL

    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = wintypes.LPVOID

    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL

    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    h_global = kernel32.GlobalAlloc(
        GMEM_MOVEABLE | GMEM_ZEROINIT,
        total_size
    )

    if not h_global:
        raise MemoryError("GlobalAlloc 실패")

    try:
        locked = kernel32.GlobalLock(h_global)

        if not locked:
            raise MemoryError("GlobalLock 실패")

        try:
            dropfiles = DROPFILES()

            dropfiles.pFiles = dropfiles_size
            dropfiles.pt.x = 0
            dropfiles.pt.y = 0
            dropfiles.fNC = 0
            dropfiles.fWide = 1

            ctypes.memmove(
                locked,
                ctypes.byref(dropfiles),
                dropfiles_size
            )

            ctypes.memmove(
                locked + dropfiles_size,
                data,
                len(data)
            )

        finally:
            kernel32.GlobalUnlock(h_global)

        if not user32.OpenClipboard(None):
            raise RuntimeError("Windows 클립보드를 열 수 없습니다.")

        try:
            if not user32.EmptyClipboard():
                raise RuntimeError("클립보드를 비울 수 없습니다.")

            if not user32.SetClipboardData(CF_HDROP, h_global):
                raise RuntimeError("파일 클립보드 설정에 실패했습니다.")

            # Windows가 메모리 소유권을 가져감
            h_global = None

        finally:
            user32.CloseClipboard()

    finally:
        if h_global:
            kernel32.GlobalFree(h_global)
# ============================================================
# GUI
# ============================================================
class ProjectAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Code Analyzer")
        self.root.geometry("450x750")
        self.root.minsize(350, 500)

        self.extension_vars = {}
        self.filtered_files = []
        
        # 파일별 체크박스 상태를 저장할 딕셔너리
        self.file_vars = {}

        self.build_ui()

    def build_ui(self):
        # 상단 제목 및 VS Code 열기 버튼
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Project Code Analyzer", font=("Arial", 18, "bold")).pack(anchor="w")
        
        path_frame = ttk.Frame(top)
        path_frame.pack(fill="x", pady=(4, 8))
        ttk.Label(path_frame, text=f"프로젝트 위치: {BASE_DIR}").pack(side="left")
        
        # VS Code 버튼 추가
        btn_vscode = ttk.Button(path_frame, text="💻 VS Code로 열기", command=self.open_vscode)
        btn_vscode.pack(side="right", padx=5)

        # 확장자 옵션 (스캔 후 자동으로 숨겨짐)
        self.type_frame = ttk.LabelFrame(self.root, text="분석할 파일 종류", padding=8)
        self.type_frame.pack(fill="x", padx=10, pady=4)

        row, column = 0, 0
        for extension, description in FILE_TYPES.items():
            var = tk.BooleanVar(value=(extension == ".cs"))
            self.extension_vars[extension] = var

            ttk.Checkbutton(
                self.type_frame,
                text=f"{extension} ({description})",
                variable=var
            ).grid(row=row, column=column, sticky="w", padx=7, pady=2)

            column += 1
            if column >= 5:
                column = 0
                row += 1

        # 스캔 + 검색 + 설정 변경(숨긴 확장자 옵션 다시 펼치기)
        self.control_frame = ttk.Frame(self.root, padding=(10, 4))
        self.control_frame.pack(fill="x")

        tk.Button(self.control_frame, text="프로젝트 스캔", command=self.start_scan).pack(side="left")
        self.settings_toggle_btn = ttk.Button(
            self.control_frame, text="파일 종류 설정", command=self.toggle_type_frame
        )
        ttk.Label(self.control_frame, text="검색:").pack(side="left", padx=(20, 5))

        self.search_var = tk.StringVar()
        ttk.Entry(self.control_frame, textvariable=self.search_var).pack(side="left", fill="x", expand=True)
        self.search_var.trace_add("write", lambda *args: self.refresh_file_list())

        # 전체 선택 / 전체 해제 컨트롤 버튼 프레임
        self.selection_frame = ttk.Frame(self.root, padding=(10, 0))
        self.selection_frame.pack(fill="x")
        ttk.Button(self.selection_frame, text="☑️ 전체 선택", command=self.select_all).pack(side="left", padx=2)
        ttk.Button(self.selection_frame, text="🔲 전체 해제", command=self.deselect_all).pack(side="left", padx=2)

        # 상태 및 복사 버튼이 들어갈 하단 프레임
        self.bottom_frame = ttk.Frame(self.root, padding=10)
        self.bottom_frame.pack(fill="x", side="bottom")

        self.status_var = tk.StringVar(value="프로젝트를 스캔해주세요.")
        ttk.Label(
            self.bottom_frame,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=5
        ).pack(fill="x", side="bottom")

        # 실제 파일 복사 버튼 추가
        tk.Button(
            self.bottom_frame, 
            text="✅ 선택된 파일 복사하기", 
            command=self.copy_selected_files,
            bg="yellow",
            font=("Arial", 11, "bold")
        ).pack(fill="x", pady=(0, 5), side="bottom", ipady=8)

        # 파일 목록 영역 (스크롤)
        self.list_container = ttk.Frame(self.root, padding=(10, 4, 10, 10))
        self.list_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.list_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.list_container, orient="vertical", command=self.canvas.yview)
        self.file_rows_frame = ttk.Frame(self.canvas)

        self.file_rows_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas_window = self.canvas.create_window((0, 0), window=self.file_rows_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind("<Configure>", self.resize_rows)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

    # ------------------------------------------------------------
    # 신규 추가 기능 메서드들
    # ------------------------------------------------------------
    def open_vscode(self):
        try:
            # 운영체제 명령어로 VS Code를 BASE_DIR 경로에서 엽니다.
            subprocess.Popen(['code', BASE_DIR], shell=True)
            self.status_var.set("VS Code를 실행했습니다.")
        except Exception as e:
            messagebox.showerror("실행 실패", f"VS Code를 실행할 수 없습니다.\n{e}")

    def select_all(self):
        for path in self.filtered_files:
            if path in self.file_vars:
                self.file_vars[path].set(True)

    def deselect_all(self):
        for path in self.filtered_files:
            if path in self.file_vars:
                self.file_vars[path].set(False)
    def copy_selected_files(self):
        """체크된 파일들을 Windows 파일 클립보드에 복사."""
        selected_paths = [
            path for path in self.filtered_files
            if self.file_vars.get(path) and self.file_vars[path].get()
        ]

        if not selected_paths:
            messagebox.showwarning("경고", "복사할 파일을 체크해주세요.")
            return

        try:
            copy_files_to_windows_clipboard(selected_paths)

            self.status_var.set(
                f"총 {len(selected_paths)}개의 파일을 클립보드에 복사했습니다."
            )

            self.deselect_all()

        except Exception as e:
            messagebox.showerror(
                "복사 실패",
                f"파일 클립보드 복사 중 오류가 발생했습니다:\n{e}"
            )
    # ------------------------------------------------------------
    # 기존 메서드들
    # ------------------------------------------------------------
    def resize_rows(self, event):
        try:
            self.canvas.itemconfigure(self.canvas_window, width=event.width)
        except Exception:
            pass

    def on_mousewheel(self, event):
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def start_scan(self):
        selected_extensions = {ext for ext, var in self.extension_vars.items() if var.get()}
        if not selected_extensions:
            messagebox.showwarning("경고", "최소 하나 이상의 확장자를 선택하세요.")
            return

        def update_status(text):
            self.status_var.set(text)
            self.root.update_idletasks()

        scan_project(selected_extensions, status_callback=update_status)

        self.type_frame.pack_forget()
        self.settings_toggle_btn.pack(side="left", padx=(10, 0))

        self.refresh_file_list()

    def toggle_type_frame(self):
        if self.type_frame.winfo_ismapped():
            self.type_frame.pack_forget()
        else:
            self.type_frame.pack(fill="x", padx=10, pady=4, before=self.control_frame)

    def copy_content(self, path):
        content = all_files[path]["content"]
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()
        self.status_var.set(f"내용 복사 완료: {all_files[path]['name']}")

    def copy_filename(self, path):
        filename = all_files[path]["name"]
        self.root.clipboard_clear()
        self.root.clipboard_append(filename)
        self.root.update()
        self.status_var.set(f"파일명 복사 완료: {filename}")

    def copy_real_file(self, path):
        try:
            copy_files_to_windows_clipboard([path])
            self.status_var.set(f"실제 파일 복사 완료 (탐색기에 붙여넣기 가능): {all_files[path]['name']}")
        except Exception as e:
            messagebox.showerror("오류", f"파일 복사 실패:\n{e}")

    def copy_path(self, path):
        rel_p = all_files[path]["relative"]
        self.root.clipboard_clear()
        self.root.clipboard_append(rel_p)
        self.root.update()
        self.status_var.set(f"경로 복사 완료: {rel_p}")

    def show_references(self, path):
        info = all_files[path]
        ref_win = tk.Toplevel(self.root)
        ref_win.title(f"참조 분석 - {info['name']}")
        ref_win.geometry("500x600")

        ttk.Label(
            ref_win,
            text=f"파일: {info['name']}\n경로: {info['relative']}",
            font=("Arial", 11, "bold"),
            padding=10
        ).pack(anchor="w")

        notebook = ttk.Notebook(ref_win)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        tab_fw = ttk.Frame(notebook)
        notebook.add(tab_fw, text=f"이 파일이 참조하는 파일 ({len(forward_graph[path])})")
        self.build_ref_list(tab_fw, forward_graph[path])

        tab_rv = ttk.Frame(notebook)
        notebook.add(tab_rv, text=f"이 파일을 참조하는 파일 ({len(reverse_graph[path])})")
        self.build_ref_list(tab_rv, reverse_graph[path])

    def build_ref_list(self, parent, path_set):
        if not path_set:
            ttk.Label(parent, text="참조 관계가 없습니다.", padding=20).pack()
            return

        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)

        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        c_win = canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(c_win, width=e.width))

        paths = sorted(list(path_set), key=lambda p: all_files[p]["relative"].lower())

        for p in paths:
            lbl_name = tk.Label(frame, text=all_files[p]["name"], fg="blue",
                                 cursor="hand2", anchor="w",
                                 font=("Arial", 9), padx=6, pady=0, bd=0)
            lbl_name.pack(fill="x", expand=True)
            lbl_name.bind("<Button-1>", lambda e, path=p: self.open_ref_popup(path))

    def open_ref_popup(self, path):
        info = all_files[path]
        popup = tk.Toplevel(self.root)
        popup.title(info["name"])
        popup.geometry("300x180")
        popup.resizable(False, False)
        popup.transient(self.root)

        ttk.Label(popup, text=info["name"], font=("Arial", 10, "bold"), padding=10,
                  wraplength=280).pack(fill="x")
        ttk.Separator(popup, orient="horizontal").pack(fill="x")
        btns = [
            ("파일명 복사", lambda: self.copy_filename(path)),
            ("내용 복사", lambda: self.copy_content(path)),
            ("참조 탐색", lambda: self.show_references(path)),
        ]
        for text, cmd in btns:
            ttk.Button(popup, text=text, command=lambda c=cmd: (c(), popup.destroy())).pack(fill="x", padx=10, pady=4)
        ttk.Button(popup, text="닫기", command=popup.destroy).pack(fill="x", padx=10, pady=(4, 10))

    def open_file_popup(self, path):
        info = all_files[path]
        popup = tk.Toplevel(self.root)
        popup.title(info["name"])
        popup.geometry("300x260")
        popup.resizable(False, False)
        popup.transient(self.root)

        ttk.Label(popup, text=info["name"], font=("Arial", 10, "bold"), padding=10,
                  wraplength=280).pack(fill="x")
        ttk.Label(popup, text=info["relative"], foreground="gray", padding=(10, 0, 10, 10),
                  wraplength=280).pack(fill="x")
        ttk.Separator(popup, orient="horizontal").pack(fill="x")

        btns = [
            ("파일명 복사", lambda: self.copy_filename(path)),
            ("내용 복사", lambda: self.copy_content(path)),
            ("경로 복사", lambda: self.copy_path(path)),
            ("실제 파일 복사", lambda: self.copy_real_file(path)),
            ("참조 탐색", lambda: self.show_references(path)),
        ]
        for text, cmd in btns:
            ttk.Button(popup, text=text, command=lambda c=cmd: (c(), popup.destroy())).pack(fill="x", padx=10, pady=4)

        ttk.Button(popup, text="닫기", command=popup.destroy).pack(fill="x", padx=10, pady=(4, 10))

    # [수정된 파일 목록 렌더링] 체크박스와 함께 표시
    def refresh_file_list(self):
        for widget in self.file_rows_frame.winfo_children():
            widget.destroy()

        query = self.search_var.get().strip().lower()

        self.filtered_files = [
            p for p, info in all_files.items()
            if not query or query in info["name"].lower() or query in info["relative"].lower()
        ]

        if not self.filtered_files:
            ttk.Label(self.file_rows_frame, text="표시할 파일이 없습니다.", padding=20).pack()
            return

        for path in self.filtered_files:
            # 체크박스 변수 초기화
            if path not in self.file_vars:
                self.file_vars[path] = tk.BooleanVar(value=False)

            info = all_files[path]
            
            # 가로로 체크박스와 라벨을 나열하기 위한 프레임
            row_frame = ttk.Frame(self.file_rows_frame)
            row_frame.pack(fill="x", expand=True)

            # 체크박스 배치
            cb = ttk.Checkbutton(row_frame, variable=self.file_vars[path])
            cb.pack(side="left", padx=(5, 0))

            # 라벨 배치 (클릭 시 기존처럼 팝업 열림)
            lbl_name = tk.Label(
                row_frame,
                text=info["name"],
                fg="blue",
                cursor="hand2",
                anchor="w",
                font=("Arial", 10),
                padx=6,
                pady=2,
                bd=0,
                height=1,
            )
            lbl_name.pack(side="left", fill="x", expand=True)
            lbl_name.bind("<Button-1>", lambda e, p=path: self.open_file_popup(p))

# ============================================================
# 메인 실행
# ============================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectAnalyzerApp(root)
    root.mainloop()