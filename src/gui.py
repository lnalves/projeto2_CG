"""Interface gráfica (tkinter) para o pipeline de medição de plântulas.

Uso:
    python gui.py

Requisitos: os arquivos main.py, calibration.py, livewire.py, preprocess.py,
render.py, config.py e measure.py devem estar no mesmo diretório.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk


# ─────────────────────────────────────────────────────────────────
#  Paleta de cores
# ─────────────────────────────────────────────────────────────────
BG        = "#1e1e2e"   # fundo principal
SURFACE   = "#2a2a3e"   # cards / painéis
ACCENT    = "#7c6af7"   # roxo principal
ACCENT2   = "#56cfb2"   # verde-água para sucesso
WARN      = "#f4845f"   # laranja para atenção
TEXT      = "#cdd6f4"   # texto principal
SUBTEXT   = "#6c7086"   # texto secundário
BORDER    = "#383851"   # bordas sutis
RED_BTN   = "#f38ba8"   # botão de perigo


class PlaceholderEntry(tk.Entry):
    """Entry com texto de dica (placeholder) em cinza."""

    def __init__(self, master, placeholder="", *args, **kwargs):
        self.placeholder = placeholder
        self._has_focus = False
        kwargs.setdefault("fg", SUBTEXT)
        kwargs.setdefault("bg", SURFACE)
        kwargs.setdefault("insertbackground", TEXT)
        kwargs.setdefault("relief", "flat")
        kwargs.setdefault("font", ("Segoe UI", 10))
        super().__init__(master, *args, **kwargs)
        self._show_placeholder()
        self.bind("<FocusIn>",  self._on_focus_in)
        self.bind("<FocusOut>", self._on_focus_out)

    def _show_placeholder(self):
        self.insert(0, self.placeholder)
        self["fg"] = SUBTEXT

    def _on_focus_in(self, _):
        if self.get() == self.placeholder:
            self.delete(0, "end")
            self["fg"] = TEXT

    def _on_focus_out(self, _):
        if not self.get():
            self._show_placeholder()

    def real_value(self):
        v = self.get()
        return "" if v == self.placeholder else v


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🌱 Medição de Plântulas — Live-wire")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(700, 580)

        self._image_path = tk.StringVar()
        self._out_dir    = tk.StringVar(value=os.path.join(os.getcwd(), "output"))
        self._known_mm   = tk.StringVar(value="90")
        self._max_dim    = tk.StringVar(value="1600")
        self._debug      = tk.BooleanVar(value=False)
        self._process    = None   # subprocess em execução

        self._build_ui()
        self._center_window()

    # ─────────────────────────────────────────────────────────────
    #  Layout
    # ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Cabeçalho
        hdr = tk.Frame(self, bg=ACCENT, pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🌱  Medição de Plântulas", bg=ACCENT,
                 fg="white", font=("Segoe UI", 16, "bold")).pack()
        tk.Label(hdr, text="Pipeline semiautomático por live-wire (ADR 0001)",
                 bg=ACCENT, fg="#ddd6fe", font=("Segoe UI", 9)).pack()

        # Corpo central
        body = tk.Frame(self, bg=BG, padx=24, pady=18)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)

        # ── Card: Arquivo de entrada ──────────────────────────
        self._card(body, row=0, title="📁  Arquivo de entrada",
                   builder=self._build_file_section)

        # ── Card: Parâmetros ─────────────────────────────────
        self._card(body, row=1, title="⚙️  Parâmetros",
                   builder=self._build_params_section)

        # ── Card: Pasta de saída ─────────────────────────────
        self._card(body, row=2, title="💾  Pasta de saída",
                   builder=self._build_output_section)

        # ── Botões de ação ───────────────────────────────────
        btn_row = tk.Frame(body, bg=BG)
        btn_row.grid(row=3, column=0, sticky="ew", pady=(8, 4))
        btn_row.columnconfigure(0, weight=1)
        btn_row.columnconfigure(1, weight=1)

        self._btn_run = self._make_button(
            btn_row, "▶  Executar Pipeline", self._run, ACCENT, col=0)
        self._btn_stop = self._make_button(
            btn_row, "⏹  Interromper", self._stop, RED_BTN, col=1, state="disabled")

        # ── Console de log ───────────────────────────────────
        log_frame = tk.Frame(body, bg=SURFACE, bd=0, highlightthickness=1,
                             highlightbackground=BORDER)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        body.rowconfigure(4, weight=1)

        tk.Label(log_frame, text=" 📋 Log de execução", bg=SURFACE,
                 fg=SUBTEXT, font=("Segoe UI", 9, "bold"), anchor="w"
                 ).pack(fill="x", padx=8, pady=(6, 0))

        self._log = scrolledtext.ScrolledText(
            log_frame, bg="#13131f", fg=TEXT, insertbackground=TEXT,
            font=("Consolas", 9), relief="flat", state="disabled",
            wrap="word", height=10,
        )
        self._log.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # Tags de cor para o log
        self._log.tag_config("ok",   foreground=ACCENT2)
        self._log.tag_config("err",  foreground=RED_BTN)
        self._log.tag_config("info", foreground=ACCENT)
        self._log.tag_config("warn", foreground=WARN)

        # Barra de status
        status_bar = tk.Frame(self, bg=SURFACE, pady=4)
        status_bar.pack(fill="x", side="bottom")
        self._status = tk.StringVar(value="Pronto.")
        tk.Label(status_bar, textvariable=self._status, bg=SURFACE,
                 fg=SUBTEXT, font=("Segoe UI", 9), anchor="w",
                 padx=12).pack(fill="x")

    def _card(self, parent, row, title, builder):
        frame = tk.Frame(parent, bg=SURFACE, bd=0, highlightthickness=1,
                         highlightbackground=BORDER, padx=14, pady=10)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text=title, bg=SURFACE, fg=ACCENT,
                 font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        builder(frame)

    def _make_button(self, parent, text, cmd, color, col, state="normal"):
        btn = tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="white" if color != RED_BTN else "#1e1e2e",
            activebackground=color, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat",
            padx=20, pady=8, cursor="hand2", state=state,
            bd=0,
        )
        btn.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0))
        return btn

    # ─────────────────────────────────────────────────────────────
    #  Seções dos cards
    # ─────────────────────────────────────────────────────────────
    def _build_file_section(self, frame):
        tk.Label(frame, text="Imagem:", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", padx=(0, 10))

        entry = tk.Entry(frame, textvariable=self._image_path,
                         bg="#13131f", fg=TEXT, insertbackground=TEXT,
                         relief="flat", font=("Segoe UI", 10))
        entry.grid(row=1, column=1, sticky="ew")

        tk.Button(frame, text="Procurar…", command=self._choose_image,
                  bg=BORDER, fg=TEXT, activebackground=ACCENT,
                  font=("Segoe UI", 9), relief="flat", padx=10, pady=4,
                  cursor="hand2", bd=0
                  ).grid(row=1, column=2, padx=(8, 0))

        tk.Label(frame,
                 text="Formatos suportados: PNG, JPG, BMP, TIFF, HEIC (se pillow-heif instalado)",
                 bg=SURFACE, fg=SUBTEXT, font=("Segoe UI", 8)
                 ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

    def _build_params_section(self, frame):
        params = [
            ("Distância conhecida (mm):", self._known_mm,
             "Distância real entre os 2 pontos da calibração manual (padrão: 90)"),
            ("Dimensão máxima (px):",     self._max_dim,
             "Maior lado da imagem após redimensionar. 0 = sem limite (padrão: 1600)"),
        ]
        for i, (label, var, tip) in enumerate(params, start=1):
            tk.Label(frame, text=label, bg=SURFACE, fg=TEXT,
                     font=("Segoe UI", 10)).grid(
                row=i, column=0, sticky="w", padx=(0, 10), pady=3)
            tk.Entry(frame, textvariable=var, bg="#13131f", fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     font=("Segoe UI", 10), width=12).grid(
                row=i, column=1, sticky="w")
            tk.Label(frame, text=tip, bg=SURFACE, fg=SUBTEXT,
                     font=("Segoe UI", 8)).grid(
                row=i, column=2, sticky="w", padx=(12, 0))

        tk.Checkbutton(
            frame, text="Modo debug (salva imagens intermediárias)",
            variable=self._debug, bg=SURFACE, fg=TEXT,
            selectcolor=ACCENT, activebackground=SURFACE,
            font=("Segoe UI", 10), cursor="hand2",
        ).grid(row=len(params) + 1, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _build_output_section(self, frame):
        tk.Label(frame, text="Pasta:", bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", padx=(0, 10))

        tk.Entry(frame, textvariable=self._out_dir,
                 bg="#13131f", fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 10)).grid(
            row=1, column=1, sticky="ew")

        tk.Button(frame, text="Escolher…", command=self._choose_out,
                  bg=BORDER, fg=TEXT, activebackground=ACCENT,
                  font=("Segoe UI", 9), relief="flat", padx=10, pady=4,
                  cursor="hand2", bd=0
                  ).grid(row=1, column=2, padx=(8, 0))

    # ─────────────────────────────────────────────────────────────
    #  Diálogos de arquivo
    # ─────────────────────────────────────────────────────────────
    def _choose_image(self):
        path = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=[
                ("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.heic *.heif"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if path:
            self._image_path.set(path)

    def _choose_out(self):
        path = filedialog.askdirectory(title="Selecionar pasta de saída")
        if path:
            self._out_dir.set(path)

    # ─────────────────────────────────────────────────────────────
    #  Execução do pipeline
    # ─────────────────────────────────────────────────────────────
    def _run(self):
        img = self._image_path.get().strip()
        out = self._out_dir.get().strip()
        mm  = self._known_mm.get().strip()
        dim = self._max_dim.get().strip()

        if not img:
            messagebox.showwarning("Campo vazio", "Selecione uma imagem de entrada.")
            return
        if not os.path.isfile(img):
            messagebox.showerror("Arquivo não encontrado", f"Imagem não encontrada:\n{img}")
            return
        try:
            float(mm)
        except ValueError:
            messagebox.showerror("Valor inválido", "Distância conhecida deve ser um número.")
            return

        # Localiza main.py no mesmo diretório do gui.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        main_py    = os.path.join(script_dir, "main.py")
        if not os.path.isfile(main_py):
            messagebox.showerror(
                "main.py não encontrado",
                f"Não encontrei main.py em:\n{script_dir}\n\n"
                "Certifique-se de que gui.py está na mesma pasta que main.py.",
            )
            return

        cmd = [
            sys.executable, main_py,
            "--image",     img,
            "--out",       out,
            "--known-mm",  mm,
            "--max-dim",   dim or "1600",
        ]
        if self._debug.get():
            cmd.append("--debug")

        self._log_clear()
        self._log_write(f"▶ Executando:\n  {' '.join(cmd)}\n", "info")
        self._status.set("Executando pipeline…")
        self._btn_run.config(state="disabled")
        self._btn_stop.config(state="normal")

        def worker():
            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    cwd=script_dir,
                )
                for line in self._process.stdout:
                    self._log_write(line)
                self._process.wait()
                rc = self._process.returncode
            except Exception as exc:
                self._log_write(f"\nErro ao iniciar processo: {exc}\n", "err")
                rc = -1
            finally:
                self._process = None
                self.after(0, self._on_done, rc)

        threading.Thread(target=worker, daemon=True).start()

    def _stop(self):
        if self._process:
            self._process.terminate()
            self._log_write("\n⏹ Processo interrompido pelo usuário.\n", "warn")
        self._btn_run.config(state="normal")
        self._btn_stop.config(state="disabled")
        self._status.set("Interrompido.")

    def _on_done(self, rc):
        self._btn_run.config(state="normal")
        self._btn_stop.config(state="disabled")
        if rc == 0:
            self._log_write("\n✅ Pipeline concluído com sucesso!\n", "ok")
            self._status.set("Concluído com sucesso.")
            out = self._out_dir.get()
            messagebox.showinfo(
                "Concluído",
                f"Pipeline finalizado!\n\nOs resultados foram salvos em:\n{out}",
            )
        else:
            self._log_write(f"\n❌ Pipeline encerrou com código {rc}.\n", "err")
            self._status.set(f"Encerrado com erro (código {rc}).")

    # ─────────────────────────────────────────────────────────────
    #  Log helpers
    # ─────────────────────────────────────────────────────────────
    def _log_write(self, text, tag=None):
        def _insert():
            self._log.config(state="normal")
            # Colorização automática por palavras-chave
            if tag:
                self._log.insert("end", text, tag)
            else:
                low = text.lower()
                if any(w in low for w in ("erro", "error", "falhou", "failed", "traceback")):
                    self._log.insert("end", text, "err")
                elif any(w in low for w in ("aviso", "warning", "warn")):
                    self._log.insert("end", text, "warn")
                elif any(w in low for w in ("calibra", "escala", "mm/px", "px/mm")):
                    self._log.insert("end", text, "info")
                else:
                    self._log.insert("end", text)
            self._log.see("end")
            self._log.config(state="disabled")

        self.after(0, _insert)

    def _log_clear(self):
        self._log.config(state="normal")
        self._log.delete("1.0", "end")
        self._log.config(state="disabled")

    # ─────────────────────────────────────────────────────────────
    #  Utilitários
    # ─────────────────────────────────────────────────────────────
    def _center_window(self):
        self.update_idletasks()
        w, h = 780, 640
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w)//2}+{(sh - h)//2}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
