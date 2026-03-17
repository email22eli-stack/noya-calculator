import tkinter as tk
from tkinter import ttk
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sympy import (symbols, sympify, diff, solve, limit, oo, S, E, pi,
                   Rational, lambdify)
from sympy.calculus.util import continuous_domain


# ── Basic math functions ───────────────────────────────────────────────────────

def add(a, b):      return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

def calculate(a, operator, b):
    ops = {'+': add, '-': subtract, '*': multiply, '/': divide}
    if operator not in ops:
        raise ValueError(f"Unknown operator '{operator}'")
    return ops[operator](a, b)

def solve_quadratic(a, b, c):
    if a == 0:
        raise ValueError("'a' cannot be 0")
    disc = b**2 - 4*a*c
    if disc > 0:
        x1 = (-b + math.sqrt(disc)) / (2*a)
        x2 = (-b - math.sqrt(disc)) / (2*a)
        return {"type": "two_real", "x1": x1, "x2": x2, "discriminant": disc}
    elif disc == 0:
        return {"type": "one_real", "x1": -b/(2*a), "discriminant": disc}
    else:
        return {"type": "complex", "real": -b/(2*a),
                "imag": math.sqrt(-disc)/(2*a), "discriminant": disc}


# ── Formula data ───────────────────────────────────────────────────────────────

FORMULA_DATA = {
    "אלגברה": [
        ("כפל מקוצר",       "(a+b)² = a² + 2ab + b²"),
        ("כפל מקוצר",       "(a-b)² = a² - 2ab + b²"),
        ("כפל מקוצר",       "(a+b)(a-b) = a² - b²"),
        ("כפל מקוצר",       "(a+b)³ = a³ + 3a²b + 3ab² + b³"),
        ("כפל מקוצר",       "(a-b)³ = a³ - 3a²b + 3ab² - b³"),
        ("משוואה ריבועית",  "x = (-b ± √(b²-4ac)) / 2a"),
        ("דיסקרימיננטה",    "Δ = b² - 4ac"),
        ("לוגריתמים",       "logₐ(xy) = logₐx + logₐy"),
        ("לוגריתמים",       "logₐ(x/y) = logₐx - logₐy"),
        ("לוגריתמים",       "logₐ(xⁿ) = n·logₐx"),
        ("לוגריתמים",       "logₐb = log(b) / log(a)"),
        ("חזקות",           "aᵐ · aⁿ = aᵐ⁺ⁿ"),
        ("חזקות",           "(aᵐ)ⁿ = aᵐⁿ"),
        ("חזקות",           "a⁻ⁿ = 1/aⁿ"),
        ("חזקות",           "√a = a^(1/2)"),
    ],
    "סדרות": [
        ("סדרה חשבונית",    "aₙ = a₁ + (n-1)d"),
        ("סדרה חשבונית",    "Sₙ = n·(a₁ + aₙ)/2"),
        ("סדרה חשבונית",    "Sₙ = n·(2a₁ + (n-1)d) / 2"),
        ("סדרה הנדסית",     "aₙ = a₁ · qⁿ⁻¹"),
        ("סדרה הנדסית",     "Sₙ = a₁·(qⁿ - 1) / (q - 1)   [q ≠ 1]"),
        ("סדרה הנדסית",     "S∞ = a₁ / (1 - q)   [|q| < 1]"),
        ("צמיחה ודעיכה",    "A(t) = A₀ · (1 + r)ᵗ"),
        ("צמיחה ודעיכה",    "A(t) = A₀ · eᵏᵗ"),
    ],
    "טריגונומטריה": [
        ("זהויות יסוד",     "sin²x + cos²x = 1"),
        ("זהויות יסוד",     "tan x = sin x / cos x"),
        ("זהויות יסוד",     "1 + tan²x = 1/cos²x"),
        ("זוויות כפולות",   "sin(2x) = 2·sin x·cos x"),
        ("זוויות כפולות",   "cos(2x) = cos²x - sin²x"),
        ("זוויות כפולות",   "cos(2x) = 1 - 2sin²x"),
        ("סכום זוויות",     "sin(A±B) = sinA·cosB ± cosA·sinB"),
        ("סכום זוויות",     "cos(A±B) = cosA·cosB ∓ sinA·sinB"),
        ("משפטי מישור",     "a/sinA = b/sinB = c/sinC  (סינוסים)"),
        ("משפטי מישור",     "a² = b²+c² - 2bc·cosA  (קוסינוסים)"),
        ("שטחים",           "S△ = ½·a·b·sinC"),
        ("קשת ומגזר",       "L = r·α   (אורך קשת, α בראדיאנים)"),
        ("קשת ומגזר",       "S = ½·r²·α  (שטח מגזר)"),
        ("ערכים מיוחדים",   "sin30°=½ , cos30°=√3/2 , tan30°=1/√3"),
        ("ערכים מיוחדים",   "sin45°=√2/2 , cos45°=√2/2 , tan45°=1"),
        ("ערכים מיוחדים",   "sin60°=√3/2 , cos60°=½ , tan60°=√3"),
    ],
    "גיאומטריה אנליטית": [
        ("מרחק ואמצע",      "d = √((x₂-x₁)² + (y₂-y₁)²)"),
        ("מרחק ואמצע",      "M = ((x₁+x₂)/2 , (y₁+y₂)/2)"),
        ("ישר",             "שיפוע: m = (y₂-y₁) / (x₂-x₁)"),
        ("ישר",             "y = mx + b"),
        ("ישר",             "y - y₁ = m(x - x₁)"),
        ("ישר",             "ישרים מאונכים: m₁ · m₂ = -1"),
        ("ישר",             "ישרים מקבילים: m₁ = m₂"),
        ("מעגל",            "(x-a)² + (y-b)² = r²"),
        ("מרחק נקודה מישר", "d = |ax₀ + by₀ + c| / √(a²+b²)"),
        ("פרבולה",          "y = ax² + bx + c   ,   קודקוד: x = -b/2a"),
    ],
    'חדו"א': [
        ("נגזרות יסוד",     "(c)' = 0"),
        ("נגזרות יסוד",     "(xⁿ)' = n·xⁿ⁻¹"),
        ("נגזרות יסוד",     "(eˣ)' = eˣ"),
        ("נגזרות יסוד",     "(aˣ)' = aˣ·ln a"),
        ("נגזרות יסוד",     "(ln x)' = 1/x"),
        ("נגזרות יסוד",     "(sin x)' = cos x"),
        ("נגזרות יסוד",     "(cos x)' = -sin x"),
        ("נגזרות יסוד",     "(tan x)' = 1/cos²x"),
        ("כללי נגזרת",      "(u·v)' = u'v + uv'"),
        ("כללי נגזרת",      "(u/v)' = (u'v - uv') / v²"),
        ("כללי נגזרת",      "(f(g(x)))' = f'(g(x))·g'(x)  (כלל השרשרת)"),
        ("אינטגרלים יסוד",  "∫xⁿ dx = xⁿ⁺¹/(n+1) + C   [n ≠ -1]"),
        ("אינטגרלים יסוד",  "∫eˣ dx = eˣ + C"),
        ("אינטגרלים יסוד",  "∫(1/x) dx = ln|x| + C"),
        ("אינטגרלים יסוד",  "∫sin x dx = -cos x + C"),
        ("אינטגרלים יסוד",  "∫cos x dx = sin x + C"),
        ("אינטגרל מוגדר",   "∫[a→b] f(x)dx = F(b) - F(a)"),
        ("שטח",             "S = ∫[a→b] |f(x)| dx"),
        ("חקר פונקציה",     "f'(x)=0  ←  נקודת קיצון אפשרית"),
        ("חקר פונקציה",     "f''(x)>0  ←  מינימום מקומי"),
        ("חקר פונקציה",     "f''(x)<0  ←  מקסימום מקומי"),
        ("חקר פונקציה",     "f''(x)=0  ←  נקודת פיתול אפשרית"),
    ],
    "וקטורים": [
        ("אורך וקטור",      "|v| = √(a² + b² + c²)"),
        ("מכפלה סקלרית",    "u·v = a₁a₂ + b₁b₂ + c₁c₂"),
        ("זווית",           "cos θ = (u·v) / (|u|·|v|)"),
        ("אנכיים/מקבילים",  "מאונכים: u·v = 0   |   מקבילים: u = k·v"),
        ("מישור",           "ax + by + cz + d = 0   ,   נורמל: n=(a,b,c)"),
        ("מרחק נקודה ממישור","d = |ax₀+by₀+cz₀+d| / √(a²+b²+c²)"),
        ("זוויות",          "sin α = |n·v| / (|n|·|v|)  (ישר-מישור)"),
        ("זוויות",          "cos θ = |n₁·n₂| / (|n₁|·|n₂|)  (מישור-מישור)"),
    ],
    "מספרים מרוכבים": [
        ("הגדרה",           "i² = -1   ,   z = a + bi"),
        ("ערך מוחלט",       "|z| = √(a² + b²)"),
        ("צמוד",            "z̄ = a - bi"),
        ("כפל",             "(a+bi)(c+di) = (ac-bd) + (ad+bc)i"),
        ("חילוק",           "(a+bi)/(c+di) = [(a+bi)(c-di)] / (c²+d²)"),
        ("צורה טריגונומטרית","z = r(cos θ + i·sin θ)   ,   r=|z|"),
        ("כפל טריג'",       "z₁·z₂ = r₁r₂[cos(θ₁+θ₂) + i·sin(θ₁+θ₂)]"),
        ("משפט דה-מואבר",   "zⁿ = rⁿ(cos nθ + i·sin nθ)"),
        ("שורשי יחידה",     "wₖ = cos(2πk/n) + i·sin(2πk/n)   k=0..n-1"),
    ],
}


# ── Function Analysis ──────────────────────────────────────────────────────────

class FunctionAnalysisDialog:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("חקר פונקציה — בגרות 5 יחידות")
        self.window.configure(bg="#1e1e2e")
        self.window.geometry("680x720")
        self.window.minsize(560, 500)
        self._current_func = None
        self._build_ui()

    def _build_ui(self):
        # Input row
        top = tk.Frame(self.window, bg="#1e1e2e")
        top.pack(fill="x", padx=12, pady=(12, 4))

        tk.Label(top, text="f(x) =", font=("Arial", 14, "bold"),
                 bg="#1e1e2e", fg="#89b4fa").pack(side="left", padx=(0, 8))

        self.func_entry = tk.Entry(top, font=("Consolas", 14),
                                   bg="#313244", fg="white",
                                   insertbackground="white", bd=0, relief="flat")
        self.func_entry.pack(side="left", fill="x", expand=True, ipady=7)
        self.func_entry.insert(0, "x**3 - 3*x + 2")
        self.func_entry.bind("<Return>", lambda _: self._analyze())

        tk.Button(top, text="  חקור  ", font=("Arial", 12, "bold"),
                  bg="#a6e3a1", fg="#1e1e2e", bd=0, relief="flat",
                  padx=10, pady=6, command=self._analyze).pack(side="left", padx=(8, 0))

        tk.Label(self.window,
                 text="דוגמאות:  x**3-3*x+2  |  sin(x)  |  1/(x-1)  |  exp(x)  |  log(x)  |  sqrt(x)",
                 font=("Arial", 9), bg="#1e1e2e", fg="#6c7086").pack(anchor="w", padx=14)

        # Results area
        rf = tk.Frame(self.window, bg="#1e1e2e")
        rf.pack(fill="both", expand=True, padx=12, pady=8)

        self.result_text = tk.Text(rf, font=("Consolas", 11), bg="#181825", fg="#cdd6f4",
                                   bd=0, relief="flat", wrap="word", state="disabled")
        sb = ttk.Scrollbar(rf, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.result_text.pack(side="left", fill="both", expand=True, padx=(0, 4))

        self.result_text.tag_configure("title",     font=("Arial", 13, "bold"), foreground="#cba6f7")
        self.result_text.tag_configure("section",   font=("Arial", 12, "bold"), foreground="#89b4fa")
        self.result_text.tag_configure("result",    foreground="#a6e3a1")
        self.result_text.tag_configure("warn",      foreground="#fab387")
        self.result_text.tag_configure("error",     foreground="#f38ba8")
        self.result_text.tag_configure("separator", foreground="#45475a")

        # Graph button
        self.graph_btn = tk.Button(
            self.window, text="📈  הצג גרף", font=("Arial", 12, "bold"),
            bg="#89b4fa", fg="#1e1e2e", bd=0, relief="flat", height=2,
            state="disabled", command=self._show_graph)
        self.graph_btn.pack(fill="x", padx=12, pady=(0, 10))

        self.func_entry.focus()

    # ── Analysis engine ────────────────────────────────────────────────────────

    def _analyze(self):
        expr_str = self.func_entry.get().strip()
        if not expr_str:
            return
        self._clear()
        self._w(f"חקר פונקציה:  f(x) = {expr_str}\n", "title")
        self._w("━" * 52 + "\n", "separator")

        try:
            xv = symbols('x', real=True)
            f  = sympify(expr_str, locals={'x': xv, 'e': E, 'pi': pi,
                                           'log': __import__('sympy').log,
                                           'ln': __import__('sympy').log})
            f1 = diff(f, xv)
            f2 = diff(f1, xv)
            self._do_analysis(xv, f, f1, f2)
            self._current_func = (xv, f, f1, f2)
            self.graph_btn.config(state="normal")
        except Exception as e:
            self._w(f"\nשגיאה: {e}\n", "error")
            self._w("נסה:  x**2  |  sin(x)  |  exp(x)  |  log(x)  |  1/(x-1)\n", "warn")

    def _do_analysis(self, x, f, f1, f2):
        # 1. Domain
        self._w("\n1.  תחום הגדרה\n", "section")
        try:
            dom = continuous_domain(f, x, S.Reals)
            self._w(f"    D(f) = {dom}\n", "result")
        except Exception:
            self._w("    D(f) = ℝ\n", "result")

        # 2. Intercepts
        self._w("\n2.  חיתוכים עם הצירים\n", "section")
        try:
            y0 = f.subs(x, 0)
            self._w(f"    ציר y:  (0,  {self._fmt(y0)})\n", "result")
        except Exception:
            self._w("    ציר y:  לא מוגדרת ב-x=0\n", "warn")
        try:
            xi_list = [v for v in solve(f, x) if v.is_real]
            if xi_list:
                pts = "  ,  ".join(f"({self._fmt(v)}, 0)" for v in sorted(xi_list, key=float))
                self._w(f"    ציר x:  {pts}\n", "result")
            else:
                self._w("    ציר x:  אין חיתוכים ממשיים\n", "result")
        except Exception:
            self._w("    ציר x:  לא ניתן לחשב אנליטית\n", "warn")

        # 3. Asymptotes
        self._w("\n3.  אסימפטוטות\n", "section")
        self._asymptotes(x, f)

        # 4. First derivative
        self._w("\n4.  נגזרת ראשונה\n", "section")
        self._w(f"    f'(x) = {f1}\n", "result")

        # 5. Critical points
        self._w("\n5.  נקודות קיצון  (f'(x) = 0)\n", "section")
        critical = []
        try:
            critical = sorted([v for v in solve(f1, x) if v.is_real], key=float)
            if not critical:
                self._w("    אין נקודות קיצון\n", "result")
            for cp in critical:
                fy  = self._fmt(f.subs(x, cp))
                f2v = f2.subs(x, cp)
                try:
                    f2f = float(f2v)
                    if f2f < 0:
                        kind, col = "מקסימום מקומי", "result"
                    elif f2f > 0:
                        kind, col = "מינימום מקומי", "result"
                    else:
                        kind, col = "בדוק לוח סימנים", "warn"
                except Exception:
                    kind, col = "בדוק לוח סימנים", "warn"
                self._w(f"    x = {self._fmt(cp)}  →  f = {fy}    ({kind})\n", col)
        except Exception as e:
            self._w(f"    לא ניתן לחשב: {e}\n", "warn")

        # 6. Sign table f'
        self._w("\n6.  תחומי עלייה וירידה\n", "section")
        self._sign_table(x, f, f1, critical, prime="'",
                         pos_desc="↗  עולה", neg_desc="↘  יורדת")

        # 7. Second derivative
        self._w("\n7.  נגזרת שנייה\n", "section")
        self._w(f"    f''(x) = {f2}\n", "result")

        # 8. Inflection points
        self._w("\n8.  נקודות פיתול  (f''(x) = 0)\n", "section")
        inflection = []
        try:
            cands = sorted([v for v in solve(f2, x) if v.is_real], key=float)
            for ip in cands:
                try:
                    lv = float(f2.subs(x, ip - Rational(1, 100)))
                    rv = float(f2.subs(x, ip + Rational(1, 100)))
                    if lv * rv < 0:
                        inflection.append(ip)
                        fy = self._fmt(f.subs(x, ip))
                        self._w(f"    x = {self._fmt(ip)}  →  f = {fy}\n", "result")
                except Exception:
                    pass
            if not inflection:
                self._w("    אין נקודות פיתול\n", "result")
        except Exception as e:
            self._w(f"    לא ניתן לחשב: {e}\n", "warn")

        # 9. Concavity
        self._w("\n9.  תחומי קעירות וקמירות\n", "section")
        self._sign_table(x, f, f2, inflection, prime="''",
                         pos_desc="קמורה  ∪", neg_desc="קעורה  ∩")

        self._w("\n" + "━" * 52 + "\n", "separator")
        self._w("לחץ על 'הצג גרף' לתצוגה גרפית של הפונקציה\n", "warn")

    def _asymptotes(self, x, f):
        found = False
        # Vertical
        try:
            _, d = f.as_numer_denom()
            if not d.is_number:
                for va in solve(d, x):
                    if va.is_real:
                        self._w(f"    אסימפטוטה אנכית:  x = {self._fmt(va)}\n", "result")
                        found = True
        except Exception:
            pass
        # Horizontal
        try:
            lp = limit(f, x,  oo)
            ln = limit(f, x, -oo)
            if lp.is_finite:
                self._w(f"    אסימפטוטה אופקית (x→+∞):  y = {self._fmt(lp)}\n", "result")
                found = True
            if ln.is_finite and ln != lp:
                self._w(f"    אסימפטוטה אופקית (x→-∞):  y = {self._fmt(ln)}\n", "result")
                found = True
        except Exception:
            pass
        if not found:
            self._w("    אין אסימפטוטות\n", "result")

    def _sign_table(self, x, f, expr, key_pts, prime, pos_desc, neg_desc):
        try:
            kf = sorted([float(p) for p in key_pts])
            pts = [-float('inf')] + kf + [float('inf')]
            for i in range(len(pts) - 1):
                L, R = pts[i], pts[i+1]
                test = (0.0           if L == -float('inf') and R == float('inf') else
                        R - 1.0       if L == -float('inf') else
                        L + 1.0       if R == float('inf')  else
                        (L + R) / 2.0)
                try:
                    sign = "+" if float(expr.subs(x, test)) > 0 else "−"
                except Exception:
                    sign = "?"
                Ls = "−∞"  if L == -float('inf') else self._fmt_f(L)
                Rs = "+∞"  if R ==  float('inf') else self._fmt_f(R)
                desc = pos_desc if sign == "+" else neg_desc
                self._w(f"    ({Ls}, {Rs}):  f{prime}(x) = {sign}  →  {desc}\n", "result")

                if R != float('inf') and i < len(pts) - 2:
                    kp = kf[i]
                    try:
                        fy = self._fmt(f.subs(x, kp))
                        self._w(f"    x = {self._fmt_f(kp)}:  f{prime}(x) = 0  →  f(x) = {fy}\n", "warn")
                    except Exception:
                        self._w(f"    x = {self._fmt_f(kp)}:  f{prime}(x) = 0\n", "warn")
        except Exception as e:
            self._w(f"    לא ניתן לחשב: {e}\n", "warn")

    # ── Graph ──────────────────────────────────────────────────────────────────

    def _show_graph(self):
        if not self._current_func:
            return
        x, f, f1, f2 = self._current_func
        expr_str = self.func_entry.get().strip()

        try:
            f_np = lambdify(x, f, modules=['numpy'])
            xv   = np.linspace(-10, 10, 2000)
            with np.errstate(divide='ignore', invalid='ignore'):
                yv = np.array(f_np(xv), dtype=float)
            # Clip extreme values to avoid distorted graphs
            yv[np.abs(yv) > 100] = np.nan

            fig, ax = plt.subplots(figsize=(8, 6))
            fig.patch.set_facecolor('#1e1e2e')
            ax.set_facecolor('#181825')
            for spine in ax.spines.values():
                spine.set_color('#45475a')
            ax.tick_params(colors='#cdd6f4')
            ax.grid(color='#313244', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.axhline(0, color='#6c7086', linewidth=1)
            ax.axvline(0, color='#6c7086', linewidth=1)

            ax.plot(xv, yv, color='#89b4fa', linewidth=2.2, label='f(x)', zorder=3)

            # Critical points
            try:
                for cp in solve(f1, x):
                    if cp.is_real:
                        cx, cy = float(cp), float(f.subs(x, cp))
                        if abs(cy) <= 100:
                            f2v = float(f2.subs(x, cp))
                            col = '#f38ba8' if f2v < 0 else '#a6e3a1'
                            lbl = 'מקסימום' if f2v < 0 else 'מינימום'
                            ax.plot(cx, cy, 'o', color=col, markersize=9, zorder=5)
                            ax.annotate(f'{lbl}\n({cx:.2f}, {cy:.2f})', (cx, cy),
                                        xytext=(10, 10), textcoords='offset points',
                                        color=col, fontsize=8,
                                        arrowprops=dict(arrowstyle='->', color=col, lw=0.8))
            except Exception:
                pass

            # Inflection points
            try:
                for ip in solve(f2, x):
                    if ip.is_real:
                        ix, iy = float(ip), float(f.subs(x, ip))
                        lv = float(f2.subs(x, ip - Rational(1, 100)))
                        rv = float(f2.subs(x, ip + Rational(1, 100)))
                        if lv * rv < 0 and abs(iy) <= 100:
                            ax.plot(ix, iy, 's', color='#fab387', markersize=8, zorder=5)
                            ax.annotate(f'פיתול\n({ix:.2f}, {iy:.2f})', (ix, iy),
                                        xytext=(-50, 12), textcoords='offset points',
                                        color='#fab387', fontsize=8,
                                        arrowprops=dict(arrowstyle='->', color='#fab387', lw=0.8))
            except Exception:
                pass

            ax.set_title(f'f(x) = {expr_str}', color='#cdd6f4', fontsize=13, pad=12)
            ax.set_xlabel('x', color='#cdd6f4', fontsize=11)
            ax.set_ylabel('f(x)', color='#cdd6f4', fontsize=11)
            ax.legend(facecolor='#313244', edgecolor='#45475a', labelcolor='#cdd6f4')

            gwin = tk.Toplevel(self.window)
            gwin.title(f"גרף: f(x) = {expr_str}")
            gwin.configure(bg='#1e1e2e')
            canvas = FigureCanvasTkAgg(fig, master=gwin)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            gwin.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), gwin.destroy()))

        except Exception as e:
            self._w(f"\nשגיאה ביצירת הגרף: {e}\n", "error")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _w(self, text, tag=None, clear=False):
        self.result_text.config(state="normal")
        if clear:
            self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text, tag or "")
        self.result_text.config(state="disabled")
        self.result_text.see(tk.END)

    def _clear(self):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")

    def _fmt(self, val):
        try:
            f = float(val)
            return str(int(f)) if f == int(f) else f"{f:.4f}".rstrip('0').rstrip('.')
        except Exception:
            return str(val)

    def _fmt_f(self, val):
        try:
            f = float(val)
            return str(int(f)) if f == int(f) else f"{f:.4f}".rstrip('0').rstrip('.')
        except Exception:
            return str(val)


# ── Formula Sheet ──────────────────────────────────────────────────────────────

class FormulaSheet:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("דף נוסחאות — בגרות מתמטיקה 5 יחידות")
        self.window.configure(bg="#1e1e2e")
        self.window.geometry("620x560")
        self.window.minsize(500, 400)
        self._all = [(c, l, f) for c, items in FORMULA_DATA.items() for l, f in items]
        self._build_ui()

    def _build_ui(self):
        sf = tk.Frame(self.window, bg="#1e1e2e")
        sf.pack(fill="x", padx=12, pady=(10, 6))
        tk.Label(sf, text="🔍", font=("Arial", 14), bg="#1e1e2e", fg="#cdd6f4").pack(side="left", padx=(0, 6))
        self.sv = tk.StringVar()
        self.sv.trace("w", self._on_search)
        tk.Entry(sf, textvariable=self.sv, font=("Arial", 13),
                 bg="#313244", fg="white", insertbackground="white",
                 bd=0, relief="flat").pack(side="left", fill="x", expand=True, ipady=6)

        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",     background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4",
                        padding=[10, 5], font=("Arial", 11, "bold"))
        style.map("TNotebook.Tab", background=[("selected","#89b4fa")],
                                   foreground=[("selected","#1e1e2e")])

        self.nb = ttk.Notebook(self.window)
        self.nb.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        self._tabs = {}
        for cat in FORMULA_DATA:
            inner = self._make_tab(cat)
            self._tabs[cat] = inner
            self._populate(inner, cat)

        self._search_frame = self._make_tab("תוצאות", attach=False)
        self._search_active = False

        self.status = tk.StringVar(value="לחץ על נוסחה כדי להעתיק ללוח")
        tk.Label(self.window, textvariable=self.status, font=("Arial", 10),
                 bg="#181825", fg="#6c7086", anchor="w").pack(fill="x", padx=12, pady=(0, 6))

    def _make_tab(self, title, attach=True):
        outer  = tk.Frame(self.nb, bg="#1e1e2e")
        canvas = tk.Canvas(outer, bg="#1e1e2e", highlightthickness=0)
        sb     = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg="#1e1e2e")
        inner.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>",
            lambda ev: canvas.yview_scroll(-1*(ev.delta//120), "units")))
        canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
        if attach:
            self.nb.add(outer, text=title)
        return inner

    def _populate(self, frame, category):
        group = None
        for label, formula in FORMULA_DATA[category]:
            if label != group:
                group = label
                tk.Label(frame, text=label, font=("Arial", 10, "bold"),
                         bg="#1e1e2e", fg="#6c7086", anchor="w").pack(fill="x", padx=14, pady=(10, 2))
            self._row(frame, formula)

    def _row(self, parent, formula):
        row = tk.Frame(parent, bg="#313244", cursor="hand2")
        row.pack(fill="x", padx=10, pady=2, ipady=4)
        lbl = tk.Label(row, text=formula, font=("Consolas", 12),
                       bg="#313244", fg="#cdd6f4", anchor="w", padx=10)
        lbl.pack(side="left", fill="x", expand=True)
        ico = tk.Label(row, text="📋", font=("Arial", 12), bg="#313244", fg="#6c7086", padx=8)
        ico.pack(side="right")

        def copy(f=formula, i=ico):
            self.window.clipboard_clear()
            self.window.clipboard_append(f)
            i.config(fg="#a6e3a1")
            self.status.set(f"הועתק: {f}")
            self.window.after(1500, lambda: (i.config(fg="#6c7086"),
                              self.status.set("לחץ על נוסחה כדי להעתיק ללוח")))

        for w in (row, lbl, ico):
            w.bind("<Button-1>", lambda _, fn=copy: fn())
            w.bind("<Enter>",    lambda _, r=row: r.config(bg="#45475a"))
            w.bind("<Leave>",    lambda _, r=row: r.config(bg="#313244"))

    def _on_search(self, *_):
        q = self.sv.get().strip().lower()
        if not q:
            if self._search_active:
                self.nb.forget(self._search_frame.master.master)
                self._search_active = False
                for cat, frm in self._tabs.items():
                    self.nb.add(frm.master.master, text=cat)
            return
        if not self._search_active:
            for frm in self._tabs.values():
                self.nb.hide(frm.master.master)
            self.nb.add(self._search_frame.master.master, text="תוצאות חיפוש")
            self._search_active = True
        for w in self._search_frame.winfo_children():
            w.destroy()
        matches = [(c, l, f) for c, l, f in self._all
                   if q in f.lower() or q in l.lower() or q in c.lower()]
        if matches:
            for cat, label, formula in matches:
                tk.Label(self._search_frame, text=f"{cat} › {label}",
                         font=("Arial", 9), bg="#1e1e2e", fg="#6c7086",
                         anchor="w").pack(fill="x", padx=14, pady=(8, 1))
                self._row(self._search_frame, formula)
        else:
            tk.Label(self._search_frame, text="לא נמצאו תוצאות",
                     font=("Arial", 13), bg="#1e1e2e", fg="#6c7086").pack(pady=40)


# ── Quadratic Dialog ───────────────────────────────────────────────────────────

class QuadraticDialog:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("פותר משוואה ריבועית")
        self.window.configure(bg="#1e1e2e")
        self.window.resizable(False, False)
        self.window.grab_set()
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.window, text="ax² + bx + c = 0",
                 font=("Arial", 16, "bold"), bg="#1e1e2e", fg="#cdd6f4"
                 ).grid(row=0, column=0, columnspan=3, pady=(15, 10))
        self.entries = {}
        for col, label in enumerate(["a", "b", "c"]):
            tk.Label(self.window, text=label, font=("Arial", 14, "bold"),
                     bg="#1e1e2e", fg="#89b4fa", width=4).grid(row=1, column=col, padx=10)
            e = tk.Entry(self.window, font=("Arial", 16), justify="center",
                         bg="#313244", fg="white", insertbackground="white",
                         width=8, bd=0, relief="flat")
            e.grid(row=2, column=col, padx=10, pady=5, ipady=6)
            self.entries[label] = e
        self.entries["a"].focus()
        tk.Button(self.window, text="פתור", font=("Arial", 14, "bold"),
                  bg="#f38ba8", fg="white", bd=0, relief="flat", width=10, height=2,
                  command=self._solve).grid(row=3, column=0, columnspan=3, pady=12)
        self.res = tk.Label(self.window, text="", font=("Arial", 13),
                            bg="#1e1e2e", fg="#a6e3a1", justify="center", wraplength=300)
        self.res.grid(row=4, column=0, columnspan=3, pady=(0, 15), padx=15)

    def _solve(self):
        try:
            a = float(self.entries["a"].get())
            b = float(self.entries["b"].get())
            c = float(self.entries["c"].get())
        except ValueError:
            self.res.config(text="נא להזין מספרים תקינים.", fg="#f38ba8")
            return
        try:
            r = solve_quadratic(a, b, c)
        except ValueError as e:
            self.res.config(text=str(e), fg="#f38ba8")
            return
        d = r["discriminant"]
        if r["type"] == "two_real":
            t = f"Δ = {self._f(d)}  (שני פתרונות)\n\nx₁ = {self._f(r['x1'])}\nx₂ = {self._f(r['x2'])}"
            self.res.config(text=t, fg="#a6e3a1")
        elif r["type"] == "one_real":
            t = f"Δ = 0  (פתרון יחיד)\n\nx = {self._f(r['x1'])}"
            self.res.config(text=t, fg="#a6e3a1")
        else:
            t = (f"Δ = {self._f(d)}  (פתרונות מרוכבים)\n\n"
                 f"x₁ = {self._f(r['real'])} + {self._f(r['imag'])}i\n"
                 f"x₂ = {self._f(r['real'])} − {self._f(r['imag'])}i")
            self.res.config(text=t, fg="#fab387")

    def _f(self, v):
        return str(int(v)) if v == int(v) else f"{v:.4f}"


# ── Main Calculator ────────────────────────────────────────────────────────────

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("מחשבון — בגרות מתמטיקה 5 יחידות")
        self.root.resizable(False, False)
        self.expression = ""
        self._build_ui()

    def _build_ui(self):
        self.display = tk.Entry(
            self.root, font=("Arial", 24), justify="right",
            bd=10, relief="flat", bg="#1e1e2e", fg="white", insertbackground="white")
        self.display.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=10, pady=(10, 5))

        buttons = [
            ("C",       1, 0, 1), ("⌫",      1, 1, 1), ("%",       1, 2, 1), ("/",  1, 3, 1),
            ("7",       2, 0, 1), ("8",       2, 1, 1), ("9",       2, 2, 1), ("*",  2, 3, 1),
            ("4",       3, 0, 1), ("5",       3, 1, 1), ("6",       3, 2, 1), ("-",  3, 3, 1),
            ("1",       4, 0, 1), ("2",       4, 1, 1), ("3",       4, 2, 1), ("+",  4, 3, 1),
            ("0",       5, 0, 1), (".",       5, 1, 1), ("+/-",     5, 2, 1), ("=",  5, 3, 1),
            ("x²",      6, 0, 1), ("נוסחאות", 6, 1, 2),
            ("חקר f(x)", 7, 0, 4),
        ]
        for item in buttons:
            self._make_button(*item)

    def _make_button(self, text, row, col, span=1):
        colors = {
            "=":         ("#f38ba8", "white"),
            "C":         ("#45475a", "white"),
            "⌫":         ("#45475a", "white"),
            "%":         ("#45475a", "white"),
            "+":         ("#89b4fa", "white"),
            "-":         ("#89b4fa", "white"),
            "*":         ("#89b4fa", "white"),
            "/":         ("#89b4fa", "white"),
            "x²":        ("#a6e3a1", "#1e1e2e"),
            "נוסחאות":   ("#fab387", "#1e1e2e"),
            "חקר f(x)":  ("#cba6f7", "#1e1e2e"),
        }
        bg, fg = colors.get(text, ("#313244", "white"))
        w = 4 * span - (span - 1)
        tk.Button(
            self.root, text=text, font=("Arial", 16, "bold"),
            bg=bg, fg=fg, activebackground="#585b70", activeforeground="white",
            bd=0, relief="flat", width=w, height=2,
            command=lambda t=text: self._on_click(t)
        ).grid(row=row, column=col, columnspan=span, padx=4, pady=4, sticky="nsew")

    def _on_click(self, text):
        if   text == "C":         self.expression = ""
        elif text == "⌫":         self.expression = self.expression[:-1]
        elif text == "=":         self._evaluate(); return
        elif text == "x²":        QuadraticDialog(self.root); return
        elif text == "נוסחאות":   FormulaSheet(self.root); return
        elif text == "חקר f(x)":  FunctionAnalysisDialog(self.root); return
        elif text == "%":
            try:    self.expression = str(float(self.expression) / 100)
            except: self.expression = "Error"
        elif text == "+/-":
            try:    self.expression = str(-float(self.expression))
            except: self.expression = "Error"
        else:
            self.expression += text
        self._update_display(self.expression)

    def _evaluate(self):
        try:
            expr = self.expression
            for op in ['+', '-', '*', '/']:
                idx = expr.rfind(op)
                if idx > 0:
                    result = calculate(float(expr[:idx]), op, float(expr[idx+1:]))
                    self.expression = (str(int(result))
                                       if result == int(result) else str(result))
                    self._update_display(self.expression)
                    return
        except (ValueError, IndexError):
            self._update_display("Error")
            self.expression = ""

    def _update_display(self, value):
        self.display.delete(0, tk.END)
        self.display.insert(0, value)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.configure(bg="#1e1e2e")
    CalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
