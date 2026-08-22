"""Stage 9 -- turn the report into a shareable artifact page.

The published wrapper supplies <!doctype>, <head> and <body>, so this strips our
own skeleton and keeps the content. It also swaps the stylesheet for one that
carries a real token system: the report was written light-only, which is fine on
a local machine and wrong on a page other people open in whatever theme they
happen to use.

Three theme states are handled, not two -- an explicit light choice, an explicit
dark choice, and the default where nothing is stamped and only the OS preference
separates them.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from regime_narrative.config import output_dir

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&"
    'family=Source+Sans+3:wght@400;600;700&display=swap">'
)

# Source Serif 4 / Source Sans 3 -- one superfamily, drawn together, and the
# convention for working papers rather than the default UI sans.
CSS = """
:root{
  --ground:#fbfbfa; --raised:#f3f5f7; --ink:#161f27; --mut:#5c6a76;
  --rule:#dee5ea; --rule-strong:#161f27;
  --accent:#37567f; --warn:#9a5c17; --warn-bg:#fbf4ea; --good:#26694a;
  --good-bg:#eef5f1; --shadow:0 1px 2px rgba(22,31,39,.05);
}
/* Flat form, not CSS nesting: nesting is well supported but not universally,
   and a dropped block here means dark-mode readers get dark-on-dark. */
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#11161b; --raised:#1a2229; --ink:#e2e8ed; --mut:#93a2ae;
    --rule:#27313a; --rule-strong:#e2e8ed;
    --accent:#8aabd6; --warn:#d8a464; --warn-bg:#221a10; --good:#6dbd97;
    --good-bg:#101d17; --shadow:none;
  }
}
:root[data-theme="dark"]{
  --ground:#11161b; --raised:#1a2229; --ink:#e2e8ed; --mut:#93a2ae;
  --rule:#27313a; --rule-strong:#e2e8ed;
  --accent:#8aabd6; --warn:#d8a464; --warn-bg:#221a10; --good:#6dbd97;
  --good-bg:#101d17; --shadow:none;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:"Source Sans 3",-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.wrap{max-width:1120px; margin:0 auto; padding:56px 26px 110px;
  display:flex; flex-direction:column; gap:0;}
h1{font-family:"Source Serif 4",Georgia,serif; font-weight:700;
  font-size:clamp(30px,4.4vw,42px); line-height:1.1; letter-spacing:-.018em;
  margin:0 0 10px; text-wrap:balance;}
h2{font-family:"Source Serif 4",Georgia,serif; font-weight:600;
  font-size:23px; letter-spacing:-.008em; margin:64px 0 6px;
  padding-bottom:9px; border-bottom:1.5px solid var(--rule-strong);
  text-wrap:balance;}
h3{font-family:"Source Serif 4",Georgia,serif; font-weight:600;
  font-size:17px; margin:32px 0 8px; text-wrap:balance;}
p{margin:11px 0; max-width:74ch;}
.sub{font-size:17.5px; color:var(--mut); margin:0 0 6px; max-width:76ch;}
.meta{color:var(--mut); font-size:13px; margin-top:18px;
  font-variant-numeric:tabular-nums;}
.lede{font-size:16.5px; background:var(--raised);
  border-left:3px solid var(--accent); padding:18px 22px; margin:26px 0;
  border-radius:0 5px 5px 0; max-width:none;}
.lede p{max-width:none}

/* tables ------------------------------------------------------------- */
table{border-collapse:collapse; width:100%; margin:16px 0; font-size:14px;
  font-variant-numeric:tabular-nums;}
th,td{padding:9px 12px; text-align:left; border-bottom:1px solid var(--rule);}
th{background:var(--raised); font-weight:600; font-size:11.5px;
  text-transform:uppercase; letter-spacing:.06em; color:var(--mut);}
td.num,th.num{text-align:right;}
tr.total td{font-weight:600; border-top:1.5px solid var(--rule-strong);}
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch;}

/* figures & cards ----------------------------------------------------- */
figure{margin:24px 0}
img{max-width:100%; height:auto; display:block; border-radius:5px;
  border:1px solid var(--rule); background:#fff;}
figcaption{color:var(--mut); font-size:13px; margin-top:10px; max-width:74ch;}
.grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:14px; margin:22px 0;}
.card{border:1px solid var(--rule); border-radius:7px; padding:16px 18px;
  background:var(--raised); box-shadow:var(--shadow);}
.card .k{font-size:11px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--mut);}
.card .v{font-family:"Source Serif 4",Georgia,serif; font-size:29px;
  font-weight:600; margin:6px 0 3px; font-variant-numeric:tabular-nums;
  line-height:1.05;}
.card .n{font-size:12.5px; color:var(--mut); line-height:1.45;}

/* callouts ------------------------------------------------------------ */
.pill{display:inline-block; padding:2px 9px; border-radius:11px;
  font-size:11px; font-weight:600; letter-spacing:.02em;}
.pass{background:var(--good-bg); color:var(--good)}
.fail{background:var(--warn-bg); color:var(--warn)}
.warnp{background:var(--warn-bg); color:var(--warn)}
.na{background:var(--raised); color:var(--mut)}
.caveat{border-left:3px solid var(--warn); background:var(--warn-bg);
  padding:15px 19px; margin:18px 0; font-size:14.5px;
  border-radius:0 5px 5px 0;}
.good{border-left:3px solid var(--good); background:var(--good-bg);
  padding:15px 19px; margin:18px 0; font-size:14.5px;
  border-radius:0 5px 5px 0;}
.missing{border-left:3px solid var(--mut); background:var(--raised);
  padding:15px 19px; color:var(--mut); font-size:14.5px;
  border-radius:0 5px 5px 0;}
.caveat p,.good p,.missing p,.lede p{margin:8px 0}
code,.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  font-size:13px; background:var(--raised); padding:1px 5px; border-radius:3px;}
.mono{background:none; padding:0;}
ul{margin:10px 0; padding-left:22px; max-width:74ch;} li{margin:5px 0;}
.foot{margin-top:76px; padding-top:20px; border-top:1px solid var(--rule);
  color:var(--mut); font-size:13px; max-width:76ch;}
a{color:var(--accent);}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px;}
@media (prefers-reduced-motion:reduce){*{animation:none!important;
  transition:none!important;}}
@media (max-width:640px){
  .wrap{padding:36px 18px 80px}
  h2{margin-top:46px}
}
"""


def main() -> None:
    src = output_dir() / "report.html"
    if not src.exists():
        print("run scripts/05_report.py first")
        sys.exit(1)
    html = src.read_text(encoding="utf-8")

    # Keep only what sits inside our own <body>; the publisher supplies the rest.
    m = re.search(r"<body>(.*)</body>", html, re.S)
    body = m.group(1) if m else html

    # Every wide table gets its own horizontal scroller so the page body never
    # scrolls sideways on a phone.
    body = re.sub(r"<table>", '<div class="scroll"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)

    out = (
        "<title>Regime Narrative Audit</title>\n"
        f"{FONTS}\n<style>{CSS}</style>\n{body.strip()}\n"
    )
    dest = output_dir() / "artifact.html"
    dest.write_text(out, encoding="utf-8")
    print(f"wrote {dest} ({dest.stat().st_size // 1024} KB)")
    for bad in ("<!doctype", "<html", "<head", "<body"):
        assert bad not in out.lower(), f"{bad} leaked into the artifact file"
    print("no skeleton tags leaked; tables wrapped in scrollers")


if __name__ == "__main__":
    main()
