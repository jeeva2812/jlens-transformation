CSS = """
:root{--ground:#FBFAF7;--surface:#FFF;--surface2:#F0EEE8;--ink:#1A1815;--ink2:#403C36;
--muted:#726C62;--hair:#DDD8CE;--accent:#0B6E78;--wash:#E2EFF0;--rose:#A83A63;
--rose-wash:#F9E6EC;--amber:#8A6410;--amber-wash:#FBF2DC;--green:#2C6B33;--green-wash:#E3F0E4}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){
--ground:#12110F;--surface:#1A1917;--surface2:#232120;--ink:#EDEAE4;--ink2:#C2BCB2;
--muted:#8C8579;--hair:#302D29;--wash:#12302F;--rose-wash:#341825;--amber-wash:#2C2411;
--green-wash:#152E18}}
:root[data-theme=dark]{--ground:#12110F;--surface:#1A1917;--surface2:#232120;
--ink:#EDEAE4;--ink2:#C2BCB2;--muted:#8C8579;--hair:#302D29;--wash:#12302F;
--rose-wash:#341825;--amber-wash:#2C2411;--green-wash:#152E18}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);margin:0;
font:17px/1.72 "IBM Plex Serif",Georgia,"Times New Roman",serif}
.wrap{max-width:720px;margin:0 auto;padding:52px 24px 140px}
h1{font:600 40px/1.14 "IBM Plex Sans",system-ui,sans-serif;margin:0 0 14px;
letter-spacing:-.028em}
.sub{font:19px/1.55 "IBM Plex Sans",system-ui,sans-serif;color:var(--ink2);
margin:0 0 10px}
.byline{font:14px/1.5 "IBM Plex Sans",system-ui,sans-serif;color:var(--muted);
margin:0 0 44px}
h2{font:600 15px/1.4 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.13em;color:var(--accent);margin:72px 0 6px}
h3{font:600 27px/1.24 "IBM Plex Sans",system-ui,sans-serif;margin:0 0 20px;
letter-spacing:-.018em}
h4{font:600 19px/1.3 "IBM Plex Sans",system-ui,sans-serif;margin:36px 0 10px;
letter-spacing:-.01em}
p{margin:0 0 19px}
.lead{font-size:19px;color:var(--ink2)}
em{font-style:italic}
strong{font-weight:600}
code{font:0.87em "IBM Plex Mono",ui-monospace,monospace;background:var(--surface2);
padding:1px 5px;border-radius:3px}
pre{font:13.5px/1.62 "IBM Plex Mono",ui-monospace,monospace;background:var(--surface2);
padding:15px 17px;border-radius:8px;overflow-x:auto;margin:22px 0}
.aside{background:var(--surface);border:1px solid var(--hair);border-radius:9px;
padding:18px 21px;margin:26px 0;font-size:16px}
.aside .lbl{font:600 10.5px/1 "IBM Plex Mono",ui-monospace,monospace;
text-transform:uppercase;letter-spacing:.11em;color:var(--muted);
display:block;margin-bottom:9px}
.aside p:last-child{margin-bottom:0}
.key{background:var(--wash);border-left:3px solid var(--accent)}
.key .lbl{color:var(--accent)}
.warn{background:var(--amber-wash);border-left:3px solid var(--amber)}
.warn .lbl{color:var(--amber)}
.bad{background:var(--rose-wash);border-left:3px solid var(--rose)}
.bad .lbl{color:var(--rose)}
table{border-collapse:collapse;width:100%;margin:24px 0;
font:14.5px/1.5 "IBM Plex Sans",system-ui,sans-serif}
th{text-align:left;padding:8px 10px;border-bottom:2px solid var(--hair);
font:600 11px/1.3 "IBM Plex Mono",ui-monospace,monospace;text-transform:uppercase;
letter-spacing:.07em;color:var(--muted)}
td{padding:8px 10px;border-bottom:1px solid var(--hair);vertical-align:top}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13.5px}
figure{margin:30px 0}
figure img{width:100%;display:block;border-radius:8px;border:1px solid var(--hair);
background:#fff;cursor:zoom-in}
figcaption{font:14px/1.55 "IBM Plex Sans",system-ui,sans-serif;color:var(--muted);
padding:11px 2px 0}
.toc{background:var(--surface);border:1px solid var(--hair);border-radius:10px;
padding:22px 26px;margin:0 0 50px;font:15px/1.9 "IBM Plex Sans",system-ui,sans-serif}
.toc b{display:block;font:600 10.5px/1 "IBM Plex Mono",ui-monospace,monospace;
text-transform:uppercase;letter-spacing:.11em;color:var(--muted);margin:16px 0 7px}
.toc b:first-child{margin-top:0}
.toc a{display:block;color:var(--ink2);text-decoration:none;padding:1px 0}
.toc a:hover{color:var(--accent)}
.toc a i{color:var(--muted);font-style:normal;font-family:"IBM Plex Mono",monospace;
font-size:12.5px;margin-right:9px}
hr{border:none;border-top:1px solid var(--hair);margin:52px 0}
.num{font-family:"IBM Plex Mono",ui-monospace,monospace}
dialog{border:none;background:rgba(0,0,0,.93);width:100vw;height:100vh;
max-width:100vw;max-height:100vh;padding:0;margin:0}
dialog::backdrop{background:rgba(0,0,0,.9)}
dialog img{width:100%;height:100%;object-fit:contain;cursor:zoom-out}
@media(max-width:640px){.wrap{padding:34px 18px 90px}h1{font-size:31px}h3{font-size:23px}
body{font-size:16.5px}}
"""
