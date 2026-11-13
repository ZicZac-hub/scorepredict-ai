"""Matchday Studio presentation and URL routing; no forecasting or archive changes."""
from html import escape

PAGES = {
    'home': ('Matchday Studio', 'Live research. Five score doubles. A record you can verify.'),
    'predict': ('Daily Predictions', 'Explore upcoming fixtures and researched exact-score probabilities.'),
    'deep': ('Deep Research AI', 'Recent form, home and away records, head-to-head and available team news.'),
    'doubles': ('Five Score Doubles', 'Select two matches and generate five exact-score combinations.'),
    'archive': ('Prediction History', 'Your original forecasts, timestamps and verified final scores.'),
    'finished': ('Real Performance', 'Actual hits and misses from predictions saved before kickoff.'),
    'more': ('Settings & Methods', 'Understand the connected data, statistical models and their limits.'),
}
MODES = {'auto': 'AI Auto Select', 'manual': 'Manual Select'}


def sync_location(state, params):
    """Apply a newly opened URL once, leaving later widget changes intact."""
    page = params.get('view', 'home')
    page = page if page in PAGES else 'home'
    mode = params.get('mode', '')
    token = (page, mode)
    if state.get('_studio_location') != token:
        state['ziczac_active_page'] = page
        if page == 'doubles' and mode in MODES:
            state['smart_double_mode'] = MODES[mode]
        state['_studio_location'] = token
    state.setdefault('ziczac_active_page', page)


def navigate(state, params, page):
    page = page if page in PAGES else 'home'
    state['ziczac_active_page'] = page
    params['view'] = page
    if page != 'doubles':
        params.pop('mode', None)
    state['_studio_location'] = (page, params.get('mode', ''))


def open_double_mode(state, params, mode):
    if mode not in MODES.values():
        mode = MODES['auto']
    state['smart_double_mode'] = mode
    params['mode'] = 'manual' if mode == MODES['manual'] else 'auto'
    navigate(state, params, 'doubles')


def sync_double_mode(state, params):
    params['mode'] = 'manual' if state.get('smart_double_mode') == MODES['manual'] else 'auto'
    state['_studio_location'] = ('doubles', params['mode'])


def header_html(page):
    title, description = PAGES.get(page, PAGES['home'])
    return ('<header class="studio-header"><div class="studio-brand">'
            '<span class="studio-brand-icon" aria-hidden="true">⚡</span>'
            '<div><strong>ZicZac <span>Predict AI</span></strong>'
            '<small>FOOTBALL INTELLIGENCE</small></div></div>'
            '<span class="studio-version">VERSION 2.0</span>'
            '<div class="studio-route-title"><h1>' + escape(title) + '</h1>'
            '<p>' + escape(description) + '</p></div></header>')


def studio_css():
    return '''<style>
    .block-container{max-width:1280px!important;padding-top:1rem!important}
    [data-testid="stAppViewContainer"]{background:#0a1221!important;background-image:radial-gradient(ellipse at 100% 0,#25315c55,transparent 60%)!important}
    .studio-header{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:22px 26px;margin-bottom:18px;border:1px solid #384c6b;border-radius:16px;background:linear-gradient(120deg,#15253a,#151d33);color:#eff5fc}
    .studio-brand{display:flex;align-items:center;gap:12px}.studio-brand-icon{display:grid;place-items:center;width:43px;height:46px;border:1px solid #3f8e9b;border-radius:12px 4px;color:#7ee1f4;background:#1b3d47;font-size:25px}
    .studio-brand strong{display:block;font-size:23px;line-height:1.2;letter-spacing:-.5px}.studio-brand strong span{color:#8dc9fb}.studio-brand small{display:block;letter-spacing:2px;font-size:9px;color:#94a5bd;margin-top:5px}
    .studio-version{margin-left:auto;font-size:10px;letter-spacing:1px;font-weight:700;border:1px solid #7562a9;background:#352b54;border-radius:6px;color:#dbccff;padding:7px 10px;white-space:nowrap}
    .studio-route-title{width:100%;padding-top:16px;border-top:1px solid #33455f}.studio-route-title h1{margin:0!important;font-size:31px!important;line-height:1.15!important;letter-spacing:-1px!important;color:#f5f8ff!important}.studio-route-title p{font-size:13px!important;color:#a2b7d0;margin:9px 0 0;line-height:1.7}
    .studio-start{border:1px solid #3e5c80;border-radius:16px;padding:25px 27px;margin:5px 0 17px;background:linear-gradient(120deg,#172e4c,#1a2440 70%,#2b2546)}.studio-eyebrow{font-size:10px;font-weight:650;letter-spacing:1.8px;color:#8dbfdf;margin-bottom:12px}.studio-start h2{font-size:34px!important;line-height:1.15!important;letter-spacing:-1px!important;margin:0 0 13px!important}.studio-start h2 span{display:block;color:#81d9f5}.studio-start p{font-size:14px!important;color:#a9c0d7;line-height:1.8;margin:0}
    .zz-section-title{font-size:22px;font-weight:650;color:#edf5ff;margin:22px 0 12px}.zz-match-card{border:1px solid #395574;border-radius:13px;padding:21px;margin-bottom:13px;background:linear-gradient(135deg,#192b41,#121e33);color:#eef5fc}.zz-match-top{font-size:12px;color:#9ab8d1;margin-bottom:11px}.zz-match-teams{text-align:center;font-size:19px;font-weight:600;line-height:1.6}.zz-match-score{display:inline-block;border:1px solid #4f8c9c;border-radius:8px;background:#214650;color:#b3edfa;font-size:25px;padding:5px 19px;margin:13px auto}.zz-match-meta{text-align:center;font-size:12px;color:#a3bbd1;line-height:1.8}
    .stButton button,.stDownloadButton button{min-height:46px!important;border-radius:9px!important;font-size:14px!important;font-weight:600!important;overflow-wrap:anywhere}.stButton button[kind="primary"]{background:linear-gradient(110deg,#66d8ed,#82baf5)!important;color:#112c40!important;border-color:#89cee6!important}.stButton button p{font-size:14px!important;white-space:normal!important;line-height:1.5!important}
    [data-testid="stMetric"]{border-radius:12px!important;background:#15263a!important;border:1px solid #334d69!important;box-shadow:none!important}[data-testid="stMetricLabel"] p{font-size:12px!important;color:#a7bcd3!important}[data-testid="stMetricValue"]{font-size:27px!important}
    [data-testid="stCaptionContainer"] p{color:#97adc5!important;font-size:12px!important;line-height:1.75!important}[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{overflow-wrap:anywhere;line-height:1.75}
    @media(max-width:680px){.block-container{padding-left:13px!important;padding-right:13px!important}.studio-header{padding:18px 16px;gap:12px}.studio-brand{gap:9px}.studio-brand strong{font-size:19px}.studio-brand-icon{width:35px;height:39px;font-size:21px}.studio-brand small{font-size:7px;letter-spacing:1.5px}.studio-version{font-size:8px;padding:6px 7px}.studio-route-title{padding-top:15px}.studio-route-title h1{font-size:27px!important}.studio-route-title p{font-size:12px!important}.studio-start{padding:23px 18px}.studio-start h2{font-size:29px!important}.studio-start p{font-size:13px!important}.studio-eyebrow{font-size:9px;letter-spacing:1.2px}.stButton button{min-height:48px!important}.stButton button p{font-size:13px!important}.zz-match-card{padding:17px 13px}.zz-match-teams{font-size:17px}.zz-match-top,.zz-match-meta{font-size:11px}.zz-section-title{font-size:21px}.studio-route-title,.studio-brand{min-width:0}[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:min(135px,100%)!important;max-width:100%}[data-testid="stDataFrame"]{max-width:100%}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
    </style>'''
