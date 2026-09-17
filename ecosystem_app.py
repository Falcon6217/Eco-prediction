# Backward-compatible launcher for the original project filename.
exec(compile(open(__file__.replace('ecosystem_app.py','app.py'),encoding='utf-8').read(),'app.py','exec'))
