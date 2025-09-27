import matplotlib.font_manager as fm

# List available fonts
fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
font_names = [fm.FontProperties(fname=f).get_name() for f in fonts[:10]]  # Limit to first 10 for brevity
print("Sample of available fonts (first 10):")
for name in font_names:
    print(name)