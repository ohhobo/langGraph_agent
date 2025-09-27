import matplotlib.font_manager as fm

# 列出所有可用字体
fonts = [f.name for f in fm.fontManager.ttflist]
print("可用字体列表:")
for font in fonts:
    print(font)