import os.path
import site

PACKAGE = "holoviews"

PATH = os.path.join(site.getsitepackages()[0], f"_{PACKAGE}.pth")

if not os.path.exists(PATH):
    with open(PATH, "w") as f:
        f.write(os.getcwd())
