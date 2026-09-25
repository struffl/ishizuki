# dmgbuild settings for Ishizuki.dmg; build-dmg.sh passes app and background with -D.
import os.path

app = defines["app"]
format = "UDZO"
files = [app]
symlinks = {"Applications": "/Applications"}
background = defines["background"]
window_rect = ((100, 100), (728, 408))
default_view = "icon-view"
show_toolbar = False
show_status_bar = False
show_sidebar = False
icon_size = 96
text_size = 13
icon_locations = {os.path.basename(app): (200, 230), "Applications": (528, 230)}
