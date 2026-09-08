[app]
title = Lycian OCR
package.name = lycianocr
package.domain = org.lycian
source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,ttf,otf,tflite,json
version = 1.0
requirements = python3,kivy,kivymd,pillow,numpy,opencv,plyer
orientation = portrait
fullscreen = 0
android.permissions = CAMERA,READ_MEDIA_IMAGES
android.minapi = 24
android.api = 35

[buildozer]
warn_on_root = 0
log_level = 2
