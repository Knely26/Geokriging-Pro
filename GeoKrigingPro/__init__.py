# -*- coding: utf-8 -*-

def classFactory(iface):
    from .plugin import GeoKrigingProPlugin
    return GeoKrigingProPlugin(iface)
