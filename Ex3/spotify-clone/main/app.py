from django.apps import AppConfig


class PlaylistSongConfig(AppConfig):
    name = 'main'
    def ready(self):
        import main.signals