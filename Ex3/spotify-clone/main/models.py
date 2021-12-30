from django.db import models


# Create your models here.
class playlist_user(models.Model):
    username = models.CharField(max_length=200)

    def __str__(self):
        return f'Username = {self.username}, Liked Songs = {list(self.playlist_song_set.all())}'


class playlist_song(models.Model):
    user = models.ForeignKey(playlist_user, on_delete=models.CASCADE)
    song_id = models.CharField(max_length=200)
    song_json_data = models.CharField(max_length=1000, default='{"name":"None"}')

    def __str__(self):
        import json
        a = f'Title = {json.loads(self.song_json_data)["name"]}'
        return a
