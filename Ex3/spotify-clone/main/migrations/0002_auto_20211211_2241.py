# Generated by Django 3.0.7 on 2021-12-11 19:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='playlist_song',
            old_name='song_title',
            new_name='song_id',
        ),
        migrations.RemoveField(
            model_name='playlist_song',
            name='song_albumsrc',
        ),
        migrations.RemoveField(
            model_name='playlist_song',
            name='song_channel',
        ),
        migrations.RemoveField(
            model_name='playlist_song',
            name='song_date_added',
        ),
        migrations.RemoveField(
            model_name='playlist_song',
            name='song_dur',
        ),
        migrations.RemoveField(
            model_name='playlist_song',
            name='song_youtube_id',
        ),
        migrations.AddField(
            model_name='playlist_song',
            name='song_json_data',
            field=models.CharField(default=0, max_length=1000),
            preserve_default=False,
        ),
    ]